#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import datetime
import os
import torch
from random import randint
from normalized_splatting.utils.loss_utils import l1_loss, ssim
from normalized_splatting.gaussian_renderer import render
import sys
from normalized_splatting.scene import Scene, GaussianModel 
from normalized_splatting.utils.general_utils import safe_state
from tqdm import tqdm
from normalized_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from normalized_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, set_normalization_defaults, check_normalization_requirements
from logger import WandbWriter
from torchvision.utils import make_grid
from normalized_splatting.utils.results_utils import save_rendered_images

SAVING_DIR = None

def training(dataset, opt: OptimizationParams, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    app_opt = args.app_opt
    # testing_iterations = list(range(1000, 30001, 2000)) + [30000]
    # saving_iterations = list(range(1000, 30001, 2000)) + [30000]

    wandb_writer = prepare_output_and_logger(dataset, args)

    # WandbWriter is not callable; log() is the entry point.
    wandb_log = wandb_writer.log if wandb_writer is not None else (lambda x: None)

    gaussians = GaussianModel(dataset.sh_degree, app_opt)
    scene = Scene(dataset,
                  gaussians,
                  pose_trans_noise=args.pose_trans_noise,
                  single_frame_id=args.single_frame_id,
                  pose_lr=opt.pose_lr,
                  pose_lr_final=opt.pose_lr_final,
                  pose_lr_max_steps=opt.pose_lr_max_steps,
                  depth_trunc=args.depth_trunc,
                  shuffle = not opt.fix_order,
                  min_depth=args.min_depth,
                  image_stride=args.image_stride)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if gaussians.app_opt:
        print("Running Appearance Optimization!!")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    gaussians.reset_opacity()
    

    rand_order = None
    fixed_iteration_counter = 0

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if opt.fix_order and rand_order is None:
                num_cameras = len(viewpoint_stack)
                gen = torch.Generator()
                gen.manual_seed(42)
                
                rand_order = torch.randperm(num_cameras, generator=gen)
                
        if opt.fix_order:
            viewpoint_cam = viewpoint_stack[rand_order[fixed_iteration_counter]]
            fixed_iteration_counter += 1
            if fixed_iteration_counter >= num_cameras:
                fixed_iteration_counter = 0
        
        else:    
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_depth = viewpoint_cam.depth / gaussians.scale_factor
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            viewpoint_color=viewpoint_cam.uid if app_opt else -1)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        if args.min_depth is not None:
            image[:, gt_depth < args.min_depth/scene.scale] = 0
            gt_image[:, gt_depth < args.min_depth/scene.scale] = 0
        
        Ll1 = l1_loss(image, gt_image)
        color_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = color_loss
        
        if gt_depth is not None and args.DS:    
            gt_depth = gt_depth.cuda()
            
            depth[0][gt_depth == 0] = 0
            depth[0][gt_depth > args.depth_trunc/scene.scale] = 0
            gt_depth[gt_depth > args.depth_trunc/scene.scale] = 0
            
            if args.min_depth is not None:
                depth[0][gt_depth < args.min_depth/scene.scale] = 0
                gt_depth[gt_depth < args.min_depth/scene.scale] = 0
            
            loss += opt.lambda_depth * l1_loss(depth, gt_depth)
        
        # from PhysGaussians
        if opt.anisotropy_ratio > 0:
            max_scales = gaussians.get_scaling.max(dim=1, keepdim=True).values
            min_scales = gaussians.get_scaling.min(dim=1, keepdim=True).values            
            anisotropy_loss = torch.mean(torch.max(max_scales/min_scales, torch.full_like(max_scales, opt.anisotropy_ratio)) - opt.anisotropy_ratio)
            loss += opt.lambda_anisotropy * anisotropy_loss
            
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                wandb_log({"Loss": loss.item()})
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(wandb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), args.output_dir)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if gaussians.app_opt:
                    viewpoint_stack = scene.getTrainCameras().copy()
                    scene.save(iteration=iteration)
                    # for i in range(0,len(viewpoint_stack),1):
                    #     viewpoint = viewpoint_stack[i]
                    #     scene.save_app_opt(iteration=iteration, viewpoint_camera=viewpoint, viewpoint_id=viewpoint.uid)
                else:
                    scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_thresh = 1 if gaussians.normalized else 20
                    size_threshold = size_thresh if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity * gaussians._opacity_scale.item(), scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if not args.localization:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
                ## Optimizers for the SH model
                if gaussians.app_opt:
                    for optimizer in gaussians.app_optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none = True)

            if args.BA or args.localization:
                viewpoint_cam.optimizer.step()
                viewpoint_cam.optimizer.zero_grad(set_to_none = True)
                viewpoint_cam.update_learning_rate(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(dataset, args):
    
    if dataset.model_path is None or len(dataset.model_path) == 0:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            now = datetime.datetime.now()
            unique_str = now.strftime("%m%d%y_%H%M%S")
        dataset.model_path = os.path.join(args.output_dir, unique_str)

    global SAVING_DIR
    SAVING_DIR = dataset.model_path
    

    if args.wandb:
        wandb_op = os.path.join(args.output_dir, 'wandb')
        tags = args.source_path.split('/')[-2:]
        tags.reverse()
        if args.normalize:
            tags.append('normalized')
        else:
            tags.append('unnormalized')
        tags.reverse()
        run_name = '/'.join(tags)
        wandb_writer = WandbWriter(project_name=args.wandb_project, run_name=run_name, tags=tags, output_dir=wandb_op)
    else:
        wandb_writer = None

    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    print("Optimizing " + dataset.model_path)

    return wandb_writer


def training_report(wandb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, output_dir):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        images_list  = []
        gt_images_list  = []
        psnr_test = None
        for config in validation_configs:

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for viewpoint in config['cameras']:
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    images_list.append(image)
                    gt_images_list.append(gt_image)
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        
        if not images_list:
            # No train or test cameras were available at this iteration.
            torch.cuda.empty_cache()
            return

        images_list = torch.stack(images_list)
        images_grid = make_grid(images_list).permute(1, 2, 0).cpu().numpy()
        grid_caption = "iteration: {}, PSNR: {}".format(iteration, psnr_test)
        heading = "Rendered Images"

        gt_images_list = torch.stack(gt_images_list)
        gt_images_grid = make_grid(gt_images_list).permute(1, 2, 0).cpu().numpy()
        gt_grid_caption = ""
        gt_heading = "GT Images"

        if wandb_writer:
            wandb_writer.log_image_grid(images_grid, grid_caption, heading)
            wandb_writer.log_image_grid(gt_images_grid, gt_grid_caption, gt_heading)
            wandb_writer.log({"PSNR": psnr_test})
        else:
            global SAVING_DIR
            render_op_dir = os.path.join(SAVING_DIR, 'renders', f"{iteration}")
            save_rendered_images(images_list=images_list, iteration=iteration, psnr_test=psnr_test, output_dir=render_op_dir)
            gt_op_dir = os.path.join(SAVING_DIR, 'gt', f"{iteration}")
            save_rendered_images(images_list=gt_images_list, iteration=iteration, psnr_test=psnr_test, output_dir=gt_op_dir)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--pose_trans_noise', type=float, default=0.0)
    parser.add_argument("--DS", action="store_true", default=False)
    parser.add_argument("--BA", action="store_true", default=False)
    parser.add_argument("--localization", action="store_true", default=False)
    parser.add_argument('--single_frame_id', type=int, default=None)

    parser.add_argument("--depth_trunc", type=float, default=1000., help="for rgb-d datasets, sets the max range of the depth sensor.")
    parser.add_argument("--output_dir", type=str, default="./output", help="parent_dir for the output")
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="normalized_splatting")

    parser.add_argument("--normalize", action="store_true", help="If set, sets all settings for the default normalization"
                         "scheme (opacity_scale = 0.001, filter_radius_2d=0.0, filter_radius_3d=1e-6)")
    parser.add_argument("--force-no-DS", dest="force_no_DS", action="store_true", default=False,
                        help="Allow --normalize without --DS. Untested and unsupported.")
    parser.add_argument("--app_opt", action="store_true", help="Enable appearance optimization")
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--image_stride", type=int, default=1, help="Keep every Nth training image (e.g. 2 keeps half the images)")


    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    check_normalization_requirements(args)
    set_normalization_defaults(args)
    if args.normalize:
        print("Running Normalized Splattiing")
    # Initialize system state (RNG)
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args)

    # All done
    print("\nTraining complete.")


