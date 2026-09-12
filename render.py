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

import torch
from normalized_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from normalized_splatting.gaussian_renderer import render
import torchvision
from normalized_splatting.utils.camera_utils import loadCam
from normalized_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from normalized_splatting.arguments import ModelParams, PipelineParams, get_combined_args, set_normalization_defaults
from normalized_splatting.scene.dataset_readers import CameraInfo
from normalized_splatting.utils.graphics_utils import focal2fov
from normalized_splatting.gaussian_renderer import GaussianModel
from normalized_splatting.utils.loss_utils import ssim
from normalized_splatting.utils.image_utils import psnr
import json
import numpy as np
import cv2
from PIL import Image


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, compute_metrics=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    


    if len(views) > 0 and views[0].depth is not None:
        gts_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
        makedirs(gts_depth_path, exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    if compute_metrics:
        ssims = []
        psnrs = []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", dynamic_ncols=True)):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        depth = results["depth"]
        
        # depth_np = depth.detach().cpu().squeeze().numpy()
        
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        if compute_metrics:
            ssims.append(ssim(rendering.cuda(), gt.cuda()).item())
            psnrs.append(psnr(rendering.cuda(), gt.cuda()).mean().item())

        if view.depth is not None:
            gt_depth = view.depth
            gt_depth = gt_depth / (gt_depth.max() + 1e-5)
            torchvision.utils.save_image(gt_depth, os.path.join(gts_depth_path, '{0:05d}'.format(idx) + ".png"))




    if compute_metrics:
        mean_ssim = torch.tensor(ssims).mean().item()
        mean_psnr = torch.tensor(psnrs).mean().item()
        
        if mean_ssim == mean_ssim and  mean_psnr == mean_psnr:
        
            print(f"SSIM: {mean_ssim}, PSNR: {mean_psnr}")
        
            metric_path = os.path.join(model_path, name,"ours_{}".format(iteration), "metrics.json")
            
            metric_dict = {name: {
                    "ssim": mean_ssim,
                    "psnr": mean_psnr,
                }
            }
            
            with open(metric_path, 'a+') as f:
                json.dump(metric_dict, f)
        
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
                compute_metrics: bool = False):
    
    # with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train=skip_train)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
    if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, compute_metrics)

    if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, compute_metrics)

@torch.no_grad()
def render_from_calib_file(dataset : ModelParams, iteration : int, pipeline : PipelineParams, calib_file: str, pose_file: str = ""):
    save_path = os.path.join(dataset.model_path, "render_from_calib", "ours_{}".format(iteration))

    makedirs(save_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with open(calib_file, 'r') as json_calib:
        calib_data = json.load(json_calib)["intrinsics"]
    
    if pose_file is not None:
        with open(pose_file, 'r') as pose_f:
            pose_data = np.array(json.load(pose_f))
        calib_data["pose"] = pose_data

    pose = np.linalg.inv(np.array(calib_data["pose"]))
    R = pose[:3, :3].T
    T = pose[:3, 3]

    K = np.array(calib_data["K"])

    if "dist_coeffs" in calib_data:
        D = np.array(calib_data["dist_coeffs"])
        if np.abs(D).sum() > 1e-3:
            print("undistorting")
            K = cv2.getOptimalNewCameraMatrix(K, D, (calib_data["width"], calib_data["height"]), 0)[0]
    
    fx = K[0,0]
    fy = K[1,1]
    
    width = calib_data["width"]
    height = calib_data["height"]

    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    image = Image.fromarray(np.zeros((height, width, 3)).astype(np.uint8))
    image_name = "dummy_image"

    cam_info = CameraInfo(
        0, R, T, FovY, FovX, image, None, image_name, width, height, None, R, T, K=K)
    
    view = loadCam(dataset, 0, cam_info, 1.)
    print("K:", view.K)
    results = render(view, gaussians, pipeline, background)
    rendering = results["render"]
    depth = results["depth"]

    torchvision.utils.save_image(rendering, os.path.join(save_path, "render.png"))

    depth_np = depth.cpu().numpy()
    np.save(os.path.join(save_path, "depth.npy"), depth_np)

    depth = depth / (depth.max() + 1e-5)
    torchvision.utils.save_image(depth, os.path.join(os.path.join(save_path, "depth.png")))


@torch.no_grad()
def render_video_from_json(dataset : ModelParams, iteration : int, pipeline : PipelineParams, json_trajectory_path: str):
    """Render a camera trajectory described by a JSON file.

    The file must provide ``render_width``, ``render_height`` and either:

    * ``poses``: a list of 6-vectors ``[cx, cy, cz, lx, ly, lz]`` giving the camera
      position and the point it looks at, or
    * ``camera_path``: a list of ``{"camera_to_world": [16 floats, row-major]}``
      entries (nerfstudio convention).

    An optional top-level ``fov`` (vertical, degrees) defaults to 50.
    """
    def normalize(v):
        return v / np.linalg.norm(v)

    def compute_pose(camera_position, look_at_position, up_vector):
        forward = normalize(look_at_position - camera_position)
        right = normalize(np.cross(up_vector, forward))
        up = np.cross(forward, right)

        # Construct the rotation matrix (3x3)
        rotation_matrix = np.column_stack((right, up, -forward))

        # Construct the pose matrix (4x4)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = camera_position

        return pose_matrix

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    save_path = os.path.join(dataset.model_path, "render_from_json_trajectory", "ours_{}".format(scene.loaded_iter))

    makedirs(f"{save_path}/depth", exist_ok=True)
    makedirs(f"{save_path}/renders", exist_ok=True)


    # Open and load the JSON trajectory file
    with open(json_trajectory_path, 'r') as json_file:
        trajectory_data = json.load(json_file)

    # Get render width and height
    render_width = trajectory_data['render_width']
    render_height = trajectory_data['render_height']
    fovy_deg = trajectory_data.get('fov', 50)

    if 'poses' in trajectory_data:
        poses = [np.asarray(p, dtype=np.float64) for p in trajectory_data['poses']]
    elif 'camera_path' in trajectory_data:
        poses = [np.asarray(f['camera_to_world'], dtype=np.float64).reshape(4, 4)
                 for f in trajectory_data['camera_path']]
    else:
        raise ValueError(
            f"{json_trajectory_path} has neither a 'poses' nor a 'camera_path' entry")

    fovy = np.radians(fovy_deg)
    aspect_ratio = render_width / render_height
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)

    # Iterate over the camera path in the JSON
    for idx, pose_spec in enumerate(tqdm(poses, dynamic_ncols=True)):
        if pose_spec.shape == (4, 4):
            pose = pose_spec
        else:
            pose = compute_pose(pose_spec[:3], pose_spec[3:], [0, -1, 0])
        R = pose[:3, :3].T
        T = pose[:3, 3]

        # Create a blank image to render
        image = Image.fromarray(np.zeros((render_height, render_width, 3)).astype(np.uint8))
        image_name = "dummy_image_{}".format(idx)

        # Create CameraInfo object for rendering
        cam_info = CameraInfo(
            idx, R, T, fovx, fovy, image, None, image_name, render_width, render_height, None, R, T)

        # Load camera view
        view = loadCam(dataset, idx, cam_info, 1.0)

        # Perform rendering
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        depth = results["depth"]

        # Save the rendered image and depth
        torchvision.utils.save_image(rendering, os.path.join(save_path, "renders", f"{idx:05d}.png"))

        depth_normalized = depth / (depth.max() + 1e-5)
        torchvision.utils.save_image(depth_normalized, os.path.join(save_path, "depth", f"{idx:05d}.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_from_calib_file", type=str, default="")
    parser.add_argument("--pose_file", type=str, default="")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compute_metrics", action="store_true")
    parser.add_argument("--render_video_from_json", type=str, default="")

    args = get_combined_args(parser)
    set_normalization_defaults(args)
    

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    if len(args.render_video_from_json) > 0:
        render_video_from_json(model.extract(args), args.iteration, pipeline.extract(args), args.render_video_from_json)

    if len(args.render_from_calib_file)>0:
        render_from_calib_file(model.extract(args), args.iteration, pipeline.extract(args), args.render_from_calib_file, args.pose_file)
    else:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                    args.skip_train, args.skip_test, args.compute_metrics)