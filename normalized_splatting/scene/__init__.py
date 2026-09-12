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

import os
import random
import json
import torch
from ..utils.system_utils import searchForMaxIteration
from .dataset_readers import SceneInfo, sceneLoadTypeCallbacks
from .gaussian_model import GaussianModel
from normalized_splatting.arguments import ModelParams
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, 
                 resolution_scales=[1.0], pose_trans_noise=0, single_frame_id=None, pose_lr=None,
                 pose_lr_final=0.02, pose_lr_max_steps=300,
                 depth_trunc=1000, skip_train=False,
                 min_depth=None, image_stride=1, saga_iteration=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.loaded_saga_iter = None
        self.scale_gate = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            if saga_iteration is not None:
                if saga_iteration == -1:
                    self.loaded_saga_iter = searchForMaxIteration(
                        os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}"))
                else:
                    self.loaded_saga_iter = saga_iteration
                print("Loading trained SAGA model at scene iteration {}, saga iteration {}".format(
                    self.loaded_iter, self.loaded_saga_iter))
            else:
                print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        scene_info: SceneInfo = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Found sparse folder, assumign COLMAP data")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "traj.txt")):
            print("Found traj.txt file, assuming Replica data set!")
            scene_info = sceneLoadTypeCallbacks["Replica"](args.source_path, args.eval, pose_trans_noise=pose_trans_noise,
                                                           single_frame_id=single_frame_id, depth_trunc=depth_trunc)
        elif os.path.exists(os.path.join(args.source_path, "data.pickle")) or os.path.exists(os.path.join(args.source_path, "data.pickle.xz")):
            print("found data.pickle file, assuming PyBullet data!")
            scene_info = sceneLoadTypeCallbacks["PyBullet"](args.source_path)
        elif os.path.exists(os.path.join(args.source_path, "poses.json")):
            print(f"Found poses.json file, assuming ROSBag (extracted) data. Trucating depth: {depth_trunc}")
            scene_info = sceneLoadTypeCallbacks["ROSBag"](args.source_path, depth_trunc=depth_trunc, min_depth=min_depth, eval=args.eval)
        elif os.path.exists(os.path.join(args.source_path, "groundtruth.txt")):
            print("Found groundtruth.txt file, assuming TUM data set!")
            scene_info = sceneLoadTypeCallbacks["TUM"](args.source_path, voxel_size=0.025, eval=args.eval, target_images_total=250)
        elif os.path.exists(os.path.join(args.source_path, "kf_trajectory.txt")):
            print("found kf_trajectory.txt, assuming ORB_SLAM data")
            scene_info = sceneLoadTypeCallbacks["ORB_SLAM"](args.source_path, args.eval, depth_trunc=depth_trunc)
        elif os.path.exists(os.path.join(args.source_path, "dslr")):
            print("Found DSLR Directory. Assuming ScanNet++ dataset."
                    "Make sure you've undistorted the images and rendered depth with use_undistorted_transform: True")
            scene_info = sceneLoadTypeCallbacks["ScanNet++"](args.source_path, args.eval)
        else:
            raise ValueError(
                f"Could not recognize the scene type at {args.source_path}. Expected one of: "
                "sparse/ (COLMAP), transforms_train.json (Blender), traj.txt (Replica), "
                "data.pickle[.xz] (PyBullet), poses.json (ROSBag), "
                "groundtruth.txt (TUM), kf_trajectory.txt (ORB-SLAM), dslr/ (ScanNet++)."
            )

        if image_stride > 1:
            num_before = len(scene_info.train_cameras)
            scene_info.train_cameras[:] = scene_info.train_cameras[::image_stride]
            print(f"Downsampled training cameras by stride {image_stride}: {num_before} -> {len(scene_info.train_cameras)}")

        if args.scale_to_cube:
            self.scale = scene_info.scale_to_cube(args.cube_scale)
            print(f"Scaling data to cube (size={args.cube_scale}). Scale={self.scale:03f}")
        else:
             self.scale = 1.0
             
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if skip_train:
                self.train_cameras[resolution_scale] = []
            else:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            if pose_lr is not None:
                # Rebuild the optimizer+schedule; patching param_group['lr'] alone
                # had no effect, since update_learning_rate() overwrites it from
                # the schedule on the very next step.
                for cam in self.train_cameras[resolution_scale]:
                    cam.setup_optimizer(pose_lr=pose_lr,
                                        pose_lr_final=pose_lr_final,
                                        pose_lr_max_steps=pose_lr_max_steps)
         
        if self.loaded_iter:
            if self.loaded_saga_iter:
                saga_path = os.path.join(self.model_path,
                                         "point_cloud",
                                         "iteration_" + str(self.loaded_iter),
                                         f"saga_{self.loaded_saga_iter}")
                self.gaussians.load_ply(os.path.join(saga_path, "point_cloud.ply"))
                self.scale_gate = torch.nn.Sequential(
                    torch.nn.Linear(1, 32, bias=True),
                    torch.nn.Sigmoid()
                ).cuda()
                self.scale_gate.load_state_dict(torch.load(os.path.join(saga_path, "scale_gate.pt")))
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                     "point_cloud",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "point_cloud.ply"))
            self.gaussians.set_scale_factor(self.scale)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,
                                           filter_radius_3d=args.filter_radius_3d, 
                                           opacity_scale=args.opacity_scale,
                                           filter_radius_2d=args.filter_radius_2d,
                                           normalized=args.normalize_gaussians,
                                           num_app_opt_cameras=len(scene_info.train_cameras),
                                           scale_factor=self.scale)

    def save(self, iteration, viewpoint_camera=None, saga_iteration=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), viewpoint_camera=viewpoint_camera, saga_iteration=saga_iteration)
    
    def save_app_opt(self, iteration, viewpoint_camera=None, viewpoint_id=-1):
        # mkdir_p(os.path.dirname(output_path))
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{viewpoint_id}.ply"), viewpoint_camera=viewpoint_camera)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]