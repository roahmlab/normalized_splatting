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

from dataclasses import dataclass
import os
import sys
from PIL import Image
from typing import List

import pandas as pd
import torch
import tqdm
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from ..utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import SH2RGB
from .gaussian_model import BasicPointCloud
import glob
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    R_gt: np.array
    T_gt: np.array
    features: torch.tensor = None
    masks: torch.tensor = None
    mask_scales: torch.tensor = None
    K: np.array = None

@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str
    gaussian_init: dict = False
    scale_factor: float = 1
    
    def dump_to_pcds(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        def save_points(points, filename, colors=None):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(output_dir, filename), pcd)

        save_points(self.point_cloud.points, "points.pcd", colors=self.point_cloud.colors)

        train_centers = np.vstack([-cam.R @ cam.T for cam in self.train_cameras])
        save_points(train_centers, "train_cameras.pcd")

        if len(self.test_cameras) > 0:
            test_centers = np.vstack([-cam.R @ cam.T for cam in self.test_cameras])
            save_points(test_centers, "test_cameras.pcd")
    
    def scale_to_cube(self, cube_size=5):

        all_ts = [cam.T for cam in self.train_cameras + self.test_cameras]
        all_ts = np.vstack(all_ts)

        min_corner = np.min(all_ts, axis=0)
        max_corner = np.max(all_ts, axis=0)
        scale = np.max(max_corner - min_corner) / cube_size
        
        return scale
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_saga_info(data_root, img_stem):
    mask_path = f"{data_root}/saga/sam_masks/{img_stem}.pt"
    scale_path = f"{data_root}/saga/mask_scales/{img_stem}.pt"
    features_path = f"{data_root}/saga/clip_features/{img_stem}.pt"
    
    maybe_load = lambda path : torch.load(path) if os.path.exists(path) else None
    
    mask = maybe_load(mask_path)
    scale = maybe_load(scale_path)
    features = maybe_load(features_path)
    
    return features, mask, scale
    
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder, scale_factor=1):
    cam_infos = []
    print('images_folder: ', images_folder)
    print('depths_folder: ', depths_folder)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) # R_colmap.T !!!!!!
        T = np.array(extr.tvec) * scale_factor

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if depths_folder is not None:
            depth_extension = os.listdir(depths_folder)[0].split(".")[-1]
            depth_path = os.path.join(depths_folder, os.path.basename(extr.name[:-3]+depth_extension)) # just a hack
            if depth_extension in ["np", "npy"]:
                depth = np.load(depth_path)
                depth = Image.fromarray(depth)
            else:
                depth = Image.open(depth_path)
        else:
            depth = None
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=depth,
                              R_gt = R, T_gt=T)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def build_tum_poses_from_df(df: pd.DataFrame, zero_origin=False):
    data = torch.from_numpy(df.to_numpy(dtype=np.float64))

    ts = data[:,0]
    xyz = data[:,1:4]
    quat = data[:,4:]

    rots = torch.from_numpy(Rotation.from_quat(quat).as_matrix())
    
    poses = torch.cat((rots, xyz.unsqueeze(2)), dim=2)

    homog = torch.Tensor([0,0,0,1]).tile((poses.shape[0], 1, 1)).to(poses.device)

    poses = torch.cat((poses, homog), dim=1)

    if zero_origin:
        rot_inv = poses[0,:3,:3].T
        t_inv = -rot_inv @ poses[0,:3,3]
        start_inv = torch.hstack((rot_inv, t_inv.reshape(-1, 1)))
        start_inv = torch.vstack((start_inv, torch.tensor([0,0,0,1.0], device=start_inv.device)))
        poses = start_inv.unsqueeze(0) @ poses

    return poses.float(), ts

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    if os.path.exists(f"{path}/sparse/0/scale_factor.txt"):
        with open(f"{path}/sparse/0/scale_factor.txt") as f:
            scale_factor = float(f.read().strip())
    else:
        scale_factor = 1
    
    print("Scale Factor is", scale_factor)
    
    reading_dir = "images" if images is None else images
    if depths is None:
        depth_dir = None 
    elif os.path.exists(f"{path}/depths"):
        depth_dir = "depths"
        print("depth_dir is", depth_dir)
    elif os.path.exists(f"{path}/depth"):
        depth_dir = "depth"
        print("depth_dir is", depth_dir)
    else:
        print("Depths was not None, but couldn't find anything! not using depths")
        depth_dir = None

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           depths_folder=None if depth_dir is None else os.path.join(path, depth_dir),
                                           scale_factor=scale_factor)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        xyz *= scale_factor
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    ### viz ###
    # -----
    # tf_colmap = tf inverse
    # -----
    # R_gs = R
    # T_gs = T inverse
    # -----

    viz_list=[]
    for cam_info in cam_infos[:20]:
        t = cam_info.T
        R = cam_info.R # cam_info.R = R_colmap.T (done by readColmapCameras)
        # invert t
        t = -R @ t 

        mat = np.identity(4)
        mat[:3,:3] = R
        mat[:3,3] = t
        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis_mesh.scale(0.4, center=axis_mesh.get_center())
        mesh = axis_mesh.transform(mat)
        viz_list.append(mesh)

    axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    viz_list.append(axis_mesh)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)  
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    viz_list.append(o3d_pcd)
    # o3d.visualization.draw_geometries(viz_list)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"]
            if file_path[0] == "." and file_path[1] != ".":
                file_path = file_path[1:]
                if file_path[0] == "/":
                    file_path = file_path[1:]

            cam_name = os.path.join(path, file_path + extension)
            depth_name = os.path.join(path, file_path + + '.png')

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            
            try:
                depth = Image.open(depth_name)
            except:
                depth = None
            
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], depth=depth,
                            R_gt=R, T_gt=T))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readPyBulletData(path):
    import lzma
    import pickle
    train_cam_infos = []
    
    data_dir = path
    data_path = f"{data_dir}/data.pickle"
    compr_data_path = f"{data_path}.xz"

    if os.path.exists(data_path):
        with open(data_path, 'rb') as data_pkl:
            data = pickle.load(data_pkl)
    elif os.path.exists(compr_data_path):
        with lzma.open(compr_data_path, 'rb') as data_pkl:
            data = pickle.load(data_pkl)
    else:
        raise Exception("Can't find the data path (or compressed variant) at ", data_path)
    
    img_data = []
    np.set_printoptions(precision=4)
    
    for i in range(len(data['rgb'])):
        
        cam_name = "BASECAM" if i == 0 else f"CAMERA{i-1}"
        pose = data['poses'][cam_name].copy()
        # print(pose)
        
        # pose[:3, :3] = pose[:3, :3] @ np.diag([1,-1,-1])
        pose = pose @ np.diag([1,-1,-1,1])

        rgb = (data['rgb'][i]).astype(np.uint8)

        rgb = Image.fromarray(rgb, "RGB")
        depth = Image.fromarray(data['depth'][i])

        img_data.append((pose, rgb, depth))
    fovy = float(np.pi) / 3.
    fovx = focal2fov(fov2focal(fovy, rgb.size[1]), rgb.size[0])
    
    for idx, (pose, rgb, depth) in enumerate(img_data):
        R = pose[:3, :3]
        T = pose[:3, 3]

        train_cam_infos.append(CameraInfo(uid=idx, R=R, T=-R.T @ T,FovX=fovx, FovY=fovy, image=rgb,
            image_path=None, image_name=f"rgb_{idx}", width=rgb.size[0], height=rgb.size[1],
            R_gt=R, T_gt=T, depth=depth))
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    focalx = fov2focal(fovx, rgb.size[0])
    focaly = fov2focal(fovy, rgb.size[1])
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb.size[0], rgb.size[1], focalx, focaly, rgb.size[0]/2, rgb.size[1]/2)
    
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))
    
    ply_path = os.path.join(data_dir, "points3d.ply")
    if True: # not os.path.exists(ply_path):
        for idx, (pose, rgb, depth) in enumerate(img_data):
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(rgb).astype(np.uint8))
            
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1.0, depth_trunc=254, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            o3d_pc = o3d_pc.voxel_down_sample(0.05)
            o3d_pc = o3d_pc.transform(pose)
            
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)), axis=0)
            
        # downsample
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)
        o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)
        o3d_pcd = o3d_pcd.voxel_down_sample(0.05)

        pc_init = np.asarray(o3d_pcd.points)
        color_init = np.asarray(o3d_pcd.colors)
  
        num_pts = pc_init.shape[0]
        xyz = pc_init

        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz
        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
    else:
        pcd = fetchPly(ply_path)
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readReplicaInfo(path, eval, extension=".png", pose_trans_noise=0, single_frame_id=None, depth_trunc=1000):
    traj_file = os.path.join(path, 'traj.txt')
    with open(traj_file, 'r') as poses_file:
        poses = poses_file.readlines()
    
    image_paths = sorted(glob.glob(os.path.join(path, 'images/*')))
    depth_paths = sorted(glob.glob(os.path.join(path, 'depths/*')))
    
    if len(depth_paths) == 0:
        depth_paths = [None] * len(image_paths)
    
    cam_infos = []
    mat_list=[]
    viz_list=[]
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))
    
    for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        if (single_frame_id is not None) and (idx is not single_frame_id):
            continue
            
        mat = np.array(poses[idx].strip().split('\n')[0].strip().split(' ')).reshape((4,4)).astype('float64')
        mat_list.append(mat)

        R = mat[:3,:3]
        T = mat[:3, 3]

        R_gt=R.copy()
        T_gt=T.copy()
        
        # Add noise to poses
        if pose_trans_noise > 0:
            np.random.seed(0)
            noise = np.random.rand(3) * pose_trans_noise
            T += noise

        # Invert
        T = -R.T @ T # convert from real world to GS format: R=R, T=T.inv()
        T_gt = -R_gt.T @ T_gt # convert from real world to GS format: R=R, T=T.inv()

        height = 680
        width = 1200
        focal_length_x = 600
        focal_length_y = 600
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        if depth_path is None:
            depth = None
        else:
            depth = Image.fromarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 6553.5)

        features, mask, scale = load_saga_info(path,image_name)
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, 
                              height=height, depth=depth, R_gt=R_gt, T_gt=T_gt,
                              features=features, masks=mask, mask_scales=scale)
        cam_infos.append(cam_info)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    else:    
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
            mat = np.array(poses[idx].split('\n')[0].split(' ')).reshape((4,4)).astype('float64')
            image = Image.open(image_path)
            depth = Image.open(depth_path)
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            # Replica dataset cam: H: 680 W: 1200 fx: 600.0 fy: 600.0 cx: 599.5 cy: 339.5 png_depth_scale: 6553.5
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)
            # o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d_depth, intrinsic=o3d_intrinsic, extrinsic=np.identity(4), depth_scale=6553.5, stride=50)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=6553.5, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            dist = np.linalg.norm(np.asarray(o3d_pc.points), axis=1)
    
            o3d_pc = o3d_pc.transform(mat)
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)[::20]), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)[::20]), axis=0)
            
        # downsample
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)
        o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)
        o3d_pcd = o3d_pcd.voxel_down_sample(0.05)
        pc_init = np.asarray(o3d_pcd.points)
        color_init = np.asarray(o3d_pcd.colors)

        num_pts = pc_init.shape[0]
        xyz = pc_init

        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz

        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def create_coordinate_frame(position, rotation_matrix, size=1.0):
    # Create a coordinate frame at the origin with the given size
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    # Create a transformation matrix from rotation matrix and position
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    
    # Apply the transformation
    mesh_frame.transform(transform)
    
    return mesh_frame

def detectDepthUnitScale(depth_paths):
    """Detect whether depth files store millimeters instead of meters.
    Returns the divisor that converts the stored values to meters."""
    percentiles = []
    integer_dtype = False
    for depth_path in depth_paths:
        depth = np.load(depth_path)
        integer_dtype = integer_dtype or np.issubdtype(depth.dtype, np.integer)
        nonzero = depth[depth > 0]
        if nonzero.size > 0:
            percentiles.append(np.percentile(nonzero, 95))
    # Depths in meters rarely reach 100; mm data sits in the hundreds/thousands.
    if percentiles and (integer_dtype or float(np.median(percentiles)) > 100):
        return 1000.0
    return 1.0

def readROSData(path, depth_trunc=10, min_depth = None, eval=False):
    json_path = f"{path}/poses.json"
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    poses = []

    np.random.seed(0)

    cam_infos = []

    frames = json_data['frames']

    sample_idxs = np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int)
    depth_unit_scale = detectDepthUnitScale([f"{path}/{frames[i]['depth_file_path']}" for i in sample_idxs])
    if depth_unit_scale != 1.0:
        print(f"Depth appears to be in millimeters, dividing by {depth_unit_scale} to convert to meters")
    
    pose_mats = []
    
    if min_depth is not None:
        print("Min Depth:", min_depth)

    for idx in tqdm.trange(0, len(frames), dynamic_ncols=True):
        frame = frames[idx]
        K = np.array(frame["K"])
        depth_path = f"{path}/{frame['depth_file_path']}"
        rgb_path = f"{path}/{frame['color_file_path']}"

        T_world_to_cam = np.linalg.inv(np.array(frame["transform_matrix"]))

        with Image.open(rgb_path) as im:
            rgb_image = im.copy()
            
        depth_data = np.load(depth_path).astype(np.float32) / depth_unit_scale

        if min_depth is not None:
            depth_data[depth_data<min_depth] = 0

        depth_image = Image.fromarray(depth_data)        
        R = T_world_to_cam[:3, :3].T
        T = -R @ T_world_to_cam[:3, 3]
        
        pose_mats.append((R, T))

        height, width = rgb_image.height, rgb_image.width

        fovx = focal2fov(K[0][0], width)
        fovy = focal2fov(K[1][1], height)

        features, mask, scale = load_saga_info(path, f"rgb_{idx}")

        cam_info = CameraInfo(uid=idx, R=R, T=-R.T @ T, FovX=fovx, FovY=fovy, image=rgb_image,
            image_path=None, image_name=f"rgb_{idx}", width=width, height=height,
            R_gt=R.T, T_gt=T, depth=depth_image, features=features, masks=mask, mask_scales=scale)

        cam_infos.append(cam_info)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    else:    
        train_cam_infos = cam_infos
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))

    ply_path = os.path.join(path, "points3d.ply")

    poses = None
    
    if not os.path.exists(ply_path):
        pc = []
        colors = []
        bad_idxs = set()
        n_frame = len(json_data["frames"])
        frames = json_data['frames']
        for idx in tqdm.trange(0, n_frame, dynamic_ncols=True):
            # if idx > 100:
            #     break
            frame = frames[idx]

            depth_path = f"{path}/{frame['depth_file_path']}"
            rgb_path = f"{path}/{frame['color_file_path']}"
            K = np.array(frame["K"])
            
            rgb_image = np.array(Image.open(rgb_path))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            depth_data = np.load(depth_path).astype(np.float32) / depth_unit_scale

            if depth_data is not None and min_depth is not None:
                depth_data[depth_data < min_depth] = 0
                
            if depth_data is not None and depth_trunc is not None:
                depth_data[depth_data > depth_trunc] = 0

            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_image.shape[0], rgb_image.shape[1], K[0,0], K[1,1], K[0,2], K[1,2])
            
            o3d_depth = o3d.geometry.Image(np.array(depth_data).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(rgb_image).astype(np.uint8))

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.eye(4))
            
            pose = np.eye(4)
            R,T = pose_mats[idx]
            pose[:3, :3] = R
            pose[:3, 3] = T
            
            o3d_pc = o3d_pc.transform(pose)
            # o3d.io.write_point_cloud(f"pcds/test_{idx}.pcd", o3d_pc)

            # o3d.io.write_point_cloud(f"./reg_point_clouds/{idx}.pcd", o3d_pc)
            
            pc.append(np.array(o3d_pc.points[::30]).copy())
            colors.append(np.array(o3d_pc.colors[::30]).copy())
        
        # print(f"Excluding {len(bad_idxs)} poses")
        train_cam_infos = [cam for (idx,cam) in enumerate(train_cam_infos) if idx not in bad_idxs]
        pc_init = np.concatenate(pc)

        color_init = np.concatenate(colors)
        # downsample
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)
        o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)

        o3d_pcd = o3d_pcd.voxel_down_sample(0.05)
        pc_init = np.asarray(o3d_pcd.points)
        color_init = np.asarray(o3d_pcd.colors)
        num_pts = pc_init.shape[0]
        xyz = pc_init
        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None
    # print(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readOrbSlamData(path, eval, depth_trunc=25):
    intrinsics_path = f"{path}/calibration.json"

    with open(intrinsics_path, 'r') as intr_f:
        intrinsics_data = json.load(intr_f)
    
    height = intrinsics_data["height"]
    width = intrinsics_data["width"]

    fx = intrinsics_data["K"][0][0]
    fy = intrinsics_data["K"][1][1]
    cx = intrinsics_data["K"][0][2]
    cy = intrinsics_data["K"][1][2]

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3,3)
    
    print("Load K", K)
    
    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)

    keyframe_traj_file = f"{path}/kf_trajectory.txt"

    keyframe_traj = pd.read_csv(keyframe_traj_file, delimiter=' ', header=None).to_numpy()

    # Load trajectory data
    keyframe_times = keyframe_traj[:, 0]
    keyframe_positions = keyframe_traj[:, 1:4]
    quat = keyframe_traj[:, 4:]
    keyframe_rotations = Rotation.from_quat(quat)

    frame_times_file = f"{path}/frame_times.txt"
    frame_times = pd.read_csv(frame_times_file)["value"].to_numpy()

    cam_infos = []

    kf_to_frame = {}

    topk_focus = intrinsics_data["topk_focus"] if "topk_focus" in intrinsics_data else None
    topk_focus_path = f"{path}/topk_focus_indices_{topk_focus}.json" if topk_focus else None
    has_focused_images = topk_focus_path is not None and os.path.exists(topk_focus_path)

    if topk_focus and has_focused_images:
        with open(topk_focus_path, 'r') as f:
            kf_indices = json.load(f)
    else:
        kf_indices = range(len(keyframe_times))

    for kf_idx in tqdm.tqdm(kf_indices, dynamic_ncols=True):
        keyframe_time, position, rotation = keyframe_times[kf_idx], keyframe_positions[kf_idx], keyframe_rotations[kf_idx]
        
        frame_idx = np.argmin(np.abs(keyframe_time - frame_times))
        frame_time = frame_times[frame_idx]

        if not np.isclose(frame_time, keyframe_time):
            raise RuntimeError(f"Association error: KeyFrame was at time {keyframe_time}, but matched frame at time {frame_time}")

        kf_to_frame[kf_idx] = frame_idx

        img_path = f"{path}/images/{str(frame_idx).zfill(6)}.png"
        depth_path = f"{path}/depth/{str(frame_idx).zfill(6)}.png"

        image = Image.open(img_path)

        depth_image = Image.open(depth_path)
        depth_data = np.array(depth_image)/1000.0
        depth_image = Image.fromarray(depth_data)
        
        R = rotation.as_matrix().T
        T = -R @ position

        features, mask, scale = load_saga_info(path, str(frame_idx).zfill(6))

        cam_info = CameraInfo(
            uid=kf_idx,
            R=R.T, T=T,
            FovX = FovX, FovY=FovY,
            image=image, depth=depth_image,
            image_path=img_path,
            width=width, height=height,
            R_gt=rotation.as_matrix().T, T_gt=position,
            image_name = os.path.basename(img_path).split(".")[0],
            K = K,
            features=features, masks=mask, mask_scales=scale)
        
        cam_infos.append(cam_info)
    
    if topk_focus and not has_focused_images:
        print(f"Choosing the {topk_focus} most focused images")
        focus_measures = [cv2.Laplacian(cv2.cvtColor(np.asarray(cam_info.image), cv2.COLOR_BGR2GRAY), cv2.CV_16S).var() for cam_info in cam_infos]
        sort_order = np.argsort(focus_measures)
        
        topk_indices = sort_order[-topk_focus:]
        cam_infos = [cam_infos[i] for i in topk_indices]

        sort_order = sort_order.tolist()

        with open(topk_focus_path, 'w') as f:
            json.dump(sort_order[-topk_focus:], f)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    else:    
        train_cam_infos = cam_infos
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        
        pc_init = np.zeros((0,3))
        color_init = np.zeros((0,3))

        for cam in train_cam_infos:
            kf_idx = cam.uid
            frame_idx = kf_to_frame[kf_idx]

            pose_mat = np.eye(4)
            pose_mat[:3, 3] = keyframe_positions[kf_idx]
            pose_mat[:3, :3] = keyframe_rotations[kf_idx].as_matrix()

            img_path = f"{path}/images/{str(frame_idx).zfill(6)}.png"
            depth_path = f"{path}/depth/{str(frame_idx).zfill(6)}.png"

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            depth_image = Image.open(depth_path)
            depth_data = np.array(depth_image)/1000.0
            
            o3d_depth = o3d.geometry.Image(depth_data.astype(np.float32))
            o3d_image = o3d.geometry.Image(image.astype(np.uint8))

            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            # o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d_depth, intrinsic=o3d_intrinsic, extrinsic=np.identity(4), depth_scale=6553.5, stride=50)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
    
            o3d_pc = o3d_pc.transform(pose_mat)
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)[::20]), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)[::20]), axis=0)
            
        # downsample
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)
        o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)
        o3d_pcd = o3d_pcd.voxel_down_sample(0.025)
        pc_init = np.asarray(o3d_pcd.points)
        color_init = np.asarray(o3d_pcd.colors)

        num_pts = pc_init.shape[0]
        xyz = pc_init

        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz

        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_infos,
                        test_cameras=test_cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path)
    return scene_info

def readTUMInfo(path, pose_trans_noise=0, single_frame_id=[], voxel_size=None, force_load_ply=False, eval=False, target_images_total=None):
    folder_name = os.path.basename(path)
        
    traj_file = os.path.join(path, 'groundtruth.txt')
    ground_truth_df = pd.read_csv(traj_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ", skiprows=3)
    ground_truth_df = ground_truth_df.drop(ground_truth_df[(ground_truth_df.timestamp == '#')].index)
    poses, timestamps = build_tum_poses_from_df(ground_truth_df, False)
    ts_pose = np.asarray([t for t in timestamps])

    image_ts_file = os.path.join(path, 'rgb.txt')
    depth_ts_file = os.path.join(path, 'depth.txt')
    image_data = np.loadtxt(image_ts_file, delimiter=' ', dtype=np.unicode_, skiprows=0)
    depth_data = np.loadtxt(depth_ts_file, delimiter=' ', dtype=np.unicode_, skiprows=0)
    ts_image = image_data[:, 0].astype(np.float64)
    ts_depth = depth_data[:, 0].astype(np.float64)
    
    TUM_FPS=30
    max_dt = 1.0 / TUM_FPS *  1.1
    associations = []
    for img_idx, img_ts in enumerate(ts_image):
        depth_idx = np.argmin(np.abs(ts_depth - img_ts))
        pose_idx = np.argmin(np.abs(ts_pose - img_ts))

        if (np.abs(ts_depth[depth_idx] - img_ts) < max_dt) and \
                (np.abs(ts_pose[pose_idx] - img_ts) < max_dt):
            associations.append((img_idx, depth_idx, pose_idx))
    
    cam_infos = []
    mat_list = []
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))
    
    height = 480
    width = 640
    if 'freiburg1' in folder_name:
        print("Detect freiburg1. Use freiburg1 intrinsic.")
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        skip_step = 3
    elif 'freiburg2' in folder_name:
        print("Detect freiburg2. Use freiburg12 intrinsic.")
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
        skip_step = 10
    elif 'freiburg3' in folder_name:
        print("Detect freiburg3. Use freiburg3 intrinsic.")
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        skip_step = 10
    else:
        raise RuntimeError("Unknown intrinsic for TUM")
    
    if target_images_total is not None:
        num_images = len(associations)
        skip_step = max(1, num_images // target_images_total)
        print(f"Keeping one image for every {skip_step}")
        
    depth_scale = 5000.
    
    FovY = focal2fov(fy, height) # check where to incoorporate cx, cy
    FovX = focal2fov(fx, width)
        
    for img_idx, depth_idx, pose_idx in associations[::skip_step]:

        if len(single_frame_id)>0 and (img_idx not in single_frame_id):
            continue

        mat = np.array(poses[pose_idx])
        mat_list.append(mat)
        R = mat[:3,:3]
        T = mat[:3, 3]

        R_gt=R.copy()
        T_gt=T.copy()
        
        # Add noise to poses
        if pose_trans_noise > 0:
            np.random.seed(0)
            noise = np.random.rand(3) * pose_trans_noise
            T += noise

        # Invert
        T = -R.T @ T # convert from real world to GS format: R=R, T=T.inv()
        T_gt = -R_gt.T @ T_gt # convert from real world to GS format: R=R, T=T.inv()
        
        image_path = f"{path}/"+image_data[img_idx][1]
        depth_path = f"{path}/"+depth_data[depth_idx][1]
        temp = Image.open(image_path)
        image = temp.copy()
        temp = Image.open(depth_path)
        depth = temp.copy()
        temp.close()
        depth_scaled = Image.fromarray(np.array(depth) / depth_scale)

        image_name = os.path.basename(image_path).split(".png")[0]

        if len(single_frame_id)>0 and (img_idx not in single_frame_id):
            print("img_idx: ", img_idx)
            continue 
        else:
            o3d_depth = o3d.geometry.Image(np.array(depth_scaled).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) # w, h, fx, fy, cx, cy

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1., depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            o3d_pc = o3d_pc.transform(mat)

            cam_info = CameraInfo(uid=img_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width,
                                height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt)
            cam_infos.append(cam_info)
    

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    print("Starting to build ply")
    
    if not force_load_ply:
        for img_idx, depth_idx, pose_idx in (associations):
            if len(single_frame_id)>0 and (img_idx not in single_frame_id):
                continue
            mat = np.array(poses[pose_idx])
            image_path = f"{path}/"+image_data[img_idx][1]
            depth_path = f"{path}/"+depth_data[depth_idx][1]
            temp = Image.open(image_path)
            image = temp.copy()
            temp = Image.open(depth_path)
            depth = temp.copy()
            temp.close()
            depth_scaled = np.array(depth) / depth_scale
            
            o3d_depth = o3d.geometry.Image(np.array(depth_scaled).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) # w, h, fx, fy, cx, cy
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1., depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            dist = np.linalg.norm(np.asarray(o3d_pc.points), axis=1)

            o3d_pc = o3d_pc.transform(mat)
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)[::100]), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)[::100]), axis=0)
        
        num_pts = pc_init.shape[0]
        xyz = pc_init
        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz
        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
        print('save pcd')
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None
    

    if voxel_size is not None:
        # downsample
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
        o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size)
        pc_init = np.asarray(o3d_pcd.points)
        color_init = np.asarray(o3d_pcd.colors)
        pcd = BasicPointCloud(points=pc_init, colors=color_init, normals=np.zeros((pc_init.shape[0], 3)))

    viz_list=[]
    if False:
        mat = mat_list[0]
        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis_mesh.scale(0.5, center=axis_mesh.get_center())
        mesh = axis_mesh.transform(mat)
        viz_list.append(mesh)
        
        for mat in mat_list:
            axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            axis_mesh.scale(0.1, center=axis_mesh.get_center())
            mesh = axis_mesh.transform(mat)
            viz_list.append(mesh)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)  
        o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)    
        viz_list.append(o3d_pcd)

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        viz_list.append(axis_mesh)

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh = axis_mesh.translate((1,0,0))
        axis_mesh.scale(0.5, center=axis_mesh.get_center())
        viz_list.append(axis_mesh)

        o3d.visualization.draw_geometries(viz_list)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readScanNetPPInfo(data_root, eval=False):
    transforms_path = f"{data_root}/dslr/nerfstudio/transforms_undistorted.json"
    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)
        
    focal_x, focal_y = transforms_data['fl_x'], transforms_data['fl_y']
    cx, cy = transforms_data['cx'], transforms_data['cy']
    width, height = transforms_data['w'], transforms_data['h']
    
    fov_x = focal2fov(focal_x, width)
    fov_y = focal2fov(focal_y, height)
    
    train_cams, test_cams = [], []
    
    cameras_extrinsic_file = os.path.join(data_root, "dslr", "colmap", "images.txt")
    colmap_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    
    camera_poses = {}
    for cam in colmap_extrinsics.values():
        name = cam.name
        qvec = cam.qvec
        tvec = cam.tvec
        R = np.transpose(qvec2rotmat(qvec)) # R_colmap.T !!!!!!
        t = np.array(tvec)
        camera_poses[name] = (R,t)
        
    def process_frame(frame_id, frame):
        R, t = camera_poses[frame['file_path']]
        
        img_path = f"{data_root}/dslr/undistorted_images/{frame['file_path']}"
        depth_path = f"{data_root}/dslr/render_depth/{Path(frame['file_path']).stem}.png"
        
        image = Image.open(img_path)
        
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.
                
        features, mask, scale = load_saga_info(data_root, Path(frame['file_path']).stem)
        
        cam_info = CameraInfo(
            uid=frame_id,
            R=R, T=t,
            FovX = fov_x, FovY=fov_y,
            image=image, depth=Image.fromarray(depth_img),
            image_path=img_path,
            width=width, height=height,
            R_gt=R, T_gt=t,
            image_name = os.path.basename(img_path).split(".")[0],
            K = None,
            features=features,
            masks=mask,
            mask_scales=scale
        )
        
        return cam_info
        
    for frame_id, frame in enumerate(transforms_data['frames']):
        train_cams.append(process_frame(frame_id, frame))
    
    num_train_cams = len(transforms_data['frames'])
    for frame_id, frame in enumerate(transforms_data['test_frames']):
        new_cam = process_frame(frame_id + num_train_cams, frame)
        if eval:
            test_cams.append(new_cam)
        else:
            train_cams.append(new_cam)
            
    ply_path = os.path.join(data_root, "points3d.ply")
    if not os.path.exists(ply_path):
        
        plydata = PlyData.read(f"{data_root}/scans/mesh_aligned_0.05.ply")
        vertices = plydata['vertex']
        x = np.asarray(vertices['x'])
        y = np.asarray(vertices['y'])
        z = np.asarray(vertices['z'])
        r = np.asarray(vertices['red'])
        g = np.asarray(vertices['green'])
        b = np.asarray(vertices['blue'])
        
        points = np.hstack((x[:, None], y[:, None], z[:, None]))[::100]
        colors = np.hstack((r[:, None], g[:, None], b[:, None]))[::100]

        storePly(ply_path, points, colors*255)
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None
        
    nerf_normalization = getNerfppNorm(train_cams)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cams,
                           test_cameras=test_cams,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica" : readReplicaInfo,
    "PyBullet": readPyBulletData,
    "ROSBag": readROSData,
    "ORB_SLAM": readOrbSlamData,
    "TUM": readTUMInfo,
    "ScanNet++": readScanNetPPInfo
}
