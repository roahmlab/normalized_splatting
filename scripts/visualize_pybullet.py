"""Visualize the point cloud and camera frustums from a PyBullet dataset."""

import argparse
import lzma
import math
import os
import pickle

import numpy as np
import open3d as o3d
from PIL import Image


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def create_camera_frustum(pose, fovx, fovy, width, height, scale=0.3, color=(1, 0, 0)):
    """Create a wireframe camera frustum as an Open3D LineSet.

    Args:
        pose: 4x4 camera-to-world matrix
        fovx, fovy: horizontal and vertical field of view in radians
        width, height: image dimensions in pixels
        scale: depth of the frustum in world units
        color: RGB tuple for the frustum color
    """
    # Half-extents of the image plane at unit depth
    half_w = math.tan(fovx / 2) * scale
    half_h = math.tan(fovy / 2) * scale

    # Frustum corners in camera-local coords (OpenCV convention: +Z forward)
    corners_cam = np.array([
        [0, 0, 0],               # camera center
        [-half_w, -half_h, scale],  # top-left
        [ half_w, -half_h, scale],  # top-right
        [ half_w,  half_h, scale],  # bottom-right
        [-half_w,  half_h, scale],  # bottom-left
    ])

    # Transform to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    # Lines: center to each corner, plus the rectangle
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # edges from apex
        [1, 2], [2, 3], [3, 4], [4, 1],  # image-plane rectangle
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def load_pybullet_data(path):
    """Load data from a PyBullet pickle file, mirroring readPyBulletData."""
    data_path = f"{path}/data.pickle"
    compr_data_path = f"{data_path}.xz"

    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    elif os.path.exists(compr_data_path):
        with lzma.open(compr_data_path, "rb") as f:
            data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Can't find data at {data_path} or {compr_data_path}")

    img_data = []
    pose_keys = sorted(data["poses"].keys())
    for i in range(len(data["rgb"])):
        cam_name = pose_keys[i]
        pose = data["poses"][cam_name].copy()
        pose = pose @ np.diag([1, -1, -1, 1])

        rgb = data["rgb"][i].astype(np.uint8)
        rgb = Image.fromarray(rgb, "RGB")
        depth = Image.fromarray(data["depth"][i])

        img_data.append((pose, rgb, depth))

    fovy = float(np.pi) / 3.0
    fovx = 2 * math.atan(rgb.size[0] / (2 * fov2focal(fovy, rgb.size[1])))

    return img_data, fovx, fovy


def build_point_cloud(img_data, fovx, fovy):
    """Reconstruct a coloured point cloud from RGBD frames."""
    width, height = img_data[0][1].size
    focalx = fov2focal(fovx, width)
    focaly = fov2focal(fovy, height)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, focalx, focaly, width / 2, height / 2
    )

    pc_all = np.zeros((0, 3))
    color_all = np.zeros((0, 3))

    for pose, rgb, depth in img_data:
        o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
        o3d_image = o3d.geometry.Image(np.array(rgb).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_image, o3d_depth, depth_scale=1.0, depth_trunc=254,
            convert_rgb_to_intensity=False,
        )
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd, intrinsic=intrinsic, extrinsic=np.identity(4),
        )
        pc = pc.voxel_down_sample(0.05)
        pc = pc.transform(pose)

        pc_all = np.concatenate((pc_all, np.asarray(pc.points)), axis=0)
        color_all = np.concatenate((color_all, np.asarray(pc.colors)), axis=0)

    # Global downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_all)
    pcd.colors = o3d.utility.Vector3dVector(color_all)
    pcd = pcd.voxel_down_sample(0.05)
    return pcd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the PyBullet dataset directory")
    parser.add_argument("--frustum-scale", type=float, default=0.3,
                        help="Depth of each camera frustum (default: 0.3)")
    parser.add_argument("--no-pointcloud", action="store_true",
                        help="Skip point cloud reconstruction (faster)")
    args = parser.parse_args()

    print("Loading data...")
    img_data, fovx, fovy = load_pybullet_data(args.path)
    width, height = img_data[0][1].size
    print(f"  {len(img_data)} cameras, image size {width}x{height}")

    geometries = []

    # Point cloud
    if not args.no_pointcloud:
        print("Building point cloud from RGBD...")
        pcd = build_point_cloud(img_data, fovx, fovy)
        print(f"  {len(pcd.points)} points")
        geometries.append(pcd)

    # Camera frustums with colour gradient
    print("Creating camera frustums...")
    cmap = o3d.utility.Vector3dVector(
        [[1, 0, 0]]  # just to get the type; we'll assign per-camera below
    )
    for idx, (pose, rgb, depth) in enumerate(img_data):
        t = idx / max(len(img_data) - 1, 1)
        # Blue -> Green -> Red gradient
        if t < 0.5:
            color = (0, 2 * t, 1 - 2 * t)
        else:
            color = (2 * (t - 0.5), 1 - 2 * (t - 0.5), 0)

        frustum = create_camera_frustum(
            pose, fovx, fovy, width, height,
            scale=args.frustum_scale, color=color,
        )
        geometries.append(frustum)

    # World-origin coordinate frame
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(origin)

    print(f"Displaying {len(geometries)} geometries. Close the window to exit.")
    o3d.visualization.draw_geometries(geometries, window_name="PyBullet Scene")


if __name__ == "__main__":
    main()
