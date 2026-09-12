import os

import lzma
import pickle
import json
import argparse
import numpy as np
from PIL import Image
from normalized_splatting.utils.graphics_utils import fov2focal


def readPyBulletData(input_path, output_path, scale_to_unit_cube=False):
    data_dir = input_path
    data_path = os.path.join(data_dir, "data.pickle")
    compr_data_path = data_path + ".xz"

    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
    elif os.path.exists(compr_data_path):
        with lzma.open(compr_data_path, 'rb') as f:
            raw_data = pickle.load(f)
    else:
        raise Exception("Can't find the data file at ", data_path)

    h, w, _ = raw_data['rgb'][0].shape
    focal = fov2focal(np.pi / 3., h)

    out_data = {
        "camera_model": "OPENCV",
        "fl_x": focal,
        "fl_y": focal,
        "cx": w / 2.0,
        "cy": h / 2.0,
        "w": w,
        "h": h,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": [],
    }

    output_image_dir = os.path.join(output_path, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    output_depth_dir = os.path.join(output_path, "depth")
    os.makedirs(output_depth_dir, exist_ok=True)

    # Compute scale and translation for unit cube normalization
    scale_factor = 1.0
    translation = np.zeros(3)

    if scale_to_unit_cube:
        # Extract all camera positions
        camera_positions = []
        for i in range(len(raw_data['rgb'])):
            if "BASECAM" in raw_data['poses']:
                cam_name = "BASECAM" if i == 0 else f"CAMERA{i-1}"
            else:
                cam_name = f"CAMERA{i}"

            pose = raw_data['poses'][cam_name].copy()
            camera_positions.append(pose[:3, 3])

        camera_positions = np.array(camera_positions)

        # Compute bounding box
        min_coords = camera_positions.min(axis=0)
        max_coords = camera_positions.max(axis=0)

        # Compute center and extent
        center = (min_coords + max_coords) / 2.0
        extent = max_coords - min_coords

        # Compute scale to fit in [-0.5, 0.5]^3 (size 1.0)
        max_extent = extent.max()
        scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0

        # Translation to center at origin, then scale
        translation = -center

        print(f"Scaling to unit cube: center={center}, extent={extent}, scale={scale_factor}")

    for i in range(len(raw_data['rgb'])):
        if "BASECAM" in raw_data['poses']:
            cam_name = "BASECAM" if i == 0 else f"CAMERA{i-1}"
        else:
            cam_name = f"CAMERA{i}"
            
        pose = raw_data['poses'][cam_name].copy()
        # pose = np.linalg.inv(pose)
        # pose[1:3, :3] *= -1

        # Apply scale to unit cube transformation
        if scale_to_unit_cube:
            # Translate and scale the camera position
            pose[:3, 3] = (pose[:3, 3] + translation) * scale_factor

        transform = pose
        rgb_array = raw_data['rgb'][i].astype(np.uint8)
        image_filename = f"frame_{i:05d}.png"
        image_path = os.path.join(output_image_dir, image_filename)
        Image.fromarray(rgb_array, "RGB").save(image_path)

        depth_filename = None
        if "depth" in raw_data and len(raw_data['depth']) > i:
            depth_array = raw_data['depth'][i].copy()

            # Scale depth values if using unit cube normalization
            if scale_to_unit_cube:
                depth_array = depth_array * scale_factor

            depth_filename = f"depth_{i:05d}.png"
            depth_array = (depth_array * 1000).astype(np.uint32)
            depth_path = os.path.join(output_depth_dir, depth_filename)
            Image.fromarray(depth_array).save(depth_path)

        frame_info = {
            "file_path": os.path.join("images", image_filename),
            "transform_matrix": transform.tolist()
        }
        if depth_filename:
            frame_info["depth_file_path"] = os.path.join(
                "depth", depth_filename)

        out_data["frames"].append(frame_info)

    transforms_path = os.path.join(output_path, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print("Transforms saved to", transforms_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyBullet data to Nerfstudio format")
    parser.add_argument("input_path", type=str, help="Input data directory")
    parser.add_argument("output_path", type=str, nargs='?', help="Output directory (defaults to input_path)")
    parser.add_argument("--scale-to-unit-cube", action="store_true",
                        help="Scale and translate the scene to fit within a [-0.5, 0.5]^3 unit cube")

    args = parser.parse_args()

    output_path = args.output_path if args.output_path else args.input_path

    readPyBulletData(args.input_path, output_path, scale_to_unit_cube=args.scale_to_unit_cube)
