import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)

import json
import shutil
import argparse
import numpy as np
from PIL import Image

from normalized_splatting.scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, SceneInfo
from normalized_splatting.utils.graphics_utils import fov2focal


def cam_to_c2w(cam: CameraInfo) -> np.ndarray:
    """Reconstruct a C2W matrix from CameraInfo's R and T, in Nerfstudio's
    OpenGL/Blender convention (+X right, +Y up, +Z back).

    GS convention (all dataset readers): R = C2W rotation, T = W2C translation,
    in OpenCV/COLMAP camera axes (+X right, +Y down, +Z forward).
    Camera position in world = -R @ T.
    """
    # cam.R is the C2W rotation, so cam.R.T is the W2C rotation; together with
    # the W2C translation cam.T this is the world-to-camera matrix.
    w2c = np.eye(4)
    w2c[:3, :3] = cam.R.T
    w2c[:3, 3] = cam.T
    c2w = np.linalg.inv(w2c)

    # Dataset readers store OpenCV camera axes; Nerfstudio expects OpenGL.
    # Flip the camera's Y and Z axes (columns 1 and 2 of the rotation).
    c2w[:3, 1:3] *= -1
    return c2w


def detect_and_load(source_path: str, eval: bool, depth_trunc: float, min_depth: float | None) -> SceneInfo:
    if os.path.exists(os.path.join(source_path, "sparse")):
        print("Found sparse/, assuming COLMAP data")
        return sceneLoadTypeCallbacks["Colmap"](source_path, None, None, eval)
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json, assuming Blender data")
        return sceneLoadTypeCallbacks["Blender"](source_path, False, eval)
    elif os.path.exists(os.path.join(source_path, "traj.txt")):
        print("Found traj.txt, assuming Replica data")
        return sceneLoadTypeCallbacks["Replica"](source_path, eval, depth_trunc=depth_trunc)
    elif os.path.exists(os.path.join(source_path, "data.pickle")) or \
            os.path.exists(os.path.join(source_path, "data.pickle.xz")):
        print("Found data.pickle, assuming PyBullet data")
        return sceneLoadTypeCallbacks["PyBullet"](source_path)
    elif os.path.exists(os.path.join(source_path, "poses.json")):
        print(f"Found poses.json, assuming ROSBag data (depth_trunc={depth_trunc})")
        return sceneLoadTypeCallbacks["ROSBag"](source_path, depth_trunc=depth_trunc, min_depth=min_depth, eval=eval)
    elif os.path.exists(os.path.join(source_path, "groundtruth.txt")):
        print("Found groundtruth.txt, assuming TUM data")
        return sceneLoadTypeCallbacks["TUM"](source_path, eval=eval, target_images_total=250)
    elif os.path.exists(os.path.join(source_path, "kf_trajectory.txt")):
        print("Found kf_trajectory.txt, assuming ORB_SLAM data")
        return sceneLoadTypeCallbacks["ORB_SLAM"](source_path, eval, depth_trunc=depth_trunc)
    elif os.path.exists(os.path.join(source_path, "dslr")):
        print("Found dslr/, assuming ScanNet++ data")
        return sceneLoadTypeCallbacks["ScanNet++"](source_path, eval)
    else:
        raise ValueError(f"Could not recognize scene type in: {source_path}")


def export_cameras(cameras: list[CameraInfo], output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    image_dir = os.path.join(output_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    has_depth = any(cam.depth is not None for cam in cameras)
    if has_depth:
        depth_dir = os.path.join(output_path, "depth")
        os.makedirs(depth_dir, exist_ok=True)

    first = cameras[0]
    fl_x = fov2focal(first.FovX, first.width)
    fl_y = fov2focal(first.FovY, first.height)

    out = {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": first.width / 2.0,
        "cy": first.height / 2.0,
        "w": first.width,
        "h": first.height,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0,
        "frames": [],
    }

    for i, cam in enumerate(cameras):
        stem = cam.image_name if cam.image_name else f"frame_{i:05d}"
        img_filename = f"{stem}.png"
        img_out = os.path.join(image_dir, img_filename)

        if cam.image_path and os.path.exists(cam.image_path):
            shutil.copy2(cam.image_path, img_out)
        elif cam.image is not None:
            img = cam.image if isinstance(cam.image, Image.Image) else Image.fromarray(np.array(cam.image))
            img.convert("RGB").save(img_out)
        else:
            print(f"Warning: camera {i} has no image, skipping.")
            continue

        c2w = cam_to_c2w(cam)
        frame: dict = {
            "file_path": os.path.join("images", img_filename),
            "transform_matrix": c2w.tolist(),
        }

        # Per-camera intrinsics override when they differ from the header
        cam_fl_x = fov2focal(cam.FovX, cam.width)
        cam_fl_y = fov2focal(cam.FovY, cam.height)
        if abs(cam_fl_x - fl_x) > 0.5 or abs(cam_fl_y - fl_y) > 0.5 \
                or cam.width != first.width or cam.height != first.height:
            frame.update({
                "fl_x": cam_fl_x, "fl_y": cam_fl_y,
                "cx": cam.width / 2.0, "cy": cam.height / 2.0,
                "w": cam.width, "h": cam.height,
            })

        if has_depth and cam.depth is not None:
            depth_filename = f"{stem}.npy"
            np.save(os.path.join(depth_dir, depth_filename), np.array(cam.depth, dtype=np.float32))
            frame["depth_file_path"] = os.path.join("depth", depth_filename)

        out["frames"].append(frame)

    transforms_path = os.path.join(output_path, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out['frames'])} frames -> {transforms_path}")


def scene_to_nerfstudio(source_path: str, output_path: str, split: str,
                         eval: bool, depth_trunc: float, min_depth: float | None) -> None:
    scene_info = detect_and_load(source_path, eval=eval, depth_trunc=depth_trunc, min_depth=min_depth)

    if split in ("train", "all") and scene_info.train_cameras:
        dest = output_path if split == "train" else os.path.join(output_path, "train")
        export_cameras(scene_info.train_cameras, dest)

    if split in ("test", "all") and scene_info.test_cameras:
        dest = output_path if split == "test" else os.path.join(output_path, "test")
        export_cameras(scene_info.test_cameras, dest)

    if split == "all" and not scene_info.test_cameras:
        # No test split — move train output to top level if it was nested
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert any scene type supported by normalized_splatting to Nerfstudio transforms.json format."
    )
    parser.add_argument("input_path", type=str, help="Input scene directory")
    parser.add_argument("output_path", type=str, nargs="?",
                        help="Output directory (defaults to input_path)")
    parser.add_argument("--eval", action="store_true",
                        help="Enable eval mode (splits data into train/test)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="Which split to export (default: all)")
    parser.add_argument("--depth-trunc", type=float, default=1000.0,
                        help="Depth truncation distance in metres for loaders that support it (default: 1000)")
    parser.add_argument("--min-depth", type=float, default=None,
                        help="Minimum depth threshold in metres for loaders that support it")

    args = parser.parse_args()
    out = args.output_path if args.output_path else args.input_path

    scene_to_nerfstudio(
        source_path=args.input_path,
        output_path=out,
        split=args.split,
        eval=args.eval,
        depth_trunc=args.depth_trunc,
        min_depth=args.min_depth,
    )
