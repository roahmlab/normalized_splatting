import os
import json
import shutil
import argparse
import numpy as np
from PIL import Image


def load_transforms(transforms_path: str) -> dict:
    with open(transforms_path, "r") as f:
        return json.load(f)


def frame_intrinsics(frame: dict, header: dict) -> np.ndarray:
    """Build a 3x3 K from per-frame values, falling back to the header."""
    def get(key):
        return frame.get(key, header.get(key))

    fl_x, fl_y = get("fl_x"), get("fl_y")
    cx, cy = get("cx"), get("cy")
    if fl_x is None or fl_y is None or cx is None or cy is None:
        raise ValueError("transforms.json is missing intrinsics (fl_x/fl_y/cx/cy)")

    return np.array([
        [fl_x, 0.0, cx],
        [0.0, fl_y, cy],
        [0.0, 0.0, 1.0],
    ])


def c2w_opengl_to_opencv(c2w: np.ndarray, applied_transform: np.ndarray | None) -> np.ndarray:
    """Convert a Nerfstudio (OpenGL/Blender axes) C2W matrix to OpenCV camera axes.

    This is the exact inverse of scene_to_nerfstudio.cam_to_c2w: flip the
    camera's Y and Z axes back. If the file carries an applied_transform
    (added by ns-process-data), undo it first to recover the original world frame.
    """
    c2w = np.array(c2w, dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, [0, 0, 0, 1]])

    if applied_transform is not None:
        at = np.eye(4)
        at[:3, :4] = np.array(applied_transform, dtype=np.float64)[:3, :4]
        c2w = np.linalg.inv(at) @ c2w

    c2w[:3, 1:3] *= -1
    return c2w


def convert_depth(src_path: str, dst_path: str, depth_scale: float) -> None:
    """Copy or convert a depth file to a float32 .npy in metres."""
    if src_path.endswith(".npy"):
        if depth_scale == 1.0:
            shutil.copy2(src_path, dst_path)
        else:
            np.save(dst_path, np.load(src_path).astype(np.float32) * depth_scale)
    else:
        # Nerfstudio-style depth image (e.g. 16-bit PNG, typically millimetres)
        depth = np.array(Image.open(src_path)).astype(np.float32) * depth_scale
        np.save(dst_path, depth)


def nerfstudio_to_scene(input_path: str, output_path: str, depth_scale: float) -> None:
    transforms_path = input_path
    if os.path.isdir(transforms_path):
        transforms_path = os.path.join(transforms_path, "transforms.json")
    source_dir = os.path.dirname(os.path.abspath(transforms_path))

    header = load_transforms(transforms_path)

    os.makedirs(output_path, exist_ok=True)
    rgb_dir = os.path.join(output_path, "rgb")
    depth_dir = os.path.join(output_path, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    applied_transform = header.get("applied_transform")

    out_frames = []
    skipped_depth = 0
    for i, frame in enumerate(header["frames"]):
        img_src = os.path.join(source_dir, frame["file_path"])
        if not os.path.exists(img_src):
            print(f"Warning: frame {i} image not found ({img_src}), skipping.")
            continue

        stem = os.path.splitext(os.path.basename(frame["file_path"]))[0]
        img_filename = f"{stem}{os.path.splitext(img_src)[1]}"
        shutil.copy2(img_src, os.path.join(rgb_dir, img_filename))

        c2w = c2w_opengl_to_opencv(frame["transform_matrix"], applied_transform)
        out_frame = {
            "transform_matrix": c2w.tolist(),
            "K": frame_intrinsics(frame, header).tolist(),
            "color_file_path": os.path.join("rgb", img_filename),
        }

        depth_src = frame.get("depth_file_path")
        if depth_src is not None and os.path.exists(os.path.join(source_dir, depth_src)):
            depth_filename = f"{stem}.npy"
            convert_depth(os.path.join(source_dir, depth_src),
                          os.path.join(depth_dir, depth_filename), depth_scale)
            out_frame["depth_file_path"] = os.path.join("depth", depth_filename)
        else:
            skipped_depth += 1

        out_frames.append(out_frame)

    if skipped_depth:
        print(f"Warning: {skipped_depth}/{len(out_frames) + skipped_depth} frames have no depth; "
              "the ROSBag reader (poses.json) requires depth for every frame.")

    poses_path = os.path.join(output_path, "poses.json")
    with open(poses_path, "w") as f:
        json.dump({"frames": out_frames}, f, indent=2)
    print(f"Saved {len(out_frames)} frames -> {poses_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Nerfstudio transforms.json scene to the ROSBag poses.json format "
                    "readable by normalized_splatting (inverse of scene_to_nerfstudio.py)."
    )
    parser.add_argument("input_path", type=str,
                        help="Path to transforms.json, or a directory containing it")
    parser.add_argument("output_path", type=str, help="Output scene directory")
    parser.add_argument("--depth-scale", type=float, default=1.0,
                        help="Multiplier applied to depth values to get metres "
                             "(default: 1.0; use 0.001 for 16-bit PNG depth in millimetres)")

    args = parser.parse_args()
    nerfstudio_to_scene(args.input_path, args.output_path, args.depth_scale)
