#!/usr/bin/env python3
"""
Precompute point clouds for all .pickle.xz files in a directory.

This script restructures the data to match the expected format for readPyBulletData
and generates the points3d.ply files.

Input structure:
    scenes/
        scene_name_1.pickle.xz
        scene_name_2.pickle.xz
        ...

Output structure:
    scenes/
        scene_name_1/
            data.pickle.xz
            points3d.ply
        scene_name_2/
            data.pickle.xz
            points3d.ply
        ...
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add parent directory to path so we can import from normalized_splatting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from normalized_splatting.scene.dataset_readers import readPyBulletData


def precompute_pointclouds(input_dir: str, dry_run: bool = False):
    """
    Process all .pickle.xz files in input_dir.

    Args:
        input_dir: Directory containing .pickle.xz files
        dry_run: If True, only print what would be done without executing
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        sys.exit(1)

    # Find all .pickle.xz files
    pickle_files = list(input_path.glob("*.pickle.xz"))

    if not pickle_files:
        print(f"No .pickle.xz files found in {input_dir}")
        return

    print(f"Found {len(pickle_files)} .pickle.xz files to process")

    for pickle_file in pickle_files:
        # Extract scene name (remove .pickle.xz extension)
        scene_name = pickle_file.name.replace(".pickle.xz", "")
        scene_dir = input_path / scene_name
        target_path = scene_dir / "data.pickle.xz"

        print(f"\nProcessing: {scene_name}")

        if dry_run:
            print(f"  Would create directory: {scene_dir}")
            print(f"  Would move {pickle_file} -> {target_path}")
            print(f"  Would generate point cloud")
            continue

        # Create scene directory if it doesn't exist
        scene_dir.mkdir(exist_ok=True)

        # Move/copy the pickle file to the expected location
        if not target_path.exists():
            shutil.move(str(pickle_file), str(target_path))
            print(f"  Moved to: {target_path}")
        else:
            print(f"  Target already exists: {target_path}")

        # Check if point cloud already exists
        ply_path = scene_dir / "points3d.ply"
        if ply_path.exists():
            print(f"  Point cloud already exists: {ply_path}")
            continue

        # Generate point cloud by calling readPyBulletData
        print(f"  Generating point cloud...")
        try:
            readPyBulletData(str(scene_dir))
            print(f"  Created: {ply_path}")
        except Exception as e:
            print(f"  Error processing {scene_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute point clouds for .pickle.xz scene files"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .pickle.xz files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )

    args = parser.parse_args()
    precompute_pointclouds(args.input_dir, args.dry_run)


if __name__ == "__main__":
    main()
