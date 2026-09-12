"""Export a Gaussian Splatting .ply to a flat CSV.

One row per Gaussian, with columns:
    mu_x, mu_y, mu_z      position (scene units, scale factor applied)
    s_x, s_y, s_z         scale
    q_w, q_x, q_y, q_z    rotation quaternion
    opacity               opacity
    r, g, b               SH DC colour

Usage:
    python -m normalized_splatting.data_generation.ply_to_csv input.ply out.csv
"""
import argparse
import math
import os

import pandas as pd
import torch

from normalized_splatting.scene.gaussian_model import GaussianModel


def process_ply(input_path, sh_degree=3):
    """Load a Gaussian .ply and return its parameters as a DataFrame."""
    model = GaussianModel(sh_degree)
    model.load_ply(input_path)

    means = model.get_xyz * model.scale_factor
    scaling = model.get_scaling * math.sqrt(model.scale_factor)
    rotation = model.get_rotation
    opacity = model.get_opacity
    colors = model.get_rgb.squeeze()

    header = ["mu_x", "mu_y", "mu_z",
              "s_x", "s_y", "s_z",
              "q_w", "q_x", "q_y", "q_z",
              "opacity", "r", "g", "b"]
    data = torch.hstack((means, scaling, rotation, opacity, colors))
    return pd.DataFrame(data.detach().cpu().numpy(), columns=header)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_ply")
    parser.add_argument("output_path")
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)

    process_ply(args.input_ply, args.sh_degree).to_csv(args.output_path, index=None)


if __name__ == "__main__":
    main()
