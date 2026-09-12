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
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _detect_arch_list():
    """Compute capabilities of the GPUs visible on this machine, as "8.6" strings.

    Returns an empty list when no GPU can be queried (e.g. a container built on
    a CPU-only host), in which case PyTorch picks its own default arch list.
    """
    try:
        import torch
        if torch.cuda.is_available():
            caps = {torch.cuda.get_device_capability(i)
                    for i in range(torch.cuda.device_count())}
            if caps:
                return sorted(f"{major}.{minor}" for major, minor in caps)
    except Exception:
        pass

    # torch.cuda is unavailable in some build environments (no driver in the
    # build stage, CPU-only wheel); nvidia-smi can still know the hardware.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.SubprocessError):
        return []

    return sorted({line.strip() for line in out.splitlines() if line.strip()})


# An explicit TORCH_CUDA_ARCH_LIST always wins, so cross-compiling for other
# GPUs still works the usual way.
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    arch_list = _detect_arch_list()
    if arch_list:
        # +PTX on the newest arch keeps the build forward-compatible with GPUs
        # newer than anything installed here.
        arch_list[-1] += "+PTX"
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
        print(f"Detected CUDA architectures: {os.environ['TORCH_CUDA_ARCH_LIST']}")
    else:
        print("No CUDA device detected; falling back to PyTorch's default "
              "architecture list. Set TORCH_CUDA_ARCH_LIST to override.")

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(ROOT_DIR, "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
