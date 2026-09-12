#!/bin/sh

sudo ldconfig

# The CUDA extensions are compiled inside this container and pick their target
# architectures from the GPUs they can see. Without a GPU they fall back to
# PyTorch's default list, which may not cover this machine's card -- the build
# succeeds and then fails at runtime with "no kernel image is available for
# execution on the device". Warn early rather than after a long build.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && ! nvidia-smi -L >/dev/null 2>&1; then
  echo "WARNING: no GPU is visible in this container (is it started with --gpus=all?)."
  echo "         Building the CUDA extensions here may target the wrong architectures."
  echo "         Set TORCH_CUDA_ARCH_LIST (e.g. \"8.9;12.0+PTX\") before pip install to be explicit."
fi

/bin/bash
