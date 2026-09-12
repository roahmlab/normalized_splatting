from functools import lru_cache

import torch

from .modules.lpips import LPIPS


@lru_cache(maxsize=None)
def _get_criterion(net_type: str, version: str, device: str):
    return LPIPS(net_type, version).to(device)


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    The underlying network is cached per (net_type, version, device); building
    it on every call made metric runs dominated by network construction.

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    criterion = _get_criterion(net_type, version, str(x.device))
    return criterion(x, y)
