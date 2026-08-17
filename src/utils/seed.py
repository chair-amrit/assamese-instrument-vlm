"""
Reproducibility utilities.

Centralizes random-seed configuration for Python, NumPy, and PyTorch.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Random seed used across supported libraries.
    deterministic:
        If True, request deterministic PyTorch algorithms where possible.
        This may reduce performance or affect compatibility, so it is
        disabled by default.

    Notes
    -----
    This preserves the seed behavior used in the original notebook:
    Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations may not have deterministic implementations.
            pass