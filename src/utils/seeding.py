"""Global seed control for fully reproducible runs.

Call `set_global_seed(seed)` once at the top of every experiment entry point
before constructing any models, dataloaders, or samplers.
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set every relevant RNG source to `seed` for reproducibility.

    Covers:
    - Python built-in random
    - NumPy
    - PyTorch CPU and CUDA
    - cuDNN deterministic mode (disables non-deterministic CUDA kernels)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=False ensures cuDNN picks the same algorithm every run;
    # setting deterministic alone is not sufficient on some hardware.
    torch.backends.cudnn.benchmark = False
