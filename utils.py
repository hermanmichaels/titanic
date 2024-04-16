import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set all seeds.

    Args:
        seed: seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
