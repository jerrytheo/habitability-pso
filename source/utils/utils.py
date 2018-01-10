import numpy as np
import sys


def _round(*args, **kwargs):
    return np.round(*args, **kwargs, decimals=6)


def _uniform(*args, **kwargs):
    return _round(np.random.uniform(*args, **kwargs))
