import numpy as np

ERR = 1e-6


def _round(*args, **kwargs):
    return np.round(*args, **kwargs, decimals=-int(np.log10(ERR)))


def _uniform(*args, **kwargs):
    return _round(np.random.uniform(*args, **kwargs))
