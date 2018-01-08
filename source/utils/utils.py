from functools import partial
import numpy as np

_round = partial(np.round, decimals=6)


def _uniform(*args, **kwargs):
    return _round(np.random.uniform(*args, **kwargs))
