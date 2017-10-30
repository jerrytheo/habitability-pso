from functools import partial
import numpy as np
from scipy.misc import derivative


def gradient_ascent(f, df=None, ndim=1, eta=0.01, initial=1, dx=1.0,
                    precision=1e-5, iterations=100000):
    """Maximize f using the Gradient Ascent approach.
    If df (derivative) is not specified, it is approximated using
    scipy.misc.derivative."""

    if not df:
        df = partial(derivative, f, dx)

    prev_step = cur_x = initial
    while prev_step > precision:
        prev_x = cur_x
        cur_x += eta * df(cur_x)
        prev_step = np.abs(cur_x - prev_x)
    return cur_x


if __name__ == '__main__':
    print(gradient_ascent(np.sin, np.cos))
