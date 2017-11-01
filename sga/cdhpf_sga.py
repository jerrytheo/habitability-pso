import numpy as np
# from scipy.misc import derivative

_threshold = 1e-4
_learningr = 0.001


def optimize_cdhs_by_sga(R, D, Ts, Ve):
    prev = np.array([.5, .05])
    prev = prev / np.sum(prev)

    step = prev
    # coef = np.log(np.array([R, D, Ts, Ve]))
    coef = np.log(np.array([R, D]))
    lamb = 0.1

    while np.any(step > _threshold):
        curr = prev + _learningr * (coef - lamb / np.sum(prev))
        curr = curr / np.sum(curr)
        lamb -= np.log(np.sum(prev))
        step = np.abs(curr - prev)
        prev = curr
        print(prev)
    return np.prod(np.exp(coef * prev)), prev


if __name__ == '__main__':
    print(optimize_cdhs_by_sga(1.69, 1.11, 1, 1))
