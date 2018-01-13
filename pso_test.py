#!/usr/bin/python

import numpy as np
from source import conmax_by_pso


def fun(x):
    return -((x.T[0]-2)**2 + (x.T[1]-1)**2)


def constraints(x):
    return np.apply_along_axis(lambda x: np.array((
        np.max((x[0] - 2*x[1] + 1 - 1e-7, 0)),
        np.max((-x[0] + 2*x[1] - 1 - 1e-7, 0)),
        np.max((((x[0]**2)/4) + (x[1]**2) - 1), 0)
    )), axis=1, arr=x)


start = np.random.uniform(0, 2, (25, 2))
gbest, it = conmax_by_pso(fun, start, constraints, learnrate1=.7,
                          learnrate2=.7, max_velocity=1.)
print(-1*fun(gbest))
