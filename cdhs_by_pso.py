#!/usr/bin/python

import functools
import numpy as np
from exoplanets import exoplanets
from pso import Swarm

# Required constants.
SCALE = 20
NPART = 100


# Drop rows with missing values.
exoplanets.dropna(how='any', inplace=True)


def penalty(pos, type_):
    """The penalty function for the Cobb Douglas function."""
    q = np.array((
        max(-pos[0], 0),                    # pos[0] > 0
        max(-pos[1], 0),                    # pos[1] > 0
        max(pos[0]-1, 0),                   # pos[0] < 1
        max(pos[1]-1, 0),                   # pos[1] < 1
        0                                   # either CRS or DRS constraint.
    ), dtype=np.float)

    if type_ == 'CRS':
        q[4] = np.abs(pos[0]+pos[1]-1)
    elif type_ == 'DRS':
        q[4] = max(pos[0]+pos[1]-1+1e-3, 0)

    theta = np.piecewise(q, [q == 0, q > 0, q > .5, q > 1.], [0, 2., 4., 8.])
    gamma = np.piecewise(q, [q <= 1, q > 1], [1, 2])
    return np.sum(theta*(q**gamma))


def construct_cdhs_function(coeff1, coeff2, type_):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    penalty_ = functools.partial(penalty, type_=type_)

    def cdhpf(pos):
        return (coeff1 ** pos[0]) * (coeff2 ** pos[1]) - penalty_(pos)

    return cdhpf


print('{:25}{:>10}{:>10}{:>10}'.format('Name', 'Alpha', 'Beta', 'Value'))
print('{:25}{:>10}{:>10}{:>10}'.format('----', '-----', '----', '-----'))

for _, row in exoplanets.iterrows():
    # Estimating CDHSi at CRS.
    cdhsi_crs = construct_cdhs_function(coeff1=row['Radius'],
                                        coeff2=row['Density'],
                                        type_='CRS')
    swarm = Swarm(npart=NPART,
                  ndim=2,
                  fitness_function=cdhsi_crs,
                  min_=0,
                  max_=1,
                  friction=.8,
                  learn_rate1=.001,
                  learn_rate2=.001,
                  max_velocity=.1)
    swarm.converge(max_iter=1000, max_stable=10, verbose=True)
    best = swarm.best_particle.best
    print('{:25}{:10.4}{:10.4}{:10.4}'.format(row['Name'], best[0], best[1],
                                              swarm.global_best))
    break
