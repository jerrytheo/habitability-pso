#!/usr/bin/python

import functools
import numpy as np
from exoplanets import exoplanets
from pso import Swarm

# Parameters for the swarm.
sw_kwargs = {
    'npart': 1000,                 # Number of particles.
    'ndim': 2,                      # Dimensions of input.
    'min_': 0,                      # Min. value for initial pos.
    'max_': 1,                      # Max. value for initial pos.
    'friction': .8,                 # Friction coefficient.
    'learn_rate1': 1e-3,            # c1 learning rate.
    'learn_rate2': 1e-3,            # c2 learning rate.
    'max_velocity': .1,             # Max. velocity.
}


# Drop rows with missing values.
exoplanets.dropna(how='any', inplace=True)


def penalty(pos, type_):
    """The penalty function for the Cobb Douglas function."""
    e = .1
    q = np.array((
        max(-pos[0] + e, 0),                    # pos[0] > 0
        max(-pos[1] + e, 0),                    # pos[1] > 0
        max(pos[0]-1 + e, 0),                   # pos[0] < 1
        max(pos[1]-1 + e, 0),                   # pos[1] < 1
        0                                       # either CRS or DRS constraint.
    ), dtype=np.float)

    if type_ == 'CRS':
        q[4] = np.abs(pos[0]+pos[1]-1)
    elif type_ == 'DRS':
        q[4] = max(pos[0]+pos[1]-1 + e, 0)

    theta = np.piecewise(q, [q == 0, q > 0, q > .5, q > 1.], [0, 5., 10., 15.])
    gamma = np.piecewise(q, [q <= 1, q > 1], [1, 2])
    return np.sum(theta*(q**gamma))


def construct_cdhs_function(coeff1, coeff2, type_):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    penalty_ = functools.partial(penalty, type_=type_)

    def cdhpf(pos):
        pos = np.round(pos, 1)
        return (coeff1 ** pos[0]) * (coeff2 ** pos[1]) - penalty_(pos)

    return cdhpf


msg = '{:25}{:>10}{:>10}{:>10}{:>20}'
print(msg.format('Name', 'Alpha', 'Beta', 'Value', 'Class'))
print(msg.format('----', '-----', '----', '-----', '-----'))

for _, row in exoplanets.iterrows():
    # Estimating CDHSi at CRS.
    cdhsi_crs = construct_cdhs_function(row['Radius'], row['Density'], 'CRS')
    swarm = Swarm(fitness_function=cdhsi_crs, **sw_kwargs)
    swarm.converge(max_iter=100, max_stable=10, verbose=True, proglen=65)
    best = np.round(swarm.best_particle.best, 2)
    global_ = round(swarm.global_best, 4)
    print(msg.format(row['Name'], best[0], best[1], global_, row['Habitable']))
