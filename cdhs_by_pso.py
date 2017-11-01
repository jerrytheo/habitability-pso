#!/usr/bin/python

import functools
import numpy as np
from exoplanets import exoplanets
from pso import Swarm

# Required constants.
SCALE = 20
NPART = 20


# Drop rows with missing values.
exoplanets.dropna(how='any', inplace=True)


def penalty(pos, type_):
    """The penalty function for the Cobb Douglas function."""
    sum_ = np.sum(pos)
    if type_ == 'CRS' and sum_ == 1:
        return 0
    elif type_ == 'DRS' and sum_ < 1:
        return 0
    else:
        return SCALE * (1 + np.exp(np.abs(sum_ - 1)))


def construct_cdhs_function(coeff1, coeff2, type_):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    penalty_ = functools.partial(penalty, type_=type_)

    def cdhpf(pos):
        return (coeff1 ** pos[0]) * (coeff2 ** pos[1]) - penalty_(pos)

    return cdhpf


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
                  learn_rate1=0.1,
                  learn_rate2=0.1,
                  max_velocity=1.0)
    swarm.converge(max_iter=100, verbose=False)
    print(swarm.best_particle.position)
    break
