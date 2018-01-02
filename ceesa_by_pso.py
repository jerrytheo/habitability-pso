#!/usr/bin/python

import csv
import numpy as np
from exoplanets import exoplanets
from pso import Swarm
# import sys


# Constants
HEADERS = ('Name', 'Cls', 'r', 'd', 't', 'v', 'e', 'Rho', 'Eta', 'Hab')
MESSAGE = '{:25}{:>7.5}{:>6}{:>6}{:>6}{:>6}{:>6}{:>6}{:>6}{:>10}'
ERROR = '{:25}{:^59}'
TITLE = MESSAGE.format(*HEADERS)


# Parameters for the swarm.
sw_kwargs = {
    'npart': 100,                   # Number of particles.
    'ndim': 2,                      # Dimensions of input.
    'min_': 0,                      # Min. value for initial pos.
    'max_': 1,                      # Max. value for initial pos.
    'friction': .8,                 # Friction coefficient.
    'learn_rate1': 1e-3,            # c1 learning rate.
    'learn_rate2': 1e-3,            # c2 learning rate.
    'max_velocity': .1,             # Max. velocity.
}


def penalty_drs(pos, rho, eta):
    """The penalty function for CEESA with DRS."""
    return ((np.sum(pos) == 1) + np.sum((pos <= 0) | (pos >= 1)) +
            (0 < rho <= 0) + (0 < eta < 1)) * 1e8


def penalty_crs(pos, rho):
    """The penalty function for CEESA with CRS."""
    return ((np.sum(pos) == 1) + np.sum((pos <= 0) | (pos >= 1)) +
            (0 < rho <= 0)) * 1e8


def construct_ceesa_fn(planet, type_):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    if type_ == 'DRS':
        def ceesa(pos):
            """Calculate CEESA with DRS."""
            return np.sum(planet * (pos[:5]**pos[5])) ** (pos[6]/pos[5]) - \
                penalty_drs(pos[:5], pos[5], pos[6])

    elif type_ == 'CRS':
        def ceesa(pos):
            """Calculate CEESA with CRS."""
            return np.sum(planet * (pos[:5]**pos[5])) ** (1/pos[5]) - \
                penalty_crs(pos[:5], pos[5])

    return ceesa


def converge_by_pso(restarts=3, **kwargs):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        swarm = Swarm(**kwargs)
        converged = swarm.converge(verbose=False)
        if converged:
            return swarm
    return None


def evaluate_values(type_):
    global sw_kwargs
    results = [HEADERS]
    print('\n' + MESSAGE.format(*results[-1]))

    for _, row in exoplanets.iterrows():
        row = row[['Radius', 'Density', 'STemp', 'Escape', 'Eccentricity']]
        ceesa = construct_ceesa_fn(row, type_)
        swarm = converge_by_pso(fn=ceesa, **sw_kwargs)
        if not swarm:
            print(ERROR.format(row['Name'], '** Convergence failed. **'))
            continue

        # Calculate the score here.
        opt_pos = swarm.best_particle
        hab_score = swarm.global_best

        results.append([row['Name'], row['Habitable'], opt_pos, hab_score])
        print(MESSAGE.format(*results[-1]))

    print('\n')

    fpath = 'res/ceesa_{0}.csv'.format(type_)
    with open(fpath, 'w') as resfile:
        csv.writer(resfile).writerows(results)


# Execution begins here.
if __name__ == '__main__':
    # Drop rows with all missing values. Replace with 0, otherwise.
    # Scale temperature to EU.
    exoplanets.dropna(how='all', inplace=True)
    exoplanets.replace(np.nan, 0, inplace=True)

    # Temporary for testing.
    exoplanets = exoplanets[1:20]

    # Round pos; Get Elasticity from original file.
    sw_kwargs['ndim'] = 6
    evaluate_values('CRS')
    
    sw_kwargs['ndim'] = 7
    evaluate_values('DRS')
