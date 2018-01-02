#!/usr/bin/python

from cdhs import construct_cdhpf
import csv
from exoplanets import exoplanets
import numpy as np
from pso import Swarm
from os import path

# Parameters for the swarm.
SWARM_PARAMS = {
    'npart': 100,                   # Number of particles.
    'ndim': 2,                      # Dimensions of input.
    'min_': 0,                      # Min. value for initial pos.
    'max_': 1,                      # Max. value for initial pos.
    'friction': .8,                 # Friction coefficient.
    'learn_rate1': 1e-3,            # c1 learning rate.
    'learn_rate2': 1e-3,            # c2 learning rate.
    'max_velocity': .1,             # Max. velocity.
}

# Miscellaneous Consts.
MESSAGE = '{:25}{:>5}{:>5}{:>10}{:>5}{:>5}{:>10}{:>10}{:>7.5}'
ERROR = '{:25}{:^57}'
PROGRESS_BAR = '[{:72}]  ({:>3}%)'
HEADERS = ('Name', 'A', 'B', 'CDHSi', 'G', 'D', 'CDHSs', 'CDHS', 'Cls')
ERR_CDHSi = '** CDHSi convergence failed. **'
ERR_CDHSs = '** CDHSs convergence failed. **'


# Function for convergence.
def converge_by_pso(restarts=3, **kwargs):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        swarm = Swarm(**kwargs)
        converged = swarm.converge(verbose=False)
        if converged:
            return swarm
    return None


def evaluate_cdhs_values(exoplanets, fpath):
    """Evaluates the CDHS values of each exoplanet and stores it in
    the file given by os.path.join('res', fpath.format(constraint)).
    """
    global SWARM_PARAMS
    total = len(exoplanets)

    for constraint in ['CRS', 'DRS']:
        results = []
        results.append(HEADERS)

        print('\n' + ('=' * 82), end='\n\n')
        print('#', constraint, end='\n\n')
        print(MESSAGE.format(*results[-1]))
        print('-' * 82)

        for _, row in exoplanets.iterrows():
            name = row['Name']
            habc = row['Habitable']

            r = row['Radius']
            d = row['Density']
            v = row['Escape']
            t = row['STemp']

            # CDHS interior.
            cdhpf_i = construct_cdhpf(r, d, constraint)
            swarm_i = converge_by_pso(fn=cdhpf_i, **SWARM_PARAMS)
            if not swarm_i:
                print(ERROR.format(name, ERR_CDHSi))
                continue

            A, B = np.round(swarm_i.best_particle.best, 2)
            cdhs_i = round(swarm_i.global_best, 4)

            # CDHS surface.
            cdhpf_s = construct_cdhpf(v, t, constraint)
            swarm_s = converge_by_pso(fn=cdhpf_s, **SWARM_PARAMS)
            if not swarm_s:
                print(ERROR.format(name, ERR_CDHSs))
                continue

            G, D = np.round(swarm_s.best_particle.best, 2)
            cdhs_s = round(swarm_s.global_best, 4)

            cdhs = np.round(cdhs_i*.99 + cdhs_s*.01, 4)
            results.append((name, A, B, cdhs_i, G, D, cdhs_s, cdhs, habc))
            print(MESSAGE.format(*results[-1]))

            progress = (_ * 72) // total
            print(PROGRESS_BAR.format('='*progress, (_*100)//total), end='\r')

        print('\n')

        fpath = path.join('res', fpath.format(constraint))
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)

    print('=' * 82, end='\n\n')


# Execution begins here.
if __name__ == '__main__':
    exoplanets.dropna(how='any', inplace=True)
    exoplanets = exoplanets[1:20]

    evaluate_cdhs_values(exoplanets, 'pso_{0}.csv')
