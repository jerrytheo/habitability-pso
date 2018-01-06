#!/usr/bin/python

import csv
import numpy as np
from os import path

from .ceesa_fn import construct_ceesa
from ..exoplanets import exoplanets
from ..pso import converge


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
MESSAGE = '{:25}{:>7.5}{:>6}{:>6}{:>6}{:>6}{:>6}{:>6}{:>6}{:>10}'
ERROR = '{:25}{:^59}'
PROGRESS_BAR = '[{:72}]  ({:>3}%)'
HEADERS = ('Name', 'Cls', 'r', 'd', 't', 'v', 'e', 'Rho', 'Eta', 'Hab')
TITLE = MESSAGE.format(*HEADERS)
ERROR_TEXT = '** Convergence failed. **'


# Function to evaluate CEESA values.
def evaluate_ceesa_values(exoplanets, fname, swkwargs, verbose=True):
    """Evaluates the CDHS values of each exoplanet and stores it in
    the file given by os.path.join('res', fname.format(constraint)).
    """
    total = len(exoplanets)
    if not verbose:
        def myprint(*args, **kwargs): pass
    else:
        myprint = print

    for constraint, ndim in (('crs', 6), ('drs', 7)):
        results = [HEADERS]
        print('\n' + MESSAGE.format(*results[-1]))

        myprint('\n' + ' '*40 + constraint.upper() + ' '*40)
        myprint(' '*40 + '-'*len(constraint) + ' '*40 + '\n')
        myprint(MESSAGE.format(*results[-1]))
        myprint('-' * 82)

        for _, row in exoplanets.iterrows():
            name = row['Name']
            habc = row['Habitable']
            info = row[['Radius', 'Density', 'STemp',
                        'Escape', 'Eccentricity']]

            ceesa = construct_ceesa(info, constraint)
            swarm = converge(fn=ceesa, **swkwargs)
            if not swarm:
                myprint(ERROR.format(name, ERROR_TEXT))
                continue

            # Calculate the score here.
            best = swarm.best_particle
            score = swarm.global_best

            results.append([name, habc, best, score])
            myprint(MESSAGE.format(*results[-1]))

            ii = _ + 1
            prog = (ii * 72) // total
            myprint(PROGRESS_BAR.format('='*prog, (ii*100)//total), end='\r')

        myprint('-' * 82 + '\n')

        fpath = path.join('res', fname.format(constraint))
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)

    myprint('')


# When executed as a script and not a module.
if __name__ == '__main__':
    exoplanets.dropna(how='all', inplace=True)
    exoplanets.replace(np.nan, 0, inplace=True)
    exoplanets.reset_index(drop=True, inplace=True)
    exoplanets = exoplanets[:5]
    evaluate_ceesa_values(exoplanets, 'ceesa_{0}.csv', SWARM_PARAMS)
