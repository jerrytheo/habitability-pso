import csv
import numpy as np
from os import path

from .cdhs_fn import construct_cdhpf
from ..pso import converge


# Miscellaneous Consts.
MESSAGE = '{:25}{:>6}{:>6}{:>10}{:>6}{:>6}{:>10}{:>10}{:>7.5}'
ERROR = '{:25}{:^57}'
PROGRESS_BAR = '[{:72}]  ({:>3}%)'
HEADERS = ('Name', 'A', 'B', 'CDHSi', 'G', 'D', 'CDHSs', 'CDHS', 'Cls')
ERR_CDHSi = '** CDHSi convergence failed. **'
ERR_CDHSs = '** CDHSs convergence failed. **'
TOTAL_CHAR = 86


# Function to evaluate CDHS values.
def evaluate_cdhs_values(exoplanets, fname, swkwargs, verbose=True):
    """Evaluates the CDHS values of each exoplanet and stores it in
    the file given by os.path.join('res', fname.format(constraint)).
    """
    exoplanets.dropna(how='any', inplace=True)
    exoplanets.reset_index(drop=True, inplace=True)

    total = len(exoplanets)
    if not verbose:
        def myprint(*args, **kwargs): pass
    else:
        myprint = print

    for constraint in ['crs', 'drs']:
        results = [HEADERS]

        spaces = (TOTAL_CHAR//2 - len(constraint)//2) * ' '
        myprint('\n' + spaces + constraint.upper())
        myprint(spaces + '-'*len(constraint) + '\n')
        myprint(MESSAGE.format(*results[-1]))
        myprint('-' * TOTAL_CHAR)

        for _, row in exoplanets.iterrows():
            name = row['Name']
            habc = row['Habitable']

            r = row['Radius']
            d = row['Density']
            v = row['Escape']
            t = row['STemp']

            # CDHS interior.
            cdhpf_i = construct_cdhpf(r, d, constraint)
            swarm_i = converge(fn=cdhpf_i, **swkwargs)
            if not swarm_i:
                myprint(ERROR.format(name, ERR_CDHSi))
                continue

            A, B = np.round(swarm_i.best_particle.best, 2)
            cdhs_i = round(swarm_i.global_best, 4)

            # CDHS surface.
            cdhpf_s = construct_cdhpf(v, t, constraint)
            swarm_s = converge(fn=cdhpf_s, **swkwargs)
            if not swarm_s:
                myprint(ERROR.format(name, ERR_CDHSs))
                continue

            G, D = np.round(swarm_s.best_particle.best, 2)
            cdhs_s = round(swarm_s.global_best, 4)

            cdhs = np.round(cdhs_i*.99 + cdhs_s*.01, 4)
            results.append((name, A, B, cdhs_i, G, D, cdhs_s, cdhs, habc))
            myprint(MESSAGE.format(*results[-1]))

            ii = _ + 1
            prog = (ii * (TOTAL_CHAR - 10)) // total
            myprint(PROGRESS_BAR.format('='*prog, (ii*100)//total), end='\r')

        myprint('-' * TOTAL_CHAR + '\n')

        fpath = path.join('res', fname.format(constraint))
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)

    myprint('')
