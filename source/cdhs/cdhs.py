import csv
import numpy as np
from os import path

from .cdhs_fn import construct_cdhpf
from ..pso import converge


# Miscellaneous Consts.
MESSAGE = '{:25}{:>8.4f}{:>8.4f}{:>10.4f}'\
        '{:>8.4f}{:>8.4f}{:>10.4f}{:>10.4f}{:>7.5}'
TITLE = '{:25}{:>8}{:>8}{:>10}{:>8}{:>8}{:>10}{:>10}{:>7.5}'
ERROR = '{:25}{:^57}'
HEADERS = ('Name', 'A', 'B', 'CDHSi', 'G', 'D', 'CDHSs', 'CDHS', 'Cls')
ERR_CDHSi = '** CDHSi convergence failed. **'
ERR_CDHSs = '** CDHSs convergence failed. **'
TOTAL_CHAR = 94
PROGRESS_BAR = '[{:' + str(TOTAL_CHAR - 10) + '}]  ({:>3}%)'


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

    for constraint in ['crs']:
        results = [HEADERS]

        spaces = (TOTAL_CHAR//2 - len(constraint)//2) * ' '
        myprint('\n' + spaces + constraint.upper())
        myprint(spaces + '-'*len(constraint) + '\n')
        myprint(TITLE.format(*results[-1]))
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

            A, B = np.round(swarm_i.best_particle.best, 4)
            cdhs_i = round((r ** A) * (d ** B), 4)

            # CDHS surface.
            cdhpf_s = construct_cdhpf(v, t, constraint)
            swarm_s = converge(fn=cdhpf_s, **swkwargs)
            if not swarm_s:
                myprint(ERROR.format(name, ERR_CDHSs))
                continue

            G, D = np.round(swarm_s.best_particle.best, 4)
            cdhs_s = round((v ** G) * (t ** D), 4)

            cdhs = np.round(cdhs_i*.99 + cdhs_s*.01, 4)
            results.append((name, A, B, cdhs_i, G, D, cdhs_s, cdhs, habc))
            myprint(MESSAGE.format(*results[-1]), end='\t\t')
            myprint('[  {:7.4f}    {:7.4f}  ]'.format(A+B-1, G+D-1))

            ii = _ + 1
            prog = (ii * (TOTAL_CHAR - 10)) // total
            myprint(PROGRESS_BAR.format('='*prog, (ii*100)//total), end='\r')

        myprint('-' * TOTAL_CHAR + '\n')

        fpath = path.join('res', fname.format(constraint))
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)

    myprint('')
