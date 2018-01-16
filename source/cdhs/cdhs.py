import csv
import numpy as np
from os import path

from .cdhs_fn import construct_fitness
from .cdhs_fn import get_constraint_fn
from .cdhs_fn import initialize_points
from ..pso import conmax_by_pso, SwarmConvergeError


# Miscellaneous Consts.
MESSAGE = '{:25}{:8.4f}{:8.4f}{:10.4f}{:8.4f}{:8.4f}{:10.4f}{:10.4f}{:>7.5}'
TITLE = '{:25}{:>8}{:>8}{:>10}{:>8}{:>8}{:>10}{:>10}{:>7.5}'
ERROR = '{:25}{:^69}'
HEADERS = ('Name', 'A', 'B', 'CDHSi', 'G', 'D', 'CDHSs', 'CDHS', 'Cls')
ERR_CDHSi = '** CDHSi convergence failed. **'
ERR_CDHSs = '** CDHSs convergence failed. **'
TOTAL_CHAR = 94
PROGRESS_BAR = '[{:' + str(TOTAL_CHAR - 10) + '}]  ({:>3}%)'


# Print functions.
def print_header(constraint, headers):
    """Print the header of the output table."""
    spaces = (TOTAL_CHAR//2 - len(constraint)//2) * ' '

    print('\n' + spaces + constraint.upper())
    print(spaces + '-'*len(constraint) + '\n')
    print(TITLE.format(*headers))

    print('-' * TOTAL_CHAR, end='\t\t')
    print('  inn : sur  ')


def print_error(name, err):
    """Print the error for exoplanet name when convergence fails."""
    print(ERROR.format(name, err))


def print_results(it, total, values, iters):
    """Print the results of the estimation for the current planet."""
    print(MESSAGE.format(*values), end='\t\t')
    print('[ {:03} : {:03} ]'.format(*iters))

    prog = (it * (TOTAL_CHAR - 10)) // total
    print(PROGRESS_BAR.format('='*prog, (it*100)//total), end='\r')


# Function to evaluate CDHS values.
def evaluate_cdhs_values(exoplanets, fname='cdhs_{0}.csv', verbose=True,
                         npart=25, **kwargs):
    """Evaluates the CDHS values of each exoplanet and stores it in the
    indicated file.

    Arguments:
        exoplanets: pandas.DataFrame
            The exoplanet parameters to operate on. Should contain columns for
            Radius, Density, Escape (Escape Velocity) and STemp (Surface Temp)
            in EU (Earth Units).
        fname: str, default 'cdhs_{0}.csv'
            A format string indicating where to store the results. The final
            filename is given by
                > os.path.join('res', fname.format(constraint))
            where constraint is either 'crs' or 'drs'.
        verbose: bool, default True
            Whether to print output to stdout.
        npart: int, default 25
            Number of particles.
        kwargs:
            The parameters for the Swarm.
    """
    total = len(exoplanets)

    for constraint in ('crs', 'drs'):
        results = [HEADERS]
        check = get_constraint_fn(constraint)

        if verbose:
            print_header(constraint, results[-1])

        for _, row in exoplanets.iterrows():
            name = row['Name']
            habc = row['Habitable']

            rad = row['Radius']
            den = row['Density']
            vel = row['Escape']
            tem = row['STemp']

            # CDHS interior.
            cdhpf = construct_fitness(rad, den, constraint)
            start = initialize_points(npart, constraint)

            try:
                gbest, it_i = conmax_by_pso(cdhpf, start, check, **kwargs)
            except SwarmConvergeError:
                print_error(name, ERR_CDHSi)
                continue

            A, B = np.round(gbest, 4)
            cdhs_i = np.round(cdhpf(gbest), 4)

            # CDHS surface.
            cdhpf = construct_fitness(vel, tem, constraint)
            start = initialize_points(npart, constraint)

            try:
                gbest, it_s = conmax_by_pso(cdhpf, start, check, **kwargs)
            except SwarmConvergeError:
                print_error(name, ERR_CDHSs)
                continue

            G, D = np.round(gbest, 4)
            cdhs_s = np.round(cdhpf(gbest), 4)

            cdhs = np.round(cdhs_i*.99 + cdhs_s*.01, 4)
            results.append((name, A, B, cdhs_i, G, D, cdhs_s, cdhs, habc))

            if verbose:
                print_results(_+1, total, results[-1], [it_i-99, it_s-99])

        if verbose:
            print('-' * TOTAL_CHAR + '\n')

        fpath = path.join('results', fname.format(constraint))
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)

    if verbose:
        print('')
