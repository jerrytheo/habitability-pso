import csv
import numpy as np
from os import path

from .ceesa_fn import construct_fitness
from .ceesa_fn import get_constraint_fn
from .ceesa_fn import initialize_points
from ..pso import conmax_by_pso, SwarmConvergeError


# Miscellaneous Consts.
MESSAGE = '{:25}{:>7.5}{:8.4f}{:8.4f}{:8.4f}{:8.4f}{:8.4f}'\
        '{:8.4f}{:8.4f}{:10.4f}{:6}'
TITLE = '{:25}{:>7}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>10}{:>6}'
ERROR = '{:25}{:^79}'
HEADERS = ('Name', 'Cls', 'r', 'd', 't', 'v', 'e',
           'Rho', 'Eta', 'CEESA', 'Iter')
ERR_TEXT = '** Convergence failed. **'
TOTAL_CHAR = 104
PROGRESS_BAR = '[{:' + str(TOTAL_CHAR - 10) + '}]  ({:>3}%)'


# Print functions.
def print_header(constraint, headers):
    """Print the header of the output table."""
    spaces = (TOTAL_CHAR//2 - len(constraint)//2) * ' '

    print('\n' + spaces + constraint.upper())
    print(spaces + '-'*len(constraint) + '\n')
    print(TITLE.format(*headers))

    print('-' * TOTAL_CHAR)


def print_error(name):
    """Print the error for exoplanet name when convergence fails."""
    print(ERROR.format(name, ERR_TEXT))


def print_results(it, total, values):
    """Print the results of the estimation for the current planet."""
    print(MESSAGE.format(*values), end='\n')
    prog = (it * (TOTAL_CHAR - 10)) // total
    print(PROGRESS_BAR.format('='*prog, (it*100)//total), end='\r')


# Function to evaluate CEESA values.
def evaluate_ceesa_values(exoplanets, fname='ceesa_{0}.csv', verbose=True,
                          npart=25, **kwargs):
    """Evaluates the CEESA scores of each exoplanet and stores it in the
    indicated file.

    Arguments:
        exoplanets: pandas.DataFrame
            The exoplanet parameters to operate on. Should contain columns for
            Radius, Density, Escape (Escape Velocity), STemp (Surface Temp) and
            Eccentricity in EU (Earth Units).
        fname: str, default 'ceesa_{0}.csv'
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
            info = row[['Radius', 'Density', 'STemp',
                        'Escape', 'Eccentricity']]

            ceesa = construct_fitness(*info, constraint)
            start = initialize_points(npart, constraint)

            kwargs['dumpfile'] = path.join(
                'temp', '{0}-{1}.txt'.format(name, constraint))
            try:
                gbest, it = conmax_by_pso(ceesa, start, check, **kwargs)
            except SwarmConvergeError:
                print_error(name)
                continue
            score = np.round(ceesa(gbest), 4)
            if constraint == 'crs':
                results.append([name, habc, *np.round(gbest, 4), 1,
                                score, it-99])
            else:
                results.append([name, habc, *np.round(gbest, 4), score, it-99])

            if verbose:
                print_results(_+1, total, results[-1])

        if verbose:
            print('-' * TOTAL_CHAR + '\n')

        fpath = path.join('results', fname.format(constraint))
        with open(fpath, 'w', newline='') as resfile:
            csv.writer(resfile).writerows(results)

    if verbose:
        print('')
