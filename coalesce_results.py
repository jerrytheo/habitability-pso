import sys
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd


PRECISION = 1e-5
PLANETS_TO_PLOT = (
    'GJ 176 b',
    'GJ 667 C b',
    'GJ 667 C e',
    'GJ 667 C f',
    'GJ 3634 b',
    'HD 20794 c',
    'HD 40307 e',
    'HD 40307 f',
    'HD 40307 g',
    'Kepler-186 f',
    'Proxima Cen b',
    'TRAPPIST-1 b',
    'TRAPPIST-1 c',
    'TRAPPIST-1 d',
    'TRAPPIST-1 e',
    'TRAPPIST-1 g',
)
COLORS = {
    'non-habitable': 'r',
    'psychroplanet': 'g',
    'mesoplanet': 'b',
    'hypopsychroplanet': 'm',
    'thermoplanet': 'y'
}


# def process_constraint(*args):
#     with open('temp/{0}/{1}-{0}.txt'.format(*args)) as file:
#         vals = [float(i.rstrip()) for i in file]

#     while math.isclose(vals[-1], vals[-2], abs_tol=PRECISION):
#         vals.pop()
#     print(*vals[-20:], sep='\n')
#     sys.exit(0)


def extract_range(planets, constraint):
    hab_intervals = []

    for planet in planets:
        fname = path.join(
            'temp', constraint, '{0}-{1}.txt'.format(planet, constraint))
        hab_range = np.genfromtxt(fname)[:-99]
        if hab_range.shape[0] > 50:
            hab_range = hab_range[-50:]
        hab_intervals.append([hab_range.min(), hab_range.max()])

    return np.array(hab_intervals)


def write_range(hab_interval, constraint):
    df = pd.read_csv(path.join('results', 'ceesa_{}.csv'.format(constraint)))
    df['ScoreMin'] = hab_intervals[:, 0]
    df['ScoreMax'] = hab_intervals[:, 1]
    df.to_csv('results/ceesa_{}_app.csv'.format(constraint), index=False)


if __name__ == '__main__':

    # First row has headers, first column is planet name.
    planets = np.genfromtxt(
        'results/ceesa_crs.csv', dtype=str, delimiter=',')[1:, 0]

    for constraint in ('crs', 'drs'):
        hab_intervals = extract_range(planets, constraint)
        write_range(hab_intervals, constraint)