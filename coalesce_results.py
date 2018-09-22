from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import re


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



def extract_range(planet_info, constraint):
    for planet, habc in planet_info:
        fname = 'temp/{1}/{0}-{1}.txt'.format(planet, constraint)
        hab_range = np.genfromtxt(fname)

        if planet in PLANETS_TO_PLOT:
            plt.plot(np.arange(len(hab_range)), hab_range, color=COLORS[habc])

    plt.show()

    return None


def write_range(hab_interval, constraint):
    pass


if __name__ == '__main__':

    # First row has headers, first column is planet name.
    planet_info = np.genfromtxt(
        'results/ceesa_crs.csv', dtype=str, delimiter=',')[1:, 0:2]

    for constraint in ('crs', 'drs'):
        hab_interval = extract_range(planet_info, constraint)
        write_range(hab_interval, constraint)
        break