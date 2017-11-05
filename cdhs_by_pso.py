#!/usr/bin/python

import csv
import functools
import numpy as np
from exoplanets import exoplanets
from pso import Swarm

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


# Drop rows with missing values.
exoplanets.dropna(how='any', inplace=True)


def penalty(pos, type_):
    """The penalty function for the Cobb Douglas function."""
    e = .1
    q = np.array((
        max(-pos[0] + e, 0),                    # pos[0] > 0
        max(-pos[1] + e, 0),                    # pos[1] > 0
        max(pos[0]-1 + e, 0),                   # pos[0] < 1
        max(pos[1]-1 + e, 0),                   # pos[1] < 1
        0                                       # either CRS or DRS constraint.
    ), dtype=np.float)

    if type_ == 'CRS':
        q[4] = np.abs(pos[0]+pos[1]-1)
    elif type_ == 'DRS':
        q[4] = max(pos[0]+pos[1]-1 + e, 0)

    theta = np.piecewise(q, [q == 0, q > 0], [0, 1e8])
    gamma = np.piecewise(q, [q <= 1, q > 1], [1, np.prod(np.exp(pos))])
    return np.sum(theta*(q**gamma))


def construct_cdhs_fn(coeff1, coeff2, type_):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    penalty_ = functools.partial(penalty, type_=type_)

    def cdhpf(pos):
        pos = np.round(pos, 1)
        return (coeff1 ** pos[0]) * (coeff2 ** pos[1]) - penalty_(pos)

    return cdhpf


def converge_by_pso(restarts=3, **kwargs):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        swarm = Swarm(**kwargs)
        converged = swarm.converge(verbose=False)
        if converged:
            return swarm
    return None


# Message text.
msg = '{:25}{:>5}{:>5}{:>10}{:>5}{:>5}{:>10}{:>10}{:>7.5}'
err = '{:25}{:^57}'
bar = '[{:72}]  ({:>3}%)'
tot = len(exoplanets)

results = {'CRS': [], 'DRS': []}

# Estimating CDHS at CRS.
print('')
for constraint in ['CRS', 'DRS']:
    results = []
    results.append(['Name', 'A', 'B', 'CDHSi', 'G', 'D', 'CDHSs',
                    'CDHS', 'Cls'])

    print('=' * 82, end='\n\n')
    print('#', constraint, end='\n\n')
    print(msg.format(*results[-1]))
    print('-' * 82)

    curr = 1
    for _, row in exoplanets.iterrows():
        # CDHSi
        cdhpf_i = construct_cdhs_fn(row['Radius'], row['Density'], constraint)
        swarm_i = converge_by_pso(fn=cdhpf_i, **sw_kwargs)
        if not swarm_i:
            print(err.format(row['Name'], '** CDHSi convergence failed. **'))
            curr += 1
            continue

        A, B = np.round(swarm_i.best_particle.best, 2)
        cdhs_i = round(swarm_i.global_best, 4)

        # CDHSs
        row['STemp'] = row['STemp'] / 288           # Normalizing to EU.
        cdhpf_s = construct_cdhs_fn(row['Escape'], row['STemp'], constraint)
        swarm_s = converge_by_pso(fn=cdhpf_s, **sw_kwargs)
        if not swarm_s:
            print(err.format(row['Name'], '** CDHSs convergence failed. **'))
            curr += 1
            continue

        G, D = np.round(swarm_s.best_particle.best, 2)
        cdhs_s = round(swarm_s.global_best, 4)

        cdhs = np.round(cdhs_i*.99 + cdhs_s*.01, 4)
        results.append([row['Name'], A, B, cdhs_i, G, D, cdhs_s, cdhs,
                        row['Habitable']])
        print(msg.format(*results[-1]))

        progress = (curr*66) // tot
        print(bar.format('='*progress, (curr*100)//tot), end='\r')
        curr += 1
    print('\n')

    with open('res/pso_' + constraint.lower() + '.csv', 'w') as resfile:
        csv.writer(resfile).writerows(results)

print('=' * 82, end='\n\n')
