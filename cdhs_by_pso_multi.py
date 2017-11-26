#!/usr/bin/python

import csv
import functools
import logging
from multiprocessing import Lock, Process
import numpy as np

from exoplanets import exoplanets
from pso import Swarm


# Constants.
HEADERS = ('Name', 'A', 'B', 'CDHSi', 'Conv_i',
           'G', 'D', 'CDHSs', 'Conv_s', 'CDHS', 'Cls')


# Setting up logging.
logging.basicConfig(format='[%(asctime)s]  %(message)s', level=logging.INFO)


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


# Utility functions.
def safe_log(logfunc, lock, pname, message):
    """Multiprocess safe logging."""
    lock.acquire()
    try:
        logfunc(pname+'  '+message)
    finally:
        lock.release()


# Functions to set up the penalty.
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


# Functions to evaluate the CDHS values for each exoplanet.
def converge_by_pso(restarts=3, **kwargs):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        swarm = Swarm(**kwargs)
        converged = swarm.converge(verbose=False)
        if converged:
            return (converged, swarm)
    return (0, None)


def evaluate_cdhs_values(lock, pname, **sw_kwargs):
    """Estimate the CDHS values of each planet for both, the CRS and
    the DRS constraints.
    """
    global exoplanets
    num_exopl = len(exoplanets)

    for constraint in ['CRS', 'DRS']:
        safe_log(logging.info, lock, pname,
                 'Commencing '+constraint+' estimation.')
        results = []
        results.append(HEADERS)

        curr = 1
        for _, row in exoplanets.iterrows():
            # CDHSi
            cdhpf_i = construct_cdhs_fn(row['Radius'], row['Density'],
                                        constraint)
            conv_i, swarm_i = converge_by_pso(fn=cdhpf_i, **sw_kwargs)
            if not swarm_i:
                safe_log(logging.ERROR, lock, pname,
                         row['Name']+'did not converge for CDHSi.')
                curr += 1
                continue

            A, B = np.round(swarm_i.best_particle.best, 2)
            cdhs_i = round(swarm_i.global_best, 4)

            # CDHSs
            row['STemp'] = row['STemp'] / 288           # Normalizing to EU.
            cdhpf_s = construct_cdhs_fn(row['Escape'], row['STemp'],
                                        constraint)
            conv_s, swarm_s = converge_by_pso(fn=cdhpf_s, **sw_kwargs)
            if not swarm_s:
                safe_log(logging.ERROR, lock, pname,
                         row['Name']+'did not converge for CDHSs.')
                curr += 1
                continue

            G, D = np.round(swarm_s.best_particle.best, 2)
            cdhs_s = round(swarm_s.global_best, 4)

            cdhs = np.round(cdhs_i*.99 + cdhs_s*.01, 4)
            results.append([row['Name'], A, B, cdhs_i, conv_i,
                            G, D, cdhs_s, conv_s, cdhs, row['Habitable']])

            q, r = divmod(curr*100 // num_exopl, 10)
            if r == 0 and q != 0:
                safe_log(logging.info, lock, pname, str(q)+'% complete.')
            curr += 1

        safe_log(logging.info, lock, pname,
                 'Completed '+constraint+' estimation.')
        fpath = 'res/pso_{0}_{1}.csv'.format(constraint.lower(),
                                             sw_kwargs['npart'])
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)
        safe_log(logging.info, lock, pname, 'Written results.')


if __name__ == '__main__':
    processes = []
    lock = Lock()

    logging.info('MAINPROG  Commencing program execution.')
    for npart in range(50, 151, 10):
        sw_kwargs['npart'] = npart
        pname = 'NPART{:03}'.format(npart)
        processes.append(Process(target=evaluate_cdhs_values,
                                 args=(lock, pname,), kwargs=sw_kwargs))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    logging.info('MAINPROG  Completed program execution.')
