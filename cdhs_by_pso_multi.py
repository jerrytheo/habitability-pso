#!/usr/bin/python

from cdhs import construct_cdhpf
import csv
from exoplanets import exoplanets
import logging
from multiprocessing import Lock, Process
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

# Miscellaneous Constants.
HEADERS = ('Name', 'A', 'B', 'CDHSi', 'Conv_i',
           'G', 'D', 'CDHSs', 'Conv_s', 'CDHS', 'Cls')


# Setting up logging.
logging.basicConfig(format='[%(asctime)s]  %(message)s', level=logging.INFO)


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


# Function for convergence.
def converge_by_pso(restarts=3, **kwargs):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        swarm = Swarm(**kwargs)
        converged = swarm.converge(verbose=False)
        if converged:
            return (converged, swarm)
    return (0, None)


def evaluate_cdhs_values(exoplanets, lock, fname):
    """Evaluates the CDHS values of each exoplanet and stores it in
    the file given by os.path.join('res', fname.format(constraint)).
    """
    global SWARM_PARAMS
    total = len(exoplanets)

    for constraint in ('crs', 'drs'):
        safe_log(logging.info, lock, pname,
                 'Commencing '+constraint+' estimation.')
        results = []
        results.append(HEADERS)

        curr = 1
        for _, row in exoplanets.iterrows():
            # CDHSi
            cdhpf_i = construct_cdhs_fn(row['Radius'], row['Density'],
                                        constraint)
            conv_i, swarm_i = converge_by_pso(fn=cdhpf_i, **SWARM_PARAMS)
            if not swarm_i:
                safe_log(logging.ERROR, lock, pname,
                         row['Name']+'did not converge for CDHSi.')
                curr += 1
                continue

            A, B = np.round(swarm_i.best_particle.best, 2)
            cdhs_i = round(swarm_i.global_best, 4)

            # CDHSs
            cdhpf_s = construct_cdhs_fn(row['Escape'], row['STemp'],
                                        constraint)
            conv_s, swarm_s = converge_by_pso(fn=cdhpf_s, **SWARM_PARAMS)
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

            q, r = divmod(curr*100 // total, 10)
            if r == 0 and q != 0:
                safe_log(logging.info, lock, pname, str(q*10)+'% complete.')
            curr += 1

        safe_log(logging.info, lock, pname,
                 'Completed '+constraint+' estimation.')
        fpath = 'res/multiple/pso_{0}_{1}.csv'.format(constraint.lower(),
                                                      SWARM_PARAMS['npart'])
        with open(fpath, 'w') as resfile:
            csv.writer(resfile).writerows(results)
        safe_log(logging.info, lock, pname, 'Written results.')


if __name__ == '__main__':
    processes = []
    lock = Lock()

    logging.info('MAINPROG  Commencing program execution.')
    for npart in range(50, 151, 10):
        SWARM_PARAMS['npart'] = npart
        pname = 'NPART{:03}'.format(npart)
        processes.append(Process(target=evaluate_cdhs_values,
                                 args=(lock, pname,), kwargs=SWARM_PARAMS))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    logging.info('MAINPROG  Completed program execution.')
