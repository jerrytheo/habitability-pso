import numpy as np
from source import conmax_by_pso, SwarmConvergeError
from source.ceesa import construct_fitness
from source.ceesa import get_constraint_fn
from source.ceesa import initialize_points


print('Earth CEESA Score averaged over 50 iterations:')

for constraint in ('crs', 'drs'):
    npart = 25

    check = get_constraint_fn(constraint)
    ceesa = construct_fitness(1, 1, 1, 1, 0.0167, constraint)
    start = initialize_points(npart, constraint)

    score = 0
    iter_ = 0
    for ii in range(50):
        gbest, it = conmax_by_pso(
            ceesa, start, check, friction=.6, learnrate1=.8,
            learnrate2=.2, max_velocity=1.)
        score += gbest
        iter_ += it
    score /= 50
    iter_ /= 50

    print('\t', constraint.upper(), ':', np.round(ceesa(score), 4), iter_)
