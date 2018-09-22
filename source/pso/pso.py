import numpy as np
from numpy.random import uniform
from scipy.spatial.distance import cdist


# Exception raised when Swarm does not converge.
class SwarmConvergeError(Exception):
    """Error raised when the swarm does not converge."""
    pass


# Function for convergence.
def conmax_by_pso(fitness, start_points, constraints, friction=.8,
                  learnrate1=.1, learnrate2=.1, max_velocity=1.,
                  max_iter=1000, stable_iter=100, thresh=1e-8, dumpfile=None):
    """Perform constrained maximization of the given fitness using particle
    swarm optimization.

    Arguments:
        fitness: function fitness(positions) -> fitness_values
            Function to maximize. The function should take as an argument a
            ndarray of shape (N, D) and return an 1d array of size N.
        start_points: ndarray of shape (N, D)
            Array of points to begin PSO from, where N is the number of points,
            and D the dimensionality of each point.
        constraints: function constraints(position) -> penalty_value
            Function that returns the constraint matrix. The function should
            take as an argument a ndarray of shape (N, D) and return a 2d array
            of size (N, S), where S is the number of constraints.
        friction: float, default 0.8
            Velocity is scaled by friction before updating.
                    velocity = friction*velocity + dv_g + dv_l
        learnrate1: float, default 0.1
            Global learning rate.
                    dv_g = learnrate1 * random(0, 1) * (gbest - current)
        learnrate2: float, default 0.1
            Local learning rate.
                    dv_l = learnrate2 * random(0, 1) * (lbest - current)
        max_velocity: float, default 1.0
            Threshold for velocity.
        max_iter: int, default 1000
            Maximum iterations to wait for convergence.
        stable_iter: int, default 100
            Number of iterations to wait before Swarm is declared stable.
        thresh: int, default 1e-5
            Threshold within which the Swarm is stable.
        dumpfile: str or None, default None
            File to write gbest values per iteration. None if no dump required.
    Returns:
        a 2-tuple (swarm, it), where swarm is the converged particle swarm and
        it are the number of iterations taken to converge.
    """
    # Initial position and velocity.
    position = start_points
    velocity = uniform(-max_velocity, max_velocity, position.shape)

    # Initial local best for each point and global best.
    lbest = position.copy()
    gbest = lbest[np.argmax(fitness(lbest))]

    if dumpfile is not None:
        dumpdata = []

    stable_count = 0

    for ii in range(max_iter):
        # Store old for threshold comparison.
        gbest_fit = fitness(gbest)
        if dumpfile is not None:
            dumpdata.append(gbest_fit)

        # Determine the velocity gradients.
        leaders = np.argmin(cdist(position, lbest, 'sqeuclidean'), axis=1)
        dv_g = learnrate1 * uniform(0, 1) * (gbest - position)
        dv_l = learnrate2 * uniform(0, 1) * (lbest[leaders] - position)

        # Update velocity such that |velocity| <= max_velocity.
        velocity *= friction
        velocity += (dv_g + dv_l)
        chk = (np.abs(velocity) > max_velocity)
        velocity[chk] = np.sign(velocity[chk]) * max_velocity

        # Update the local and global bests.
        position += velocity
        conmatrix = constraints(position)
        to_update = (fitness(position) > fitness(lbest))
        to_update &= (conmatrix.sum(axis=1) < thresh)

        if to_update.any():
            lbest[to_update] = position[to_update]
            gbest = lbest[np.argmax(fitness(lbest))]

        # Termination criteria.
        if np.abs(gbest_fit - fitness(gbest)) < thresh:
            stable_count += 1
            if stable_count == stable_iter:
                break
        else:
            stable_count = 0

    if dumpfile is not None:
        with open(dumpfile, 'w') as dfptr:
            for line in dumpdata:
                dfptr.write(line)

    if stable_count != stable_iter:
        raise SwarmConvergeError(
            'no convergence. stable_count=' + str(stable_count))

    return (gbest, ii)