import numpy as np
from scipy.spatial import distance
#  import sys

from ..utils import _round, _uniform


# Exception raised when Swarm does not converge.
class SwarmConvergeError(Exception):
    """Error raised when the swarm does not converge."""
    pass


# Function that returns the leaders. Implementation may change to include more
# complex methods.
def get_leaders(position, lbest):
    """Return the leaders for each position from lbest.

    Arguments:
        position: ndarray of shape (N, D)
            Array of the current position of each point. N is the number of
            points and D the dimensions.
        lbest: ndarray of shape (N, D)
            Array of the local best position of each point.
    Returns:
        ndarray of shape (N, D) with the leaders for each corresponding point.
    """
    return np.argmin(distance.cdist(position, lbest), axis=1)


# Function that updates local best. Not all local bests may be updated. Depends
# on the method used for get_leaders.
def update_lbest(lbest, positions, constraints):
    """Update local best to positions based on constraints. lbest is updated in
    place.

    Arguments:
        lbest: ndarray of shape (N, D)
            Array of current local best positions of each point.
        position: ndarray of shape (N, D)
            Array of the current position of each point.
        constraints: ndarray of shape (N, S)
            Constraint matrix. S is the number of constraints.
    Note:
        lbest and position should be such that,
            fitness(position) > fitness(lbest)
    """
    condn = (constraints.sum(axis=1) == 0)
    lbest[condn] = positions[condn]


# Function for convergence.
def conmax_by_pso(fitness, start_points, constraints, friction=.8,
                  learnrate1=.1, learnrate2=.1, max_velocity=1.,
                  max_iter=1000, stable_iter=100, thresh=1e-5):
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
    Returns:
        a 2-tuple (swarm, it), where swarm is the converged particle swarm and
        it are the number of iterations taken to converge.
    """
    # Initial position and velocity.
    position = start_points
    velocity = _uniform(-max_velocity, max_velocity, position.shape)

    # Initial local best for each point and global best.
    lbest = start_points
    gbest = lbest[np.argmax(fitness(lbest))]

    stable_count = 0
    for ii in range(max_iter):
        # Store old for threshold comparison.
        old_g = gbest

        # Determine the global velocity gradient.
        co_g = _round(learnrate1 * _uniform(0, 1))
        dv_g = _round(co_g * (gbest - position))

        # Determine the local velocity gradient.
        leaders = get_leaders(position, lbest)
        co_l = _round(learnrate2 * _uniform(0, 1))
        dv_l = _round(co_l * (lbest[leaders] - position))

        # Update velocity such that |velocity| <= max_velocity.
        velocity = _round(friction * velocity) + (dv_g + dv_l)
        chk = (velocity > max_velocity)
        velocity[chk] = np.sign(velocity[chk]) * max_velocity

        # Update the local and global bests.
        position = _round(position + velocity)
        improved = (fitness(position) > fitness(lbest))
        update_lbest(lbest[improved], position[improved],
                     constraints(position[improved]))
        gbest = lbest[np.argmax(fitness(lbest))]

        # Termination criteria.
        if np.abs(old_g - gbest) < thresh:
            stable_count += 1
            if stable_count == stable_iter:
                return (gbest, ii)
        else:
            stable_count = 0
    else:
        raise SwarmConvergeError('no convergence. stable_count='+stable_count)
