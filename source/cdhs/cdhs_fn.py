import numpy as np
from ..utils import _uniform


def initialize_points(npoints, constraint):
    """Initialize the points from where the Particle Swarm Optimization
    begins converging.

    Arguments:
        npoints: int
            Number of points to initialize.
        constraint: 'crs' or 'drs'
            Constraint to satisfy.
    Returns:
        numpy.ndarray of dim (npoints, 2) that satisfy constraint.
    """
    if constraint == 'crs':
        xvals = _uniform(0, 1, (npoints, 1))
        condn = (xvals == 0)
        while condn.any():
            xvals[condn] = _uniform(0, 1, xvals[condn].shape)
            condn = (xvals == 0)
        points = np.hstack((xvals, 1-xvals))

    elif constraint == 'drs':
        xvals = _uniform(0, 1, (npoints, 2))
        condn = (xvals.sum(axis=1) >= 1)
        while condn.any():
            xvals[condn] = _uniform(0, 1, xvals[condn].shape)
            condn = (xvals.sum(axis=1) >= 1)
        points = xvals

    else:
        raise ValueError('invalid constraint: ' + constraint)

    return points


def construct_fitness(exo_param1, exo_param2, constraint):
    """Construct the CDHS function for the given exoplanet parameters.

    Arguments:
        exo_param1, exo_param2: float
            The coefficients for the CDHPF.
        constraint: 'crs' or 'drs'
            Constraint to satisfy.
    Returns:
        function cdhpf(points) -> fitness values for each point.
            points -- ndarray, each row is a point of size 2.
    """
    def cdhpf(points):
        """Return the CDHPF value for each point in the Swarm."""
        return (exo_param1 ** points.T[0]) * (exo_param2 ** points.T[1])
    return cdhpf


def get_constraint_fn(constraint):
    """Construct the constraint matrix for the points.

    Arguments:
        constraint:
            The constraint to satisfy.
    Returns:
        function check_constraints(points) -> constraint matrix
            points -- darray, each row is a point of size 2.
    """
    ERR = 1e-6
    if constraint == 'crs':
        DEL = 1e-5

        def check_constraints(points):
            """Return the crs constraint matrix for the points."""
            return np.apply_along_axis(lambda x: np.array((
                    max(ERR - x[0], 0),                 # x[0] > 0
                    max(ERR + x[0] - 1, 0),             # x[0] < 1
                    max(ERR - x[1], 0),                 # x[1] > 0
                    max(ERR + x[1] - 1, 0),             # x[1] < 1
                    max(x[0] + x[1] - DEL - 1, 0),      # x[0] + x[1] - del < 1
                    max(1 - DEL - x[0] - x[1], 0)       # x[0] + x[1] + del > 1
                )), axis=0, arr=points)

    elif constraint == 'drs':

        def check_constraints(points):
            """Return the drs constraint matrix for the points."""
            return np.apply_along_axis(lambda x: np.array((
                    max(ERR - x[0], 0),                 # x[0] > 0
                    max(ERR + x[0] - 1, 0),             # x[0] < 1
                    max(ERR - x[1], 0),                 # x[1] > 0
                    max(ERR + x[1] - 1, 0),             # x[1] < 1
                    max(ERR + x[0] + x[1] - 1, 0)       # x[0] + x[1] < 1
                )), axis=0, arr=points)

    else:
        raise ValueError('invalid constraint: ' + constraint)

    #  theta_condns = [q < 1e-4, q < .001, q < .01, q < .1, q >= .1]
    #  theta_assign = [1e4, 5e4, 1e5, 5e5, 1e6]
    #  theta = np.piecewise(q, theta_condns, theta_assign)
    #  gamma = np.piecewise(q, [q < 1, q >= 1], [.5, 2])
    #  return (theta * (q ** gamma)).sum(axis=1)
    return check_constraints
