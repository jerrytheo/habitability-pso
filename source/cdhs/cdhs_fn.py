import numpy as np
from ..utils import _uniform

ERR = 1e-6


def initialize_cdhpf(npoints, constraint):
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


def construct_cdhpf(exo_param1, exo_param2, constraint):
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


def penalize_cdhpf(points, constraint):
    """Evaluate the penalty for each point if it violates constraints.
    Arguments:
        points: ndarray of shape (N, 2).
            The points to evaluate the penalty for.
        constraint:
            The constraint to satisfy.
    Returns:
        array of size N.
    """
    # Common constraints are:
    # 1. x[0] < 0               2. x[0] > 1
    # 3. x[1] < 0               4. x[1] > 1
    if constraint == 'crs':
        # Constraints are:
        # 5. x[0]+x[1] < 1          6. x[0]+x[1] > 1
        q = np.apply_along_axis(lambda x: np.array((
                    max(ERR - x[0], 0), max(ERR + x[0] - 1, 0),
                    max(ERR - x[1], 0), max(ERR + x[1] - 1, 0),
                    max(x[0] + x[1] - ERR - 1, 0),
                    max(1 - ERR - x[0] - x[1], 0)
                )), axis=0, arr=points)
    elif constraint == 'drs':
        # Constraints are:
        # 5. x[0]+x[1] < 1
        q = np.apply_along_axis(lambda x: np.array((
                    max(ERR - x[0], 0), max(ERR + x[0] - 1, 0),
                    max(ERR - x[1], 0), max(ERR + x[1] - 1, 0),
                    max(ERR + x[0] + x[1] - 1, 0)
                )), axis=0, arr=points)
    else:
        raise ValueError('invalid constraint: ' + constraint)

    theta_condns = [q < 1e-4, q < .001, q < .01, q < .1, q >= .1]
    theta_assign = [1e4, 5e4, 1e5, 5e5, 1e6]
    theta = np.piecewise(q, theta_condns, theta_assign)
    gamma = np.piecewise(q, [q < 1, q >= 1], [.5, 2])
    return (theta * (q ** gamma)).sum(axis=1)
