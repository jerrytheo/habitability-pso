import numpy as np
from numpy.random import uniform


def initialize_points(npoints, constraint):
    """Initialize the points from where the Particle Swarm Optimization
    begins converging for CDHS.

    Arguments:
        npoints: int
            Number of points to initialize.
        constraint: 'crs' or 'drs'
            Constraint to satisfy.
    Returns:
        numpy.ndarray of dim (npoints, 2) that satisfy the constraint.
    """
    if constraint == 'crs':
        xvals = uniform(0, 1, (npoints, 1))
        condn = (xvals == 0)
        while condn.any():
            xvals[condn] = uniform(0, 1, xvals[condn].shape)
            condn = (xvals == 0)
        points = np.hstack((xvals, 1-xvals))

    elif constraint == 'drs':
        points = uniform(0, 1, (npoints, 2))
        condn = (points.sum(axis=1) >= 1)
        while condn.any():
            points[condn] = uniform(0, 1, points[condn].shape)
            condn = (points.sum(axis=1) >= 1)

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
        return (exo_param1 ** points.T[0]) * (exo_param1 ** points.T[1])
    return cdhpf


def get_constraint_fn(constraint, err=1e-6, thr=1e-7):
    """Construct the constraint matrix for CDHS.

    Arguments:
        constraint: 'crs' or 'drs'
            Constraint to satisfy.
        err: float, default 1e-6
            Acceptable error in converting strict inequality to non-strict.
        thr: float, default 1e-7
            Threshold in converting equality constraint to inequality.
    Returns:
        function check_constraints(points) -> constraint matrix
            points -- darray, each row is a point of size 2.
    """
    if constraint == 'crs':

        def check_constraints(points):
            """Return the CRS constraint matrix for the points."""
            return np.apply_along_axis(lambda x: np.array((
                    np.max((err - x[0], 0)), np.max((err + x[0] - 1, 0)),
                    np.max((err - x[1], 0)), np.max((err + x[1] - 1, 0)),
                    np.max((x[0] + x[1] - thr - 1, 0)),
                    np.max((1 - thr - x[0] - x[1], 0)),
                )), axis=1, arr=points)

    elif constraint == 'drs':

        def check_constraints(points):
            """Return the DRS constraint matrix for the points."""
            return np.apply_along_axis(lambda x: np.array((
                    np.max((err - x[0], 0)), np.max((err + x[0] - 1, 0)),
                    np.max((err - x[1], 0)), np.max((err + x[1] - 1, 0)),
                    np.max((err + x[0] + x[1] - 1, 0)),
                )), axis=1, arr=points)

    else:
        raise ValueError('invalid constraint: ' + constraint)

    return check_constraints
