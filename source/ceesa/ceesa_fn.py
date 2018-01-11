import numpy as np
from numpy.random import uniform


def initialize_points(npoints, constraint):
    """Initialize the points from where the Particle Swarm Optimization
    begins converging for CEESA.

    Arguments:
        npoints: int
            Number of points to initialize.
        constraint: 'crs' or 'drs'
            Constraint to satisfy.

    Returns:
        numpy.ndarray of dim (npoints, 6) for 'crs' and (npoints, 7) for 'drs'
        that satisfy the respective constraint.
    """
    if constraint == 'crs':
        ndim = 6
    elif constraint == 'drs':
        ndim = 7
    else:
        raise ValueError('invalid constraint: ' + constraint)

    points = uniform(0, 1, (npoints, ndim))
    condn = (points == 0)
    while condn.any():
        points[condn] = uniform(0, 1, points[condn].shape)
        condn = (points == 0)

    #  Normalize the first 5 columns of each row.
    points.T[:5] /= np.sum(points.T[:5], axis=0)
    return points


def construct_fitness(ep0, ep1, ep2, ep3, ep4, constraint):
    """Construct the CEESA function for the given exoplanet parameters.

    Arguments:
        ep0, ep1, ep2, ep3, ep4: float
            The coefficients for CEESA.
        constraint: 'crs' or 'drs'
            Constraint to satisfy.
    
    Returns:
        function ceesa(points) -> fitness values for each point.
            points -- ndarray, each row is a point of size 6 (crs) or 7 (drs).

    Notes:
        Broadcasting is done as follows,
                ({5,N} * ({5,1} ** {5,1})) => {5,N}
                        sum({5,N}, axis=0) => {N,} ** {N,}
    """
    coeff = np.array((ep0, ep1, ep2, ep3, ep4), ndmin=2).T

    if constraint == 'crs':

        def ceesa(points):
            """Return the CRS-CEESA score for each point in the Swarm."""
            P = points.T
            return (P[:5] * (coeff ** P[5,None])).sum(axis=0) ** (1 / P[5])

    elif constraint == 'drs':

        def ceesa(points):
            """Return the DRS-CEESA score for each point in the Swarm."""
            P = points.T
            return (P[:5] * (coeff ** P[5,None])).sum(axis=0) ** (P[6] / P[5])

    else:
        raise ValueError('invalid constraint: ' + constraint)

    return ceesa


def get_constraint_fn(constraint, err=1e-6, thr=1e-7):
    """Construct the constraint matrix for CEESA for given constraint type.

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

    Note:
        A short explanation of what the resulting function does for each row in
        points,

        If the constraint is 'crs', there are 4 arrays representing 1 set of
        constraints,
             1. x[i] >= 0;  g(x[i]) =      -x[i];   i: 1-5,
             2. x[i] <= 1;  g(x[i]) =   x[i] - 1;   i: 1-5,
            3a. x[5] >  0;  g(x[5]) = err - x[5];   x[5]: rho,
            3b. x[5] <= 1;  g(x[5]) =   x[5] - 1;   x[5]: rho,

            4a. sum(x[i]) <= 1 + del;   g(x[i]) = sum(x[i]) - 1 - del;
            4b. sum(x[i]) >= 1 - del;   g(x[i]) = 1 - del - sum(x[i]);

        If the constraint is 'drs', there is an additional constraint resulting
        in 5 arrays instead,
             1. x[i] >= 0;  g(x[i]) =          -x[i];   i: 1-5,
             2. x[i] <= 1;  g(x[i]) =       x[i] - 1;   i: 1-5,
             3. x[j] >  0;  g(x[j]) =     err - x[5];   x[j]: rho, eta,
            4a. x[5] <= 1;  g(x[5]) =       x[5] - 1;   x[5]: rho,
            4b. x[6] <  1;  g(x[6]) = err + x[6] - 1;   x[6]: eta,
          
            5a. sum(x[i]) <= 1 + del;   g(x[i]) = sum(x[i]) - 1 - del;
            5b. sum(x[i]) >= 1 - del;   g(x[i]) = 1 - del - sum(x[i]);

        These arrays are concatenated and then horizontally stacked with a
        column vector of 0s. The max of every row corresponds to max(gi(x), 0)
        and is used to construct the constraint vector.

    """
    if constraint == 'crs':

        def check_constraints(points):
            """Return the CRS constraint matrix for the points."""
            return np.apply_along_axis(lambda x: np.max(
                np.hstack((np.concatenate((
                    -x[:5], x[:5] - 1,                      # 0 < x[i] < 1
                    np.array((err - x[5], x[5] - 1,)),      # 0 < rho <= 1
                    np.array((np.sum(x[:5]) - 1 - thr,      # sum(x) <= 1 + del
                              1 - thr - np.sum(x[:5]))),    # sum(x) >= 1 - del
                ))[:, None], np.zeros((14, 1))
                )), axis=1), axis=1, arr=points)

    elif constraint == 'drs':

        def check_constraints(points):
            """Return the DRS constraint matrix for the points."""
            return np.apply_along_axis(lambda x: np.max(
                np.hstack((np.concatenate((
                    -x[:5], x[:5] - 1,                      # 0 < x[i] < 1
                    err - x[5:],                            # 0 < rho, 0 < eta
                    np.array((x[5] - 1, x[6] - 1 + err)),   # rho <= 1, eta < 1
                    np.array((np.sum(x[:5]) - 1 - thr,      # sum(x) >= 1 + del
                              1 - thr - np.sum(x[:5]))),    # sum(x) <= 1 - del
                ))[:, None], np.zeros((16, 1))
                )), axis=1), axis=1, arr=points)

    else:
        raise ValueError('invalid constraint: ' + constraint)

    return check_constraints
