import numpy as np
from ..utils import _uniform

ERR = 1e-6


def evaluate_theta_gamma(q):
    """Returns a 2-tuple (theta, gamma) or numpy arrays of q.size
    elements. theta and gamma are values assigned by the respective
    piecewise functions for q.
    """
    theta_conditions = [q < 1e-4, q < .001, q < .01, q < .1, q >= .1]
    theta_assignments = [1e4, 5e4, 1e5, 5e5, 1e6]
    theta = np.piecewise(q, theta_conditions, theta_assignments)
    gamma = np.piecewise(q, [q < 1, q >= 1], [.5, 2])
    return (theta, gamma)


def _penalty_crs(pos, k):
    """Calculate the penalty for CDHPF under the CRS constraint."""
    q = np.array((
        max(-pos[0] + ERR, 0),                  # pos[0] > 0
        max(-pos[1] + ERR, 0),                  # pos[1] > 0
        max(pos[0] - 1 + ERR, 0),               # pos[0] < 1
        max(pos[1] - 1 + ERR, 0),               # pos[1] < 1
        max(pos[0] + pos[1] - ERR - 1, 0),      # pos[0] + pos[1] - d < 1
        max(1 - ERR - pos[0] - pos[1], 0),      # pos[0] + pos[1] - d < 1
    ), dtype=np.float)

    theta, gamma = evaluate_theta_gamma(q)
    return (k+1) * np.sqrt(k+1) * np.sum(theta * (q**gamma))


def _penalty_drs(pos, k):
    """Calculate the penalty for CDHPF under the DRS constraint."""
    q = np.array((
        max(ERR - pos[0], 0),                       # pos[0] > 0
        max(ERR - pos[1], 0),                       # pos[1] > 0
        max(ERR + pos[0] - 1, 0),                   # pos[0] < 1
        max(ERR + pos[1] - 1, 0),                   # pos[1] < 1
        max(ERR + pos[0] + pos[1] - 1, 0)           # pos[0] + pos[1] < 1
    ), dtype=np.float)

    theta, gamma = evaluate_theta_gamma(q)
    return (k+1) * np.sqrt(k+1) * np.sum(theta * (q**gamma))


def _initialize_crs(npoints):
    """Initialize npoints points for the CRS constraint."""
    xvals = _uniform(0, 1, (npoints, 1))
    condn = (xvals == 0)
    while condn.any():
        xvals[condn] = _uniform(0, 1, xvals[condn].shape)
        condn = (xvals == 0)
    return np.hstack((xvals, 1-xvals))


def _initialize_drs(npoints):
    """Initialize npoints points for the DRS constraint."""
    xvals = _uniform(0, 1, (npoints, 2))
    condn = (xvals.sum(axis=1) >= 1)
    while condn.any():
        xvals[condn] = _uniform(0, 1, xvals[condn].shape)
        condn = (xvals.sum(axis=1) >= 1)
    return xvals


def construct_cdhpf(npoints, coeff1, coeff2, constraint):
    """Initialize points and create the CDHS function by substituting
    the exoplanet parameters. Constraint could be CRS or DRS. Returns a
    2-tuple (points, cdhpf).
    """
    if constraint == 'crs':
        points = _initialize_crs(npoints)
        penalty = _penalty_crs
    elif constraint == 'drs':
        points = _initialize_drs(npoints)
        penalty = _penalty_drs
    else:
        raise ValueError('Do not understand constraint: ' + constraint)

    def cdhpf(pos, k=1):
        return (coeff1 ** pos[0]) * (coeff2 ** pos[1]) - penalty(pos, k)

    return (points, cdhpf)
