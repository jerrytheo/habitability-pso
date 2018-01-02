import numpy as np


ERR = .1


def _penalty_crs(pos):
    """Calculate the penalty for CDHPF under the CRS constraint."""
    q = np.array((
        max(-pos[0] + ERR, 0),                  # pos[0] > 0
        max(-pos[1] + ERR, 0),                  # pos[1] > 0
        max(pos[0] - 1 + ERR, 0),               # pos[0] < 1
        max(pos[1] - 1 + ERR, 0),               # pos[1] < 1
        np.abs(pos[0] + pos[1] - 1),            # CRS constraint.
    ), dtype=np.float)

    theta = np.piecewise(q, [q == 0, q > 0], [0, 1e8])
    gamma = np.piecewise(q, [q <= 1, q > 1], [1, np.prod(np.exp(pos))])
    return np.sum(theta*(q**gamma))


def _penalty_drs(pos):
    """Calculate the penalty for CDHPF under the DRS constraint."""
    q = np.array((
        max(-pos[0] + ERR, 0),                    # pos[0] > 0
        max(-pos[1] + ERR, 0),                    # pos[1] > 0
        max(pos[0] - 1 + ERR, 0),                 # pos[0] < 1
        max(pos[1] - 1 + ERR, 0),                 # pos[1] < 1
        max(pos[0] + pos[1] - 1 + ERR, 0)         # DRS constraint.
    ), dtype=np.float)

    theta = np.piecewise(q, [q == 0, q > 0], [0, 1e8])
    gamma = np.piecewise(q, [q <= 1, q > 1], [1, np.prod(np.exp(pos))])
    return np.sum(theta*(q**gamma))


def construct_cdhpf(coeff1, coeff2, constraint):
    """Create the CDHS function by substituting the exoplanet
    parameters. Constraint could be CRS or DRS.
    """
    if constraint == 'crs':
        penalty = _penalty_crs
    elif constraint == 'drs':
        penalty = _penalty_drs
    else:
        raise ValueError('Do not understand constraint: ' + constraint)

    def cdhpf(pos):
        pos = np.round(pos, 1)
        return (coeff1 ** pos[0]) * (coeff2 ** pos[1]) - penalty(pos)

    return cdhpf
