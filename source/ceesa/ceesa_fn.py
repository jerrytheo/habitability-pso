import numpy as np


def _penalty_drs(pos, rho, eta):
    """The penalty function for CEESA with DRS."""
    return ((np.sum(pos) != 1) +                # sum != 1
            np.sum((pos < 0) | (pos >= 1)) +    # 0 < p_i <= 1
            (not (0 < rho <= 1)) +              # 0 < rho <= 1
            (not (0 < eta < 1))) * 1e8          # 0 < eta < 1


def _penalty_crs(pos, rho):
    """The penalty function for CEESA with CRS."""
    return ((np.sum(pos) != 1) +                # sum != 1
            np.sum((pos < 0) | (pos >= 1)) +    # 0 < p_i <= 1
            (not (0 < rho <= 1))) * 1e8         # 0 < rho <= 1


def construct_ceesa(planet, constraint):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    if constraint == 'drs':
        def ceesa(pos):
            """Calculate CEESA with DRS."""
            return np.sum(pos[:5] * (planet**pos[5])) ** (pos[6]/pos[5]) - \
                _penalty_drs(pos[:5], pos[5], pos[6])

    elif constraint == 'crs':
        def ceesa(pos):
            """Calculate CEESA with CRS."""
            penalty = _penalty_crs(pos[:5], pos[5])
            temp = np.sum(pos[:5] * (planet**pos[5])) ** (1/pos[5])
            print(pos, '\t\t', penalty)
            return temp - penalty

    else:
        raise ValueError('Do not understand constraint: ' + constraint)

    return ceesa
