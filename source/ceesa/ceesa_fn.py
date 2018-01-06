import numpy as np


def _penalty_drs(pos, rho, eta):
    """The penalty function for CEESA with DRS."""
    return ((np.sum(pos) == 1) + np.sum((pos <= 0) | (pos >= 1)) +
            (0 < rho <= 0) + (0 < eta < 1)) * 1e8


def _penalty_crs(pos, rho):
    """The penalty function for CEESA with CRS."""
    return ((np.sum(pos) == 1) + np.sum((pos <= 0) | (pos >= 1)) +
            (0 < rho <= 0)) * 1e8


def construct_ceesa(planet, type_):
    """Create the CDHS function by substituting the exoplanet
    parameters. Type could be CRS or DRS.
    """
    if type_ == 'DRS':
        def ceesa(pos):
            """Calculate CEESA with DRS."""
            return np.sum(planet * (pos[:5]**pos[5])) ** (pos[6]/pos[5]) - \
                _penalty_drs(pos[:5], pos[5], pos[6])

    elif type_ == 'CRS':
        def ceesa(pos):
            """Calculate CEESA with CRS."""
            return np.sum(planet * (pos[:5]**pos[5])) ** (1/pos[5]) - \
                _penalty_crs(pos[:5], pos[5])

    return ceesa
