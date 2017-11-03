import os
import numpy as np
import pandas as pd

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

exoplanets = pd.read_csv(os.path.join(__location__, 'phl_hec-confirmed.csv'))
exoplanets = exoplanets[[
    'P. Name',
    'P. Radius (EU)',
    'P. Density (EU)',
    'P. Ts Mean (K)',
    'P. Esc Vel (EU)',
    'P. Habitable Class',
]]
exoplanets['P. Esc Vel (EU)'] = np.round(
        exoplanets['P. Esc Vel (EU)'] / 288, 5)
exoplanets.rename(columns={
    'P. Name': 'Name',
    'P. Radius (EU)': 'Radius',
    'P. Density (EU)': 'Density',
    'P. Ts Mean (K)': 'STemp',
    'P. Esc Vel (EU)': 'Escape',
    'P. Habitable Class': 'Habitable',
}, inplace=True)
