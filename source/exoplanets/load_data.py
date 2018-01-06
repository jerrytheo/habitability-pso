import os
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
    'P. Eccentricity',
    'P. Habitable Class',
]]

exoplanets.rename(columns={
    'P. Name': 'Name',
    'P. Radius (EU)': 'Radius',
    'P. Density (EU)': 'Density',
    'P. Ts Mean (K)': 'STemp',
    'P. Esc Vel (EU)': 'Escape',
    'P. Eccentricity': 'Eccentricity',
    'P. Habitable Class': 'Habitable',
}, inplace=True)
exoplanets['STemp'] /= 288
