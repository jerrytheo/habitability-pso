#!/usr/bin/python

import numpy as np
import pandas as pd

cdhs = {
        'crs': pd.read_csv('results/cdhs_crs.csv'),
        'drs': pd.read_csv('results/cdhs_drs.csv')
        }

ceesa = {
        'crs': pd.read_csv('results/ceesa_crs.csv'),
        'drs': pd.read_csv('results/ceesa_drs.csv')
        }

planets = [
        'GJ 176 b',
        'GJ 667 C b',
        'GJ 667 C e',
        'GJ 667 C f',
        'GJ 3634 b',
        'HD 20794 c',
        'HD 40307 e',
        'HD 40307 f',
        'HD 40307 g',
        'Kepler-186 f',
        'Proxima Cen b',
        'TRAPPIST-1 b',
        'TRAPPIST-1 c',
        'TRAPPIST-1 d',
        'TRAPPIST-1 e',
        'TRAPPIST-1 g',
        ]

for c, d in cdhs.items():
    print(c.upper() + ':')
    print('\\begin{tabular}{l l r r r r r r r r r}')
    print('  \\toprule\\\\')
    print('  Name & Class & '
          '$\\alpha$ & $\\beta$ & $\mathit{CDHS}_i$ & Iterations(inner) & '
          '$\\gamma$ & $\\delta$ & $\mathit{CDHS}_s$ & Iterations(surface) & '
          '$\mathit{CDHS}$\\\\')
    print('  \\midrule\\\\')
    for _, row in d[d.Name.isin(planets)].iterrows():
        habc = row['Cls'][:3]
        vals = np.round(np.array([*(row[2:])]), 4)
        print('  {} & {} & {:01.4f} & {:01.4f} & {:01.2f} & {:03} & '
              '{:01.4f} & {:01.4f} & {:01.2f} & {:03} & {:01.2f}\\\\'.format(
                    row['Name'], row['Cls'],
                    vals[0], vals[1], vals[2], int(vals[7]),
                    vals[3], vals[4], vals[5], int(vals[8]),
                    vals[6]))
    print('  \\bottomrule')
    print('\\end{tabular}')
