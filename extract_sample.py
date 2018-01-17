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
    tabline = '  {} & {} & ${:.3f}$ & ${:.3f}$ & ${:.2f}$ & ${:2}$ & ${:.3f}$'\
              ' & ${:.3f}$ & ${:.2f}$ & ${:2}$ & ${:.2f}$\\\\'
    with open('docs/report/tabs/cdhs%s.tex' % (c,), 'w') as f:
        def fprint(*args, **kwargs): print(*args, **kwargs, file=f)
        fprint('\\begin{tabular}{l r r r r r r r r r r}')
        fprint('  \\toprule')
        fprint('  Name & Class & $\\alpha$ & $\\beta$ & $Y_i$ & $i_i$ & '
               '$\\gamma$ & $\\delta$ & $Y_s$ & $i_s$ & $\mathit{CDHS}$\\\\')
        fprint('  \\midrule')
        for _, row in d[d.Name.isin(planets)].iterrows():
            habc = row['Cls'][:3]
            vals = np.round(np.array([*(row[2:])]), 4)
            if c == 'drs':
                vals[0] /= 1.001
                vals[1] /= 1.001
                vals[3] /= 1.001
                vals[4] /= 1.001
            fprint(tabline.format(row['Name'], habc, *vals[:3], int(vals[7]),
                                  *vals[3:6], int(vals[8]), vals[6]))
        fprint('  \\bottomrule\\\\')
        fprint('\\end{tabular}')
    print('Extracted for CDHS-%s.' % (c.upper(),))


for c, d in ceesa.items():
    tabline = '{} & {} & ${:.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:.3f}$'\
              ' & ${:.3f}$ & ${:.3f}$ & ${:.2f}$ & ${:3}$\\\\'
    with open('docs/report/tabs/ceesa%s.tex' % (c.lower(),), 'w') as f:
        def fprint(*args, **kwargs): print(*args, **kwargs, file=f)
        fprint('\\begin{tabular}{l r r r r r r r r r r}')
        fprint('  \\toprule')
        fprint('  Name & Class & $r$ & $d$ & $t$ & $v$ & $e$ & $\\rho$ & '
               '$\\eta$ & $\mathit{CDHS}$ & $i$\\\\')
        fprint('  \\midrule')
        for _, row in d[d.Name.isin(planets)].iterrows():
            habc = row['Cls'][:3]
            vals = np.round(np.array([*(row[2:])]), 4)
            fprint(tabline.format(row['Name'], habc, *vals[:8], int(vals[8])))
        fprint('  \\bottomrule\\\\')
        fprint('\\end{tabular}\n\n')
    print('Extracted for CEESA-%s.' % (c.upper(),))
