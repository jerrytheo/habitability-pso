#!/usr/bin/python

# import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys


help_text = """
USAGE: ./generate_plots.py [-h] [--help] [-n] [--nodisplay] [-s] [--save]
                           [--type <plotname>]
Generate the CDHS and CEESA plots for estimated values in results.

OPTIONAL ARGUMENTS:
    -h --help
        Display this help message.
    -n --nodisplay
        Do not display plots.
    -s --save
        Save plots to file.
    --type <plottype>
        Construct plots only for the specified type. <plottype> could be either
        'dist' or 'iter'.
"""
invalid = 'Invalid usage.\n' + help_text


scores = {
    'CDHS': {
        'CRS': pd.read_csv('results/cdhs_crs.csv'),
        'DRS': pd.read_csv('results/cdhs_drs.csv'),
    },
    'CEESA': {
        'CRS': pd.read_csv('results/ceesa_crs.csv'),
        'DRS': pd.read_csv('results/ceesa_drs.csv'),
    }
}


# Score distributions.
def plot_score_distributions(scores, display=True, save=False):
    for sc in scores:
        for cn, res in scores[sc].items():
            print('Plotting %s-%s distribution.' % (sc, cn))
            plt.hist(res[sc], bins=60, range=(0, 6), edgecolor='k')
            plt.title('Distribution of %s under %s\n' % (sc, cn))
            plt.xlabel('{0} values'.format(sc))
            plt.ylabel('Number of occurences')
            if save:
                sl, cn = sc.lower(), cn.lower()
                plt.savefig('plots/' + sl + '_' + cn + '.png', dpi=1000)
            plt.show() if display else plt.close()


# Iterations distributions.
def plot_iter_distributions(scores, display=True, save=False):
    params = dict(bins=50, histtype='bar', edgecolor='k')
    title = 'Distribution of Iterations to Maxima for {} under {}\n'
    for sc in scores:
        for cn, res in scores[sc].items():
            print('Plotting %s-%s iterations.' % (sc, cn))
            if sc == 'CDHS':
                #  plt.hist(np.array((res['Inn'], res['Sur'])).T, **params)
                plt.hist(res['Inn'], label='inner', range=(0, 100), **params)
                plt.hist(res['Sur'], label='surface', range=(0, 100),
                         alpha=.75, **params)
                plt.legend()
            elif sc == 'CEESA':
                plt.hist(res['Iter'], range=(70, 120), **params)
            plt.title(title.format(sc, cn))
            plt.xlabel('Iterations')
            plt.ylabel('Number of occurences')
            if save:
                sl, cn = sc.lower(), cn.lower()
                plt.savefig('plots/Iter_' + sl + '_' + cn + '.png', dpi=1000)
            plt.show() if display else plt.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    disp = False
    save = False
    plots = []
    try:
        while args:
            arg = args.pop(0)
            if arg in ['-d', '--display']:
                disp = True
            elif arg in ['-s', '--save']:
                save = True
            elif arg in ['-h', '--help']:
                print(help_text)
                sys.exit(0)
            elif arg == '--type':
                ptype = args.pop(0)
                if ptype == 'dist':
                    plots.append(plot_score_distributions)
                elif ptype == 'iter':
                    plots.append(plot_iter_distributions)
            else:
                print(invalid)
                sys.exit(-1)
    except IndexError:
        print(invalid)
        sys.exit(-1)

    if not plots:
        plots = [plot_score_distributions, plot_iter_distributions]

    for p in plots:
        p(scores, disp, save)
