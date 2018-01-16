#!/usr/bin/python

# import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from os import path
import pandas as pd


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
for sc in scores:
    for cn, res in scores[sc].items():
        dists = plt.hist(res[sc], bins=60, range=(0, 6),
                         alpha=.65, edgecolor='k')
        plt.title('Distribution of {0} values under {1}'.format(sc, cn))
        plt.xlabel('{0} values'.format(sc))
        plt.ylabel('Number of occurences')
        plt.savefig(path.join(
                'plots', '{0}_{1}.png'.format(sc.lower(), cn.lower())),
                 dpi=1000)
        plt.show()
