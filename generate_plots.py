#!/usr/bin/python

# import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def getdists(const):
    dists = []
    for n in range(50, 150, 10):
        fpath = 'pso_'+const+'_'+str(n)+'.csv'
        cdhs = pd.read_csv('res/multiple/'+fpath)
        dists.append((np.histogram(cdhs['Conv_i'], bins=range(48, 60))[0],
                      np.histogram(cdhs['Conv_s'], bins=range(48, 60))[0]))
    return dists


def plotgraphs(plots, label):
    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)

    fig.text(0.5, 0.95, label, ha='center', size='large')
    fig.text(0.5, 0.04, 'Iterations to Convergence', ha='center', size='small')
    fig.text(0.07, 0.31, 'Count', va='center', rotation='vertical',
             size='small')
    fig.text(0.07, 0.73, 'Count', va='center', rotation='vertical',
             size='small')
    axes = axes.flatten()

    npart = 50
    for i, dists in enumerate(plots):
        axes[i].plot(range(0, 11), dists[0], '-', linewidth=1.0)
        axes[i].plot(range(0, 11), dists[1], '--', linewidth=1.0)
        axes[i].set_title(str(npart)+' Particles', {'fontsize': 'medium'})
        axes[i].tick_params(axis='both', labelsize='small')
        npart += 10

    axes[0].set_xticks(range(0, 11, 2))
    fig.legend(axes[0].get_lines(), ('$CDHS_i$', '$CDHS_s$'))


if __name__ == '__main__':
    crs = getdists('crs')
    plotgraphs(crs, 'Constant Returns to Scale')

    drs = getdists('drs')
    plotgraphs(drs, 'Decreasing Returns to Scale')

    plt.show()
