import matplotlib.pyplot as plt

import numpy as np

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.figsize'] = (3.6, 2.7)
plt.rcParams['figure.autolayout'] = True


def adjust_plotting(ax, xticklabels, **kwargs):
    if "xlim" in kwargs:
        ax.set_xlim(kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    if "xticks" in kwargs:
        ax.set_xticks(kwargs["xticks"])
    else:
        ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=kwargs["rotation"])

    if "legend" in kwargs and kwargs["legend"] == False:
        pass
    else:
        if "location" in kwargs:
            ax.legend(frameon=False, loc=kwargs["location"])
        else:
            ax.legend(frameon=False)
    ax.set_xlabel(kwargs["xlabel"])
    ax.set_ylabel(kwargs["ylabel"])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    return ax
