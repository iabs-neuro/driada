import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def make_beautiful(ax):
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=8, pad=15)

    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.0)

    # ax.locator_params(axis='x', nbins=8)
    # ax.locator_params(axis='y', nbins=8)
    ax.tick_params(axis='x', which='major', labelsize=26)
    ax.tick_params(axis='y', which='major', labelsize=26)

    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)

    params = {'legend.fontsize': 18,
              'axes.titlesize': 30,
              }

    pylab.rcParams.update(params)

    return ax


def create_default_figure(a=16, b=12):
    fig, ax = plt.subplots(figsize=(a, b))
    ax = make_beautiful(ax)

    return fig, ax