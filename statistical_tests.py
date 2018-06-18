from itertools import combinations
from math import sqrt, sin, cos, pi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import baycomp as bc
import numpy as np
import pandas as pd
from sklearn.externals.joblib import delayed, Parallel


def plot_simplex(test, ax, names=None):
    """
    Plot the posterior distribution in a simplex.

    The distribution is shown as a triangle with regions corresponding to
    first classifier having higher scores than the other by more than rope,
    the second having higher scores, or the difference being within the
    rope.

    Args:
        names (tuple of str): names of classifiers
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def project(points):
        p1, p2, p3 = points.T / sqrt(3)
        x = (p2 - p1) * cos(pi / 6) + 0.5
        y = p3 - (p1 + p2) * sin(pi / 6) + 1 / (2 * sqrt(3))
        return np.vstack((x, y)).T

    ax.set_aspect('equal', 'box')

    # triangle
    ax.add_line(Line2D([0, 0.5, 1.0, 0],
                       [0, np.sqrt(3) / 2, 0, 0], color='orange'))
    names = names or test.names or ("C1", "C2")
    pl, pe, pr = test.probs()
    ax.text(0, -0.04,
            'p({}) = {:.3f}'.format(names[0], pl),
            horizontalalignment='center', verticalalignment='top')
    ax.text(0.5, np.sqrt(3) / 2,
            'p(rope) = {:.3f}'.format(pe),
            horizontalalignment='center', verticalalignment='bottom')
    ax.text(1, -0.04,
            'p({}) = {:.3f}'.format(names[1], pr),
            horizontalalignment='center', verticalalignment='top')
    cx, cy = project(np.array([[0.3333, 0.3333, 0.3333]]))[0]
    for x, y in project(np.array([[.5, .5, 0], [.5, 0, .5], [0, .5, .5]])):
        ax.add_line(Line2D([cx, x], [cy, y], color='orange'))

    # project and draw points
    tripts = project(test.sample[:, [0, 2, 1]])
    ax.hexbin(tripts[:, 0], tripts[:, 1], mincnt=1, cmap=plt.cm.Blues_r)
    # Leave some padding around the triangle for vertex labels
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')


def get_data(method, aggregate=False):
    results = pd.read_csv("results_nonlinear_geometric_clf_1000bags.csv")

    results = results[results.method == method].sort_values("dataset")
    maes = results["mae"].values.reshape(41, -1)
    if aggregate:
        maes = maes.mean(axis=1)

    inv_maes = 1 - maes
    return inv_maes


from matplotlib.transforms import Bbox


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


prob_df = pd.DataFrame(columns=["Cuantificador1", "Cuantificador2", "p(m1)", "p(rope)", "p(m2)"])

methods = ["EDX", "EDy", "HDX", "HDy", "AC", "CvMy"]

methods_combs = combinations(methods, 2)

fig, axs = plt.subplots(len(methods), len(methods), figsize=(25, 25))
_ = [axs[i][i].axis("off") for i in range(len(methods))]
_ = [axs[i][i].text(0.32, 0.42, methods[i], fontsize=60) for i in range(len(methods))]


# for m1, m2 in methods_combs:

def do_hierarchical_test(m1, m2):
    i, j = methods.index(m1), methods.index(m2)
    try:
        data_hdx = get_data(m1)
        data_hdy = get_data(m2)

        test = bc.HierarchicalTest(data_hdx, data_hdy, 0.01, runs=1)
        plot_simplex(test, axs[i][j], names=(m1, m2))
        plot_simplex(test, axs[j][i], names=(m1, m2))
        extent = full_extent(axs[i][j]).transformed(fig.dpi_scale_trans.inverted())
        fig.tight_layout()
        fig.savefig("figures/Hierarchical20x50_{}_vs_{}.png".format(m1, m2), bbox_inches=extent)
    except RuntimeError as err:
        print("Failed Hierarchical Test between ", m1, m2, err)

    return pd.DataFrame([[m1, m2] + list(test.probs())],
                        columns=["Cuantificador1", "Cuantificador2", "p(m1)", "p(rope)", "p(m2)"])


fig.tight_layout()
fig.savefig("figures/Hierarchical_20x50bags.png")


probs_df = pd.concat(Parallel(n_jobs=1)(
    delayed(do_hierarchical_test)(m1, m2) for m1, m2 in methods_combs))


fig, axs = plt.subplots(len(methods), len(methods), figsize=(25, 25))
_ = [axs[i][i].axis("off") for i in range(len(methods))]
_ = [axs[i][i].text(0.32, 0.42, methods[i], fontsize=60) for i in range(len(methods))]

methods_combs = combinations(methods, 2)

for m1, m2 in methods_combs:
    i, j = methods.index(m1), methods.index(m2)
    try:
        data_hdx = get_data(m1, aggregate=True)
        data_hdy = get_data(m2, aggregate=True)

        test = bc.SignTest(data_hdx, data_hdy, 0.01)
        plot_simplex(test, axs[i][j], names=(m1, m2))
        plot_simplex(test, axs[j][i], names=(m1, m2))
    except RuntimeError as err:
        print("Failed Signed Test between ", m1, m2, err)

    plt.savefig("figures/Signed_20x50bags.png")

fig.tight_layout()

prob_df.to_csv("prob_test_20x50.csv", index=None)
