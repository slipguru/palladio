# -*- coding: UTF-8 -*-
"""Plotting functions for PALLADIO."""
import os
import pandas as pd
import numpy as np
import matplotlib
import warnings

from itertools import combinations, product
from scipy import stats

matplotlib.use('Agg')  # create plots from remote
matplotlib.rcParams['pdf.fonttype'] = 42  # avoid bitmapped fonts in pdf
matplotlib.rcParams['ps.fonttype'] = 42

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns


# Dictionary of nice colors
colorsHex = {
    "Aluminium6": "#2e3436",
    "Aluminium5": "#555753",
    "Aluminium4": "#888a85",
    "Aluminium3": "#babdb6",
    "Aluminium2": "#d3d7cf",
    "Aluminium1": "#eeeeec",
    "lightPurple": "#ad7fa8",
    "mediumPurple": "#75507b",
    "darkPurple": "#5c3566",
    "lightBlue": "#729fcf",
    "mediumBlue": "#3465a4",
    "darkBlue": "#204a87",
    "lightGreen": "#8ae234",
    "mediumGreen": "#73d216",
    "darkGreen": "#4e9a06",
    "lightChocolate": "#e9b96e",
    "mediumChocolate": "#c17d11",
    "darkChocolate": "#8f5902",
    "lightRed": "#ef2929",
    "mediumRed": "#cc0000",
    "darkRed": "#a40000",
    "lightOrange": "#fcaf3e",
    "mediumOrange": "#f57900",
    "darkOrange": "#ce5c00",
    "lightButter": "#fce94f",
    "mediumButter": "#edd400",
    "darkButter": "#c4a000"
}


def distributions(v_regular, v_permutation, base_folder='', metric='nd',
                  first_run=False, is_regression=False):
    """Create a plot of the distributions of performance metrics.

    Plots are created for both "regular" experiments and permutation tests.

    Parameters
    ----------
    v_{regular, permutation}, : numpy.array
        The metric values for all {regular, permutation} experiments.

    base_folder : str, optional, default ''
        The folder where the plot and summary will be saved.

    metric : str, optional, default 'nd'
        Metric used to evaluate v_regular and v_permutation.
        Usually this should be one of ['Accuracy', 'Balanced Accuracy',
        'MCC', 'Precision', 'Recall', 'F1'].
        It is going to be used to set the title and the xlabel.

    first_run : bool, optional, default False
        If not first_run, append the logs to a single file. Otherwise append
        logs to a cleared file.

    is_regression : bool, optional, default False
        If True and plot_errors is True, do errors = -scores instead of
        1 - scores.
    """
    if np.any(np.equal(v_regular, None)) or \
            np.any(np.equal(v_permutation, None)):
        warnings.warn(
            "Cannot create {} plot due to some nan values".format(metric))
        return
    # scaling factor for percentage plot

    if is_regression:
        scale = 1
        x_min = np.min(v_regular)
        x_max = np.max(v_regular)
        _bins = 20
        kde = True
        color_regular=colorsHex['lightBlue']
    else:
        kde = False
        color_regular=colorsHex['lightBlue']
        color_permutation=colorsHex['lightRed']
        if metric.lower() not in ['mcc']: # TODO fix this old notation here
            scale = 100
            _bins = np.arange(0, 105, 5)
            x_min = 0.0
        else:
            scale = 1
            _bins = np.arange(-1, 1, 0.05)
            x_min = -1.0
        x_max = 1.0

    plt.close('all')
    fig, ax = plt.subplots(figsize=(18, 10))
    kwargs = {
        'norm_hist': False,
        'kde': kde,
        'bins': _bins,
        # hist_kws: {'alpha': 0.8}
        'kde_kws': {'color': colorsHex['darkBlue']}
    }

    v_regular = np.array(v_regular)
    v_permutation = np.array(v_permutation)

    reg_mean = np.nanmean(v_regular)
    reg_std = np.nanstd(v_regular)

    perm_mean = np.nanmean(v_permutation)
    perm_std = np.nanstd(v_permutation)

    if len(v_permutation) > 0:
        sns.distplot(v_permutation[~np.isnan(v_permutation)] * scale,
                     # label="Permutation batch \nMean = {0:.2f}, STD = {1:.2f}"
                     # .format(perm_mean, perm_std),
                     label="Permutation batch \nMean = {0:2.1f}, SD = {1:2.1f}"
                           .format(perm_mean, perm_std),
                     color=color_permutation, ax=ax,
                     hist_kws={'alpha': 0.8}, **kwargs)

    sns.distplot(v_regular[~np.isnan(v_regular)] * scale,
                 # label="Regular batch \nMean = {0:.2f}, STD = {1:.2f}"
                 #        .format(reg_mean, reg_std),
                 label="Regular batch \nMean = {0:2.1f}, SD = {1:2.1f}"
                       .format(reg_mean, reg_std),
                 color=color_regular, ax=ax,
                 hist_kws={'alpha': 0.8}, **kwargs)

    # Fit a gaussian with permutation data (DEPRECATED)
    # (mu, sigma) = stats.norm.fit(v_permutation*100)
    # kstest = stats.kstest(v_regular*100, 'norm', args=(mu, sigma))

    # Wilcoxon rank sum test
    # rstest = stats.ranksums(v_regular, v_permutation)
    if len(v_permutation) > 0:
        # rstest = stats.wilcoxon(v_regular, v_permutation)
        rstest = stats.ranksums(v_regular, v_permutation)
        print("[{}] Wilcoxon Signed-rank test: {}".format(metric, rstest))
        # print("Kolmogorov-Smirnov test: {}".format(kstest))
        # print("[{}] Wilcoxon Rank-Sum test: {}".format(metric, rstest))
    else:
        rstest = None

    ## XXX remove text file creation from here
    filemode = 'w' if first_run else 'a'
    with open(os.path.join(base_folder, 'stats.txt'), filemode) as f:
        # f.write("Kolmogorov-Smirnov test p-value: {0:.3e}\n".format(kstest[1]))
        # f.write("Testing distributions")
        f.write("\n------------------------------------------\n")
        f.write("Metric : {}\n".format(metric))
        # f.write("Wilcoxon Rank-Sum test p-value: {0:.3e}\n".format(rstest[1]))
        if len(v_permutation) > 0:
            f.write("Wilcoxon Signed-rank test p-value: {0:.3e}\n".format(rstest[1]))
            f.write("\n")

        f.write("Regular batch, {}\n".format(metric))
        f.write("Mean = {0:.2f}, SD = {1:.2f}\n".format(reg_mean, reg_std))
        f.write("Permutation batch, {}\n".format(metric))
        f.write("Mean = {0:.2f}, SD = {1:.2f}\n".format(perm_mean, perm_std))

    if metric.lower() not in ['mcc']:
        plt.xlabel("{}".format(metric), fontsize="large")
    else:
        plt.xlabel("{}".format(metric), fontsize="large")
    plt.ylabel("Absolute Frequency", fontsize="large")

    plt.title("Distribution of {}".format(metric), fontsize=20)

    # ## Determine limits for the x axis
    # x_min = v_permutation.min() - v_permutation.mean()/10
    # x_max = v_regular.max() + v_regular.mean()/10
    # see above

    plt.xlim([x_min * scale, x_max * scale])

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper right',
               ncol=2, mode="expand", borderaxespad=0., fontsize="large")

    if len(v_permutation) > 0:
        # fig.text(0.1, 0.01, "Kolmogorov-Smirnov test p-value: {0:.3e}\n"
        # .format(kstest[1]), fontsize=18)
        fig.text(0.1, 0.005, "Wilcoxon Signed-Rank test p-value: {0:.3e}\n"
                             .format(rstest[1]), fontsize=18)
        # fig.text(0.1, 0.005, "Wilcoxon Rank-Sum test p-value: {0:.3e}\n"
        #                      .format(rstest[1]), fontsize=18)

    plt.savefig(os.path.join(base_folder, 'permutation_{}_distribution.pdf'
                                          .format(metric)))


def features_manhattan(feat_arr_r, feat_arr_p, base_folder, threshold=.75):
    """Plot selected features with a manhattan plot.

    Parameters
    ----------
    feat_arr_{r, p} : array-like, 2-dimensional
        Array of feature names (first column) and frequencies (second col)
        for regular and permutation batches.
        They are like:
        >>> array([['f1', 0.5],
                   ['f2', 0.8],
                   ['f3', 0.1]], dtype=object)
    base_folder : string
        Path to the folder where to save results.
    threshold : float, optional, default 0.75
        Selection threshold of the features.
    """
    # sort by frequencies of regular
    idx = feat_arr_r[:, 1].argsort()[::-1]
    s_feat_arr_r = feat_arr_r[idx]
    s_feat_arr_p = feat_arr_p[idx]

    plt.close('all')
    plt.figure()

    xaxis = np.arange(feat_arr_r.shape[0])
    s_t = plt.scatter(xaxis, s_feat_arr_r[:, 1], marker='h', alpha=0.8, s=10,
                      color=colorsHex['lightBlue'])
    s_p = plt.scatter(xaxis, s_feat_arr_p[:, 1], marker='h', alpha=0.8, s=10,
                      color=colorsHex['lightRed'])
    threshold_line = plt.axhline(y=threshold, ls='--', lw=0.5, color='k')

    plt.xlim([-5, feat_arr_r.shape[0] + 5])
    plt.ylim([0, 1.05])

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off'  # labels along the bottom edge are off
    )

    plt.legend(
        (s_t, s_p, threshold_line),
        # ('Real signature', 'Permutation signature', 'Threshold'),
        ('Regular batch', 'Permutation batch', 'Threshold'),
        scatterpoints=1,
        loc='upper right',
        ncol=1,
        fontsize=8
    )

    plt.xlabel('Features')
    plt.ylabel('Relative frequencies')
    plt.title("Feature frequencies")
    plt.savefig(os.path.join(base_folder, 'manhattan_plot.pdf'),
                bbox_inches='tight', dpi=300)


def feature_frequencies(feat_arr, base_folder, threshold=.75):
    """Plot a bar chart of the first 2 x M features in a signature.

    M is the number of features whose frequencies is over a given threshold.

    Parameters
    ----------
    feat_arr : array-like, 2-dimensional:
        Array of feature names (first column) and frequencies (second col).
        It is like:
        >>> array([['f1', 0.5],
                   ['f2', 0.8],
                   ['f3', 0.1]], dtype=object)
    base_folder : string
        Path to the folder where to save results.
    threshold : float, optional, default 0.75
        Selection threshold of the features.
    """
    n_over_threshold = feat_arr[feat_arr[:, 1] >= threshold].shape[0]

    # sort by frequencies
    sorted_feat_arr = feat_arr[feat_arr[:, 1].argsort()[::-1]]

    plt.close('all')
    sns.set_context('notebook')
    # plt.figure(figsize=(18, 100))
    ax = sns.barplot(
        x=sorted_feat_arr[:2 * n_over_threshold, 0],
        y=sorted_feat_arr[:2 * n_over_threshold, 1],
        color=colorsHex['lightBlue'], alpha=0.9)

    # plot a horizontal line at the height of the selected threshold
    plt.axhline(y=threshold, ls='--', lw=0.5, color='k', label='Threshold')
    # plt.legend((threshold_line, ), ('Threshold',), scatterpoints=1,
    #            loc='upper right', ncol=1, fontsize=12)
    plt.legend(loc='upper right')

    # vertical line to separate selected and not selected features
    mid = np.sum(ax.get_xbound()) / 2.
    plt.axvline(x=mid, ls='-', lw=1, color=colorsHex['lightRed'])

    plt.title("Manhattan plot - top features detail", fontsize=20)
    plt.xlabel("Feature names", fontsize="large")
    plt.ylabel("Relative frequency", fontsize="large")
    plt.ylim([0, 1.05])

    plt.setp(ax.get_xticklabels(), fontsize=2, rotation='vertical')
    plt.savefig(os.path.join(base_folder, 'signature_frequencies.pdf'),
                bbox_inches='tight', dpi=300)


def select_over_threshold(feat_arr_r, feat_arr_p, base_folder, threshold=.75):
    """Plot the selection trend against the selection frequency threshold.

    Parameters
    ----------
    feat_arr_{r, p} : array-like, 2-dimensional
        Array of feature names (first column) and frequencies (second col)
        for regular and permutation batches.
        They are like:
        >>> array([['f1', 0.5],
                   ['f2', 0.8],
                   ['f3', 0.1]], dtype=object)
    base_folder : string
        Path to the folder where to save results.
    threshold : float, optional, default 0.75
        Selection threshold of the features.
    """
    # horizontal axis REVERSED
    thresh_axis = np.linspace(0, 1, 21)[::-1]

    # iterate over the horiz axis (i.e. the selection freq thresh)
    sel_true = np.zeros(thresh_axis.size)
    sel_perm = np.zeros(thresh_axis.size)
    for i, thr in enumerate(thresh_axis):
        sel_true[i] = np.count_nonzero(feat_arr_r[:, 1] >= thr)
        sel_perm[i] = np.count_nonzero(feat_arr_p[:, 1] >= thr)

    plt.close('all')
    plt.figure()
    plt.plot(100 * thresh_axis, sel_true, marker='h', alpha=0.8,
             color=colorsHex['lightBlue'], label='Regular batch')
    plt.plot(100 * thresh_axis, sel_perm, marker='h', alpha=0.8,
             color=colorsHex['lightRed'], label='Permutation batch')
    plt.axvline(x=threshold * 100, ymin=0, ymax=feat_arr_r.shape[0], ls='--',
                lw=0.5, color='k', label='Threshold')
    plt.legend()
    plt.xlabel("Selection frequency %", fontsize="large")
    plt.ylabel("Number of selected features", fontsize="large")
    plt.xlim(thresh_axis[-1] * 100, thresh_axis[0] * 100)

    plt.savefig(os.path.join(base_folder, 'selected_over_threshold.pdf'))


def score_surfaces(param_grid, results, indep_var=None, pivoting_var=None,
                   base_folder=None, logspace=None, plot_errors=False,
                   is_regression=False):
    """Plot error surfaces.

    Parameters
    ----------
    param_grid : dict
        Dictionary of grid parameters for GridSearch.
    results : dict
        Instance of an equivalent of cv_results_, as given by ModelAssessment.
    indep_var : array-like, optional, default None
        List of independent variables on which plots are based. If more that 2,
        a plot for each combination is made. If None, the 2 longest parameters
        in param_grid are selected.
    pivoting_var : array-like, optional, default None
        List of pivoting variables. For each of them, a plot is made.
        If unspecified, get the unspecified independent variable with the best
        model values.
    base_folder : str or None, optional, default None
        Folder where to save the plots.
    logspace : array-like or None, optional, default None
        List to specify which variable to visualise in logspace.
    plot_errors : bool, optional, default False
        If True, plot errors instead of scores.
    is_regression : bool, optional, default False
        If True and plot_errors is True, do errors = -scores instead of
        1 - scores.
    """
    def multicond(*args):
        cond = args[0]
        for arg in args[1:]:
            cond = np.logical_and(cond, arg)
        return cond

    if indep_var is not None:
        comb = combinations(
            zip(indep_var, [param_grid[x] for x in indep_var]), 2)
    else:
        comb = [sorted(list(
            param_grid.iteritems()), key=lambda item:len(item[1]))[-2:]]
        if len(comb[0]) == 1:
            warnings.warn("Only one grid parameter, cannot create 3D plot")
            return
        indep_var = [comb[0][0][0], comb[0][1][0]]

    if pivoting_var is None:
        pivoting_var = list(set(param_grid.keys()).difference(set(indep_var)))

        ordered_df = pd.DataFrame(pd.DataFrame(results).sort_values(
            'test_score', ascending=False).iloc[0]['cv_results_']).sort_values(
                'mean_test_score', ascending=False).iloc[0]

        # use best model, one pivot
        pivots = [tuple(ordered_df['param_' + x] for x in pivoting_var)]
    else:
        pivots = list(product(*[param_grid[x] for x in pivoting_var]))

    pivot_names = list(product(*[[x] for x in pivoting_var])) * len(pivots)

    plt.close('all')
    for id_pivot, (pivot, value) in enumerate(zip(pivot_names, pivots)):
        for id_param, (param1, param2) in enumerate(comb):
            param_names = 'param_' + np.array(
                [param1[0], param2[0]], dtype=object)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            legend_handles = []
            legend_labels = ['Train Score', 'Validation Score']
            dff = pd.DataFrame(results['cv_results_'])

            # get parameter grid from the first row, since they should be equal
            # but pivoting
            if len(pivot) == 0:
                # no pivoting
                cond = np.ones(
                    dff.iloc[0][param_names[0]].data.size, dtype=bool)
            else:
                cond = multicond(*[dff.iloc[0]['param_' + p].data == v
                                   for p, v in zip(pivot, value)])
            xx = dff.iloc[0][param_names[0]][cond].data.astype(float)
            yy = dff.iloc[0][param_names[1]][cond].data.astype(float)
            log10_x, log10_y = '', ''
            if logspace is not None:
                if param1[0] in logspace:
                    xx = np.log10(xx)
                    log10_x = r'$log_{10}$ '
                if param2[0] in logspace:
                    yy = np.log10(yy)
                    log10_y = r'$log_{10}$ '

            param_grid_xx_size = len(param2[1])
            param_grid_yy_size = len(param1[1])
            X = xx.reshape(param_grid_xx_size, param_grid_yy_size)
            Y = yy.reshape(param_grid_xx_size, param_grid_yy_size)
            # XX, YY = np.meshgrid(np.array(param2[1]), np.array(param1[1]))
            if plot_errors:
                colors = (cm.Oranges_r, cm.Blues_r)
            else:
                colors = (cm.Oranges, cm.Blues)
            for s, h, c in zip(
                    ('train', 'test'),
                    (colorsHex['lightOrange'], colorsHex['lightBlue']),
                    colors):

                # The score is the mean of each external split
                zz = np.mean(np.vstack(
                    dff['mean_%s_score' % s].tolist()), axis=0)[cond]
                Z = zz.reshape(param_grid_xx_size, param_grid_yy_size)
                if plot_errors:
                    Z = -Z if is_regression else 1 - Z

                # plt.close('all')
                ax.plot_surface(
                    X, Y, Z, cmap=c, rstride=1, cstride=1, lw=0,
                    antialiased=False)

                legend_handles.append(Rectangle((0, 0), 1, 1, fc=h))

            # plot max
            func_max = np.min if plot_errors else np.max
            pos_max = np.where(Z == func_max(Z))
            ax.plot(X[pos_max], Y[pos_max], Z[pos_max], 'o',
                    c=colorsHex['darkRed'])

            # fig.colorbar()
            ax.legend(legend_handles, legend_labels[:len(legend_handles)],
                      loc='best')
            scoring = 'error' if plot_errors else 'score'
            ax.set_title('average KCV %s, pivot %s = %s' % (
                scoring, pivot, value))
            ax.set_xlabel(log10_x + param_names[0][6:])
            ax.set_ylabel(log10_y + param_names[1][6:])
            ax.set_zlabel("avg kcv %s" % scoring)

            if base_folder is not None:
                plt.savefig(os.path.join(
                    base_folder, 'kcv_%s_piv%d_comb%d.pdf' % (
                        scoring, id_pivot, id_param)))
            else:
                plt.show()
