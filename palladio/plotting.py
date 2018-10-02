# -*- coding: UTF-8 -*-
"""Plotting functions for PALLADIO."""
import os
import pandas as pd
import numpy as np
import matplotlib
import warnings

from itertools import combinations, product
from scipy import stats
from six import iteritems
from sklearn.model_selection._search import BaseSearchCV

with warnings.catch_warnings(record=True) as w:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("always")
    matplotlib.use('Agg')  # create plots from remote

matplotlib.rcParams['pdf.fonttype'] = 42  # avoid bitmapped fonts in pdf
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa

from matplotlib import cm  # noqa
from matplotlib.patches import Rectangle  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa

from palladio.model_assessment import ModelAssessment  # noqa
from palladio.colors import COLORS_HEX  # noqa
from palladio.utils import safe_run  # noqa


def _multicond(*args):
    """Util function to concatenate multiple selection conditions."""
    if len(args) < 1:
        return True
    cond = args[0]
    for arg in args[1:]:
        cond = np.logical_and(cond, arg)
    return cond


def stats_to_file(rstest, r_mean, r_std, p_mean, p_std, metric, base_folder,
                  first_run=True):
    """Save stats to file."""
    filemode = 'w' if first_run else 'a'
    with open(os.path.join(base_folder, 'stats.txt'), filemode) as f:
        f.write("\n------------------------------------------\n"
                "Metric : %s\n" % metric)
        if rstest is not None:
            f.write("Two sample Kolmogorov-Smirnov test p-value: %.3e\n\n" % rstest[1])
            # f.write("Wilcoxon Signed-rank test p-value: %.3e\n" % rstest[1])

        f.write("Regular batch, %s\n"
                "Mean = %.3f, SD = %.3f\n" % (metric, r_mean, r_std))
        if rstest is not None:
            f.write("Permutation batch, %s\n"
                    "Mean = %.3f, SD = %.3f\n" % (metric, p_mean, p_std))


@safe_run
def distributions(v_regular, v_permutation, base_folder=None, metric='nd',
                  first_run=False, is_regression=False, fig=None, ax=None):
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

    fig : matplotlib fig, optional, default None
        Used to plot the distribution on a pre-existing figure

    ax : matplotlib axis, optional, default None
        Used to plot the distribution on a pre-existing axis
    """
    if np.any(np.equal(v_regular, None)) or \
            np.any(np.equal(v_permutation, None)):
        warnings.warn(
            "Cannot create {} plot due to some nan values".format(metric))
        return

    # color_regular = 'C0'
    # kde = True
    if is_regression:
        # scaling factor for percentage plot
        scale = 1
        x_min = np.min(v_regular)
        x_max = np.max(v_regular)
        # _bins = 20
    else:
        # kde = False
        # color_permutation = 'C1'

        # XXX do we need to separate mcc from the rest?
        if metric.lower() == 'matthews_corrcoef':
            scale = 1
            # _bins = np.arange(-1, 1, 0.05)
            x_min = -1.0
        else:
            scale = 1
            # _bins = np.arange(0, 105, 5)
            x_min = 0.0
        x_max = 1.0

    # plt.close('all')
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 10))

    # Compute mean and standard deviation for both batches
    v_regular, v_permutation = np.array(v_regular), np.array(v_permutation)
    r_mean, r_std = np.nanmean(v_regular), np.nanstd(v_regular)
    p_mean, p_std = np.nanmean(v_permutation), np.nanstd(v_permutation)

    sns.distplot(v_regular[~np.isnan(v_regular)] * scale,
                 # label="Regular batch \nMean = {0:.2f}, STD = {1:.2f}"
                 #        .format(reg_mean, reg_std),
                 label="Regular batch \nMean = {0:2.3f}, SD = {1:2.3f}"
                       .format(r_mean, r_std),
                 ax=ax)

    if len(v_permutation) > 0:
        sns.distplot(v_permutation[~np.isnan(v_permutation)] * scale,
                     # label="Permutation batch\nMean = {0:.2f}, STD = {1:.2f}"
                     # .format(perm_mean, perm_std),
                     label="Permutation batch\nMean = {0:2.3f}, SD = {1:2.3f}"
                           .format(p_mean, p_std),
                     ax=ax)

    if len(v_permutation) > 0:
        # rstest = stats.wilcoxon(v_regular, v_permutation)
        # print("[{}] Wilcoxon Signed-rank test: {}".format(metric, rstest))
        # rstest = stats.ranksums(v_regular, v_permutation)
        # print("[{}] Wilcoxon Rank-Sum test: {}".format(metric, rstest))
        rstest = stats.ks_2samp(v_regular, v_permutation)
        print("[{}] Two sample Kolmogorov-Smirnov test: {}".format(metric, rstest))
    else:
        rstest = None

    ax.set_xlabel(metric, fontsize="large")
    ax.set_ylabel("Absolute Frequency", fontsize="large")
    ax.set_title("Distribution of %s" % metric, fontsize=20)

    # ## Determine limits for the x axis
    # x_min = v_permutation.min() - v_permutation.mean()/10
    # x_max = v_regular.max() + v_regular.mean()/10
    # see above

    ax.set_xlim([x_min * scale, x_max * scale])
    ax.legend(
        # bbox_to_anchor=(0., 1.02, 1., .102), loc='upper right',
        # ncol=2, mode="expand", borderaxespad=0., fontsize="large"
    )

    if rstest is not None:
        fig.text(0.1, 0.005,
                 "Two sample Kolmogorov-Smirnov test p-value: {0:.3e}\n"
                 .format(rstest[1]), fontsize=18)

    if base_folder is not None:
        img_type = 'pdf'
        fig.savefig('{}_distribution.{}'.format(metric, img_type),
                    bbox_inches='tight', dpi=600, transparent=True)

    # XXX save to txt; maybe not here?
    if base_folder is not None:
        # Only if base_folder is defined
        stats_to_file(rstest, r_mean, r_std, p_mean, p_std,
                      metric, base_folder, first_run)
    return fig

@safe_run
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
                      color=COLORS_HEX['lightBlue'])
    s_p = plt.scatter(xaxis, s_feat_arr_p[:, 1], marker='h', alpha=0.8, s=10,
                      color=COLORS_HEX['lightRed'])
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
    if base_folder is not None:

        for img_type in ('pdf', 'png'):
            plt.savefig(os.path.join(
                        base_folder, 'figures_{}'.format(img_type),  'manhattan_plot.{}'.format(img_type)),
                        bbox_inches='tight', dpi=300)

        # plt.savefig(os.path.join(base_folder, 'manhattan_plot.png'),
                    # bbox_inches='tight', dpi=300)
    else:
        plt.show()


@safe_run
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
    if n_over_threshold < 1:
        warnings.warn("No regular features over threshold (%.2f) were found." % threshold)
        return

    # sort by frequencies
    sorted_feat_arr = feat_arr[feat_arr[:, 1].argsort()[::-1]]

    plt.close('all')
    sns.set_context('notebook')
    # plt.figure(figsize=(18, 100))
    ax = sns.barplot(
        x=sorted_feat_arr[:2 * n_over_threshold, 0],
        y=sorted_feat_arr[:2 * n_over_threshold, 1],
        color=COLORS_HEX['lightBlue'], alpha=0.9)

    # plot a horizontal line at the height of the selected threshold
    plt.axhline(y=threshold, ls='--', lw=0.5, color='k', label='Threshold')
    # plt.legend((threshold_line, ), ('Threshold',), scatterpoints=1,
    #            loc='upper right', ncol=1, fontsize=12)
    plt.legend(loc='upper right')

    # vertical line to separate selected and not selected features
    mid = np.sum(ax.get_xbound()) / 2.
    plt.axvline(x=mid, ls='-', lw=1, color=COLORS_HEX['lightRed'])

    plt.title("Manhattan plot - top features detail", fontsize=20)
    plt.xlabel("Feature names", fontsize="large")
    plt.ylabel("Relative frequency", fontsize="large")
    plt.ylim([0, 1.05])

    plt.setp(ax.get_xticklabels(), fontsize=2, rotation='vertical')
    if base_folder is not None:

        for img_type in ('pdf', 'png'):
            plt.savefig(os.path.join(
                        base_folder, 'figures_{}'.format(img_type),  'signature_frequencies.{}'.format(img_type)),
                        bbox_inches='tight', dpi=300)

        # plt.savefig(os.path.join(base_folder, 'signature_frequencies.png'),
        #             bbox_inches='tight', dpi=300)
    else:
        plt.show()


@safe_run
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
             color=COLORS_HEX['lightBlue'], label='Regular batch')
    plt.plot(100 * thresh_axis, sel_perm, marker='h', alpha=0.8,
             color=COLORS_HEX['lightRed'], label='Permutation batch')
    plt.axvline(x=threshold * 100, ymin=0, ymax=feat_arr_r.shape[0], ls='--',
                lw=0.5, color='k', label='Threshold')
    plt.legend()
    plt.xlabel("Selection frequency %", fontsize="large")
    plt.ylabel("Number of selected features", fontsize="large")
    plt.xlim(thresh_axis[-1] * 100, thresh_axis[0] * 100)

    if base_folder is not None:
        for img_type in ('pdf', 'png'):
            plt.savefig(os.path.join(
                        base_folder, 'figures_{}'.format(img_type),  'selected_over_threshold.{}'.format(img_type)),
                        bbox_inches='tight', dpi=300)

        # plt.savefig(os.path.join(base_folder, 'selected_over_threshold.png'))
    else:
        plt.show()


@safe_run
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
    if indep_var is not None:
        grid = zip(indep_var, [param_grid[x] for x in indep_var])
    else:
        grid = sorted(
            list(iteritems(param_grid)), key=lambda item: len(item[1]))[-2:]
        indep_var = [name[0] for name in grid]

    if len(grid) < 1:
        warnings.warn("No grid parameters, cannot create validation plot")
        return
    elif len(grid) < 2:
        # warnings.warn("Only one grid parameter, cannot create 3D plot")
        score_plot(param_grid, results, indep_var[0], pivoting_var,
                   base_folder=base_folder, logspace=logspace,
                   plot_errors=plot_errors, is_regression=is_regression)
        return

    comb = combinations(grid, 2)
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
    scoring = 'error' if plot_errors else 'score'
    legend_labels = np.array(['Train ', 'Validation '], dtype=object) + scoring
    for id_pivot, (pivot, value) in enumerate(zip(pivot_names, pivots)):
        for id_param, (param1, param2) in enumerate(comb):
            param_names = 'param_' + np.array(
                [param1[0], param2[0]], dtype=object)

            fig = plt.figure()
            ax = fig.gca(projection='3d', axisbg='gray')
            plt.gca().patch.set_facecolor('white')
            # ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
            # ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
            # ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
            ax.w_xaxis.gridlines.set_lw(3.0)
            ax.w_yaxis.gridlines.set_lw(3.0)
            ax.w_zaxis.gridlines.set_lw(3.0)
            legend_handles = []
            dff = pd.DataFrame(results['cv_results_'])

            # get parameter grid from the first row, since they should be equal
            # but pivoting
            if len(pivot) == 0:
                # no pivoting
                cond = np.ones(
                    dff.iloc[0][param_names[0]].data.size, dtype=bool)
            else:
                cond = _multicond(*[dff.iloc[0]['param_' + p].data == v
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
                    (COLORS_HEX['lightOrange'], COLORS_HEX['lightBlue']),
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
                    c=COLORS_HEX['darkRed'])

            # fig.colorbar()
            ax.legend(legend_handles, legend_labels[:len(legend_handles)],
                      loc='best')
            ax.set_title('average KCV %s, pivot %s = %s' % (
                scoring, pivot, value))
            ax.set_xlabel(log10_x + param_names[0][6:])
            ax.set_ylabel(log10_y + param_names[1][6:])
            ax.set_zlabel("avg kcv %s" % scoring)
            ax.set_zlim([0., 1])
            plt.tight_layout()

            if base_folder is not None:
                for img_type in ('pdf', 'png'):
                    plt.savefig(os.path.join(
                        base_folder, 'kcv_{}_piv{}_comb{}.{}'.format(
                            scoring, id_pivot, id_param, img_type)))
            else:
                plt.show()


def _get_best_params(obj):
    # if obj is a ModelAssessment, then get the first GridSearch
    if isinstance(obj, ModelAssessment):
        obj = pd.DataFrame(obj.cv_results_).sort_values(
            'test_score', ascending=False).iloc[0].estimator
    elif not isinstance(obj, BaseSearchCV):
        raise NotImplementedError("This can only work with a ModelAssessment "
                                  "or BaseSearchCV-based object. You passed "
                                  "a %s object" % obj.__class__.__name__)

    return obj.best_params_


def score_surfaces_searchcv(
    mdl, param_grid=None, indep_vars=None, pivot=None, base_folder=None,
    logspace=None, plot_errors=False, is_regression=False, filename=None,
        zlim=None):
    """Plot training and validation score for a generic search CV class.

    This produces a 3D plot.

    Parameters
    ----------
    mdl : type
        BaseSearchCV-derived object.
    param_grid : type
        Description of parameter `param_grid` (the default is None).
    indep_vars : type
        Description of parameter `indep_vars` (the default is None).
    pivot : type
        Description of parameter `pivot` (the default is None).
    base_folder : type
        Description of parameter `base_folder` (the default is None).
    logspace : type
        Description of parameter `logspace` (the default is None).
    plot_errors : type
        Description of parameter `plot_errors` (the default is False).
    is_regression : type
        Description of parameter `is_regression` (the default is False).
    filename : type
        Description of parameter `filename` (the default is None).
    zlim : type
        Description of parameter `zlim` (the default is None).

    """
    param_grid = param_grid or mdl.param_grid
    if not indep_vars:
        float_values = dict([(k, v) for k, v in iteritems(param_grid)
                             if isinstance(v[0], np.float)])
        vv = sorted(list(iteritems(float_values)), key=lambda item: len(item[1]))[-2:]
        indep_vars = [name[0] for name in vv]
    else:
        vv = zip(indep_vars, [param_grid[x] for x in indep_vars])

    if len(indep_vars) < 1:
        warnings.warn("No grid parameters, cannot create validation plot")
        return
    elif len(indep_vars) < 2:
        return score_plot_searchcv(
            mdl, param_grid, indep_vars[0], pivot,
            base_folder=base_folder, logspace=logspace,
            plot_errors=plot_errors, is_regression=is_regression,
            filename=filename)

    comb = combinations(vv, 2)
    df_result = pd.DataFrame(mdl.cv_results_)
    best_params_ = _get_best_params(mdl)

    if pivot is None:
        pivot = []
    possible_pivot = list(set(param_grid.keys()).difference(set(indep_vars)))
    fixed_pivot = list(set(possible_pivot).difference(set(pivot)))
    fixed_conditions = _multicond(*[
        df_result['param_' + p] == best_params_[p] for p in fixed_pivot])
    cond_pivots = []
    for p in pivot:
        cond_pivots.append([
            df_result['param_' + p] == i for i in
            df_result['param_' + p].unique()])

    conditions = [_multicond(fixed_conditions, *i) for i in list(product(*cond_pivots))]

    # plt.close('all')
    scoring = 'error' if plot_errors else 'score'
    legend_labels = np.array(['Train ', 'Validation '], dtype=object) + scoring
    for i, variables in enumerate(comb):
        for j, condition in enumerate(conditions):
            if condition is True:
                condition = slice(0,None)
            df_small = df_result.loc[condition, :]
            fixed_pivot_dict = dict(df_small.loc[:, df_result.columns.isin([
                'param_' + x for x in pivot + fixed_pivot])].iloc[0])

            x = df_small["param_" + variables[0][0]]
            y = df_small["param_" + variables[1][0]]

            log10_x, log10_y = '', ''
            if logspace is not None:
                if variables[0][0] in logspace:
                    x = np.log10(x.astype(float))
                    log10_x = r'$log_{10}$ '
                if variables[1][0] in logspace:
                    y = np.log10(y.astype(float))
                    log10_y = r'$log_{10}$ '

            fst_val = x if (x.values == sorted(x.values)).all() else y
            snd_val = y if (x.values == sorted(x.values)).all() else x

            X = x.values.reshape(np.unique(fst_val).size, np.unique(snd_val).size)
            Y = y.values.reshape(np.unique(fst_val).size, np.unique(snd_val).size)

            if plot_errors:
                colors = (cm.Oranges_r, cm.Blues_r)
            else:
                colors = (cm.Oranges, cm.Blues)

            fig = plt.figure()
            ax = fig.gca(projection='3d', axisbg='gray')
            plt.gca().patch.set_facecolor('white')
            # ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
            # ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
            # ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
            ax.w_xaxis.gridlines.set_lw(3.0)
            ax.w_yaxis.gridlines.set_lw(3.0)
            ax.w_zaxis.gridlines.set_lw(3.0)
            legend_handles = [
                Rectangle((0, 0), 1, 1, fc=h) for h in ('darkorange', 'navy')]
            for s, c in zip(('train', 'test'), colors):
                # The score is the mean of each external split
                z = df_small['mean_%s_score' % s].values

                Z = z.reshape(np.unique(fst_val).size, np.unique(snd_val).size)
                if plot_errors:
                    Z = -Z if is_regression else 1 - Z

                ax.plot_surface(
                    X, Y, Z, cmap=c, rstride=1, cstride=1, lw=0,
                    antialiased=False)

            # plot max
            func_max = np.min if plot_errors else np.max
            pos_max = np.where(Z == func_max(Z))
            ax.plot(X[pos_max], Y[pos_max], Z[pos_max], 'o',
                    c=COLORS_HEX['darkRed'])

            # fig.colorbar()
            ax.legend(legend_handles, legend_labels[:len(legend_handles)],
                      loc='best')
            ax.set_title('average KCV %s' % scoring)
            fig.text(0.1, 0.005, fixed_pivot_dict, fontsize=10)
            ax.set_xlabel(log10_x + variables[0][0])
            ax.set_ylabel(log10_y + variables[1][0])
            ax.set_zlabel("avg kcv %s" % scoring)
            if zlim is not None:
                ax.set_zlim(zlim)
            plt.tight_layout()

            if filename is None:
                filename = 'kcv_{}_piv{}_comb{}'.format(scoring, j, i)
            if base_folder is not None:
                for img_type in ('pdf', 'png'):
                    plt.savefig(os.path.join(
                        base_folder, '.'.join((filename, img_type))))
            else:
                plt.show()


def score_plot_searchcv(
    mdl, indep_var=None, pivot=None, base_folder=None, logspace=None,
        plot_errors=False, is_regression=False, filename=None):
    """Plot training and validation score in a generic search CV class.

    This produces a 2D plot.

    Parameters
    ----------
    mdl : type
        Description of parameter `mdl`.
    indep_var : type
        Description of parameter `indep_var` (the default is None).
    pivot : type
        Description of parameter `pivot` (the default is None).
    base_folder : type
        Description of parameter `base_folder` (the default is None).
    logspace : type
        Description of parameter `logspace` (the default is None).
    plot_errors : type
        Description of parameter `plot_errors` (the default is False).
    is_regression : type
        Description of parameter `is_regression` (the default is False).
    filename : type
        Description of parameter `filename` (the default is None).

    Returns
    -------
    type
        Description of returned object.

    """
    best_params_ = _get_best_params(mdl)
    df_result = pd.DataFrame(mdl.cv_results_)
    params = [k for k in mdl.cv_results_['params'][0]]

    if not indep_var:
        float_values = [
            k for k in params
            if isinstance(mdl.cv_results_['params'][0][k], float)]
        # float_values = dict([(k, v) for k, v in iteritems(param_grid)
        #                      if isinstance(v[0], np.float)])
        # indep_var = sorted(list(iteritems(float_values)), key=lambda item: len(item[1]))[-1][0]
        indep_var = float_values[0]

    if pivot is None:
        pivot = []
    possible_pivot = list(set(params).difference(set([indep_var])))
    fixed_pivot = list(set(possible_pivot).difference(set(pivot)))
    fixed_conditions = _multicond(*[
        df_result['param_' + p] == best_params_[p] for p in fixed_pivot])
    cond_pivots = []
    for p in pivot:
        cond_pivots.append([
            df_result['param_' + p] == i for i in
            df_result['param_' + p].unique()])

    conditions = [_multicond(fixed_conditions, *i) for i in list(product(*cond_pivots))]

    # plt.close('all')
    scoring = 'error' if plot_errors else 'score'
    legend_labels = np.array(['Train ', 'Validation '], dtype=object) + scoring
    for j, condition in enumerate(conditions):
        if condition is True:
            condition = slice(0, None)
        df_small = df_result.loc[condition, :]
        fixed_pivot_dict = dict(df_small.loc[:, df_result.columns.isin([
            'param_' + x for x in pivot + fixed_pivot])].iloc[0])

        x = df_small["param_" + indep_var].values.astype(float)
        # in the case they are not ordered
        idx = np.argsort(x)
        x = x[idx]

        f, ax = plt.subplots(1)
        if logspace is not None:
            plot = ax.semilogx if indep_var in logspace else ax.plot
        else:
            plot = ax.plot

        colors = ('darkorange', 'navy')
        for s, c, label in zip(('train', 'test'), colors, legend_labels):
            # The score is the mean of each external split
            try:
                mean_score = df_small['mean_%s_score' % s].values[idx]
            except:
                continue
            std_score = df_small['std_%s_score' % s].values[idx]
            if plot_errors:
                mean_score = -mean_score if is_regression else 1 - mean_score

            plot(x, mean_score, c=c, label=label, lw=2)
            ax.fill_between(x, mean_score - std_score,
                            mean_score + std_score, alpha=0.2,
                            color=c, lw=2)

        # plot max
        func_max = np.min if plot_errors else np.max
        pos_max = np.where(mean_score == func_max(mean_score))[0]
        plot(x[pos_max], mean_score[pos_max], 'o', c='darkred')

        # fig.colorbar()
        ax.legend()
        ax.set_title('average CV %s' % (scoring))
        f.text(0.1, 0.005, fixed_pivot_dict, fontsize=10)
        ax.set_xlabel(indep_var)
        ax.set_ylabel("avg kcv %s" % scoring)
        plt.tight_layout()

        if filename is None:
            filename = 'cv_{}_piv{}_comb{}'.format(scoring, j, i)
        if base_folder is not None:
            for img_type in ('pdf', 'png'):
                plt.savefig(os.path.join(
                    base_folder, '.'.join((filename, img_type))))
        else:
            plt.show()


def score_plot(param_grid, results, indep_var=None, pivoting_var=None,
               base_folder=None, logspace=None, plot_errors=False,
               is_regression=False):
    """Plot error 2d plot.

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
    if indep_var is None:
        indep_var = sorted(
            list(iteritems(param_grid)), key=lambda item: len(item[1]))[-1][0]

    if pivoting_var is None:
        pivoting_var = list(set(param_grid.keys()).difference(set([indep_var])))

        ordered_df = pd.DataFrame(pd.DataFrame(results).sort_values(
            'test_score', ascending=False).iloc[0]['cv_results_']).sort_values(
                'mean_test_score', ascending=False).iloc[0]

        # use best model, one pivot
        pivots = [tuple(ordered_df['param_' + x] for x in pivoting_var)]
    else:
        pivots = list(product(*[param_grid[x] for x in pivoting_var]))

    pivot_names = list(product(*[[x] for x in pivoting_var])) * len(pivots)

    plt.close('all')
    scoring = 'error' if plot_errors else 'score'
    legend_labels = np.array(['Train ', 'Validation '], dtype=object) + scoring
    for id_pivot, (pivot, value) in enumerate(zip(pivot_names, pivots)):
        param_name = 'param_' + indep_var

        f, ax = plt.subplots(1)
        dff = pd.DataFrame(results['cv_results_'])

        # get parameter grid from the first row, since they should be equal
        # but pivoting
        if len(pivot) == 0:
            # no pivoting
            cond = np.ones(
                dff.iloc[0][param_name].data.size, dtype=bool)
        else:
            cond = _multicond(*[dff.iloc[0]['param_' + p].data == v
                                for p, v in zip(pivot, value)])
        param_range = dff.iloc[0][param_name][cond].data.astype(float)
        if logspace is not None:
            plot = ax.semilogx if indep_var in logspace else ax.plot

        for string, color, label in zip(
                ('train', 'test'),
                (COLORS_HEX['lightOrange'], COLORS_HEX['lightBlue']),
                legend_labels):
            # The score is the mean of each external split
            score = np.mean(np.vstack(
                dff['mean_%s_score' % string].tolist()), axis=0)[cond]
            if plot_errors:
                score = -score if is_regression else 1 - score

            # plt.close('all')
            plot(param_range, score, c=color, label=label)

        # plot max
        func_max = np.min if plot_errors else np.max
        pos_max = np.where(score == func_max(score))
        plot(param_range[pos_max], score[pos_max], 'o', c=COLORS_HEX['darkRed'])

        ax.legend()
        ax.set_title('average KCV %s, pivot %s = %s' % (scoring, pivot, value))
        ax.set_xlabel(indep_var)
        ax.set_ylabel("avg kcv %s" % scoring)

        if base_folder is not None:
            for img_type in ('pdf', 'png'):
                plt.savefig(os.path.join(
                base_folder, 'kcv_{}_piv{}_comb{}.{}'.format(
                    scoring, id_pivot, id_param, img_type)))
            # plt.savefig(os.path.join(
            #     base_folder, 'kcv_%s_piv%d_param_%s.png' % (
            #         scoring, id_pivot, indep_var)))
        else:
            plt.show()


def validation_curve_plot(train_scores, test_scores, estimator=None,
                          param_name=None, param_range=None, score=None,
                          plot_errors=False, base_folder=None,
                          title_footer='', title=None, ax=None):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 3))

    if title is None:
        title = "Validation Curve with %s" % (type(estimator).__name__)
    ax.set_title(title + title_footer)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Score" + " (%s)" % score.__name__ if score is not None else "")
    # ax.set_ylim(0.4, 1.1)
    lw = 2
    ax.semilogx(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    ax.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    ax.legend(loc="best")
    return ax
