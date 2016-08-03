import os

# import pandas as pd

import matplotlib

matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

import seaborn as sns

from scipy import stats

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

def distributions(v_regular, v_permutation, base_folder, metric):
    """
    Create a plot of the distributions of performance metrics for both "regular"
    experiments and permutation tests.

    Parameters
    ----------

    v_regular : numpy.array
        The metric values for all regular experiments

    v_permutation : numpy.array
        The metric values for permutation tests

    base_folder : string
        The folder where the plot will be saved

    metric : string
        The object metric, this should be in ['Accuracy', 'Balanced Accuracy',
        'MCC', 'Precision', 'Recall', 'F1']. It is going to be used to set the
        title and the xlabel.
    """

    # scaling factor for percentage plot
    if metric.lower() not in ['mcc']:
        scale = 100
        _bins = np.arange(0, 105, 5)
        x_min = 0.0
    else:
      scale = 1
      _bins = np.arange(-1, 1, 0.05)
      x_min = -1.0
    x_max = 1.0

    fig, ax = plt.subplots(figsize=(18, 10))

    args = {
        'norm_hist' : False,
        'kde' : False,
        'bins' : _bins,
        # hist_kws : {'alpha' : 0.8}
    }

    reg_mean = np.mean(v_regular)
    reg_std = np.std(v_regular)

    perm_mean = np.mean(v_permutation)
    perm_std = np.std(v_permutation)

    sns.distplot(v_permutation*scale,
                 label = "Permutation tests \nMean = {0:.2f}, STD = {1:.2f}".format(perm_mean, perm_std),
                 color = colorsHex['lightRed'], ax = ax,
                 hist_kws = {'alpha' : 0.8}, **args)
    sns.distplot(v_regular*scale,
                 label = "Regular experiments \nMean = {0:.2f}, STD = {1:.2f}".format(reg_mean, reg_std),
                 color = colorsHex['lightGreen'], ax = ax,
                 hist_kws = {'alpha' : 0.8}, **args)

    ### Fit a gaussian with permutation data (DEPRECATED)
    # (mu, sigma) = stats.norm.fit(v_permutation*100)
    # kstest = stats.kstest(v_regular*100, 'norm', args=(mu, sigma))
    ### Wilcoxon rank sum test
    rstest = stats.ranksums(v_regular, v_permutation)

    with open(os.path.join(base_folder, 'stats.txt'), 'a')  as f:

        # f.write("Kolmogorov-Smirnov test p-value: {0:.3e}\n".format(kstest[1]))
        # f.write("Testing distributions")
        f.write("\n------------------------------------------\n")
        f.write("Metric : {}\n".format(metric))
        f.write("Wilcoxon Rank-Sum test p-value: {0:.3e}\n".format(rstest[1]))
        f.write("\n")

        f.write("Regular experiments, {}\n".format(metric))
        f.write("Mean = {0:.2f}, STD = {1:.2f}\n".format(reg_mean, reg_std))

        f.write("Permutation tests, {}\n".format(metric))
        f.write("Mean = {0:.2f}, STD = {1:.2f}\n".format(perm_mean, perm_std))

    # print("Kolmogorov-Smirnov test: {}".format(kstest))
    print("[{}] Wilcoxon Rank-Sum test: {}".format(metric, rstest))

    if metric.lower() not in ['mcc']:
        plt.xlabel("{} (%)".format(metric), fontsize="large")
    else:
        plt.xlabel("{}".format(metric), fontsize="large")
    plt.ylabel("Absolute Frequency", fontsize="large")

    plt.title("Distribution of {}".format(metric), fontsize = 20)

    ### Determine limits for the x axis
    # x_min = v_permutation.min() - v_permutation.mean()/10
    # x_max = v_regular.max() + v_regular.mean()/10
    # see above


    plt.xlim([x_min*scale,x_max*scale])

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper right',
           ncol=2, mode="expand", borderaxespad=0., fontsize="large")

    # fig.text(0.1, 0.01, "Kolmogorov-Smirnov test p-value: {0:.3e}\n".format(kstest[1]), fontsize=18)
    fig.text(0.1, 0.005, "Wilcoxon Rank-Sum test p-value: {0:.3e}\n".format(rstest[1]), fontsize=18)

    plt.savefig(os.path.join(base_folder, 'permutation_'+metric+'_distribution.pdf'))

def features_manhattan(sorted_keys, frequencies_true, frequencies_perm,
                       base_folder, N_jobs_regular, N_jobs_permutation, threshold = 75):
    """
    Parameters
    ----------

    sorted_keys : list
    """

    r_sorted_keys = reversed(sorted_keys)

    fake_x = np.arange(len(sorted_keys))

    y_true = list()
    y_perm = list()

    for k in r_sorted_keys:
        y_true.append(frequencies_true[k])
        y_perm.append(frequencies_perm[k])

    y_true = np.array(y_true)
    y_perm = np.array(y_perm)

    plt.figure()

    s_t = plt.scatter(fake_x, y_true, marker = 'h', alpha = 0.8, s = 10, color = colorsHex['lightGreen'])
    s_p = plt.scatter(fake_x, y_perm, marker = 'h', alpha = 0.8, s = 10, color = colorsHex['lightRed'])
    threshold_line = plt.axhline(y=threshold, ls = '--', lw = 0.5, color = 'k')

    plt.xlim([-5,len(sorted_keys) + 5])
    plt.ylim([-5,N_jobs_regular+5])

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

    plt.legend((s_t, s_p, threshold_line),
           ('Real signature', 'Permutation signature', 'Threshold'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)

    plt.xlabel('Features')
    plt.ylabel('Absolute frequencies ({} regular, {} permutation)'.format(N_jobs_regular, N_jobs_permutation))

    plt.title("Feature frequencies")

    plt.savefig(os.path.join(base_folder, 'manhattan_plot.pdf'))

def feature_frequencies(sorted_keys, frequencies, base_folder, threshold = 75):
    """
    Plot a bar chart of the first 2 x M features in a signature,
    where M is the number of features whose frequencies is over a given threshold

    Parameters
    ----------

    sorted_keys : list

    frequencies : dict
    """

    x = list()
    y = list()

    M = 0

    r_sorted_keys = reversed(sorted_keys)

    for k in r_sorted_keys:

        if frequencies[k] >= threshold:

            M += 1

        else:
            break

    # print "M = {}".format(M)

    N = 2*M
    r_sorted_keys = reversed(sorted_keys)
    for k in r_sorted_keys:

        if N == 0:
            break

        # x.append(k)
        x.append('_' + str(k)) ### This is required for some reason
        y.append(frequencies[k])

        # print frequencies[k]

        N -= 1

    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(18, 10))

    plt.title("Manhattan plot - top features detail", fontsize = 20)

    ax = sns.barplot(x = x, y = y, color = colorsHex['lightGreen'], alpha = 0.9)

    ### Rotate x ticks
    for item in ax.get_xticklabels():
        item.set_rotation(45)

    plt.savefig(os.path.join(base_folder, 'signature_frequencies.pdf'))
    ### plot a horizontal line at the height of the selected threshold
    threshold_line = plt.axhline(y=threshold, ls = '--', lw = 0.5, color = 'k')

    plt.legend((threshold_line, ),
           ('Threshold',),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=12)

    ### plot a vertical line which separates selected features from those not selected
    xmin, xmax = ax.get_xbound()
    mid = float(xmax + xmin)/2
    plt.axvline(x=mid, ls = '-', lw = 1, color = colorsHex['lightRed'])

    plt.xlabel("Feature names", fontsize="large")
    plt.ylabel("Absolute Frequency", fontsize="large")

    plt.savefig(os.path.join(base_folder, 'signature_frequencies.pdf'))


def selected_over_threshold(frequencies_true, frequencies_perm, N_jobs_regular, N_jobs_permutation, base_folder, threshold = 75):
    """Plot the selection trend against the selection frequency threshold.


    Parameters
    ----------

    sorted_keys : list
    """
    # horizontal axis REVERSED
    thresh_axis = np.linspace(0,1,21)[::-1]

    # number of selected features for true/perm
    y_true = list()
    y_perm = list()

    # Unroll dict
    for k in frequencies_true.keys():
        y_true.append(frequencies_true[k])
        y_perm.append(frequencies_perm[k])
    y_true = np.array(y_true)
    y_perm = np.array(y_perm)

    # tot number features
    n_feat = len(y_true)

    # init selected counts
    sel_true = np.zeros(len(thresh_axis))
    sel_perm = np.zeros(len(thresh_axis))

    # iterate over the horiz axis (i.e. the selection freq thresh)
    for i, thr in enumerate(thresh_axis):
        # sel_true[i] = np.count_nonzero(y_true > thr)
        # sel_perm[i] = np.count_nonzero(y_perm > thr)
        sel_true[i] = np.count_nonzero(y_true >= thr*N_jobs_regular)
        sel_perm[i] = np.count_nonzero(y_perm >= thr*N_jobs_permutation)

    # make plot
    plt.figure()
    plt.plot(100*thresh_axis, sel_true, marker = 'h', alpha = 0.8, color = colorsHex['lightGreen'], label='Real signature')
    plt.plot(100*thresh_axis, sel_perm, marker = 'h', alpha = 0.8, color = colorsHex['lightRed'], label='Permutation signature')
    plt.axvline(x=100-threshold, ymin=0, ymax=n_feat, ls = '--', lw = 0.5, color = 'k', label='Threshold')
    plt.legend()
    plt.xlabel("Selection frequency %", fontsize="large")
    plt.ylabel("Number of selected features", fontsize="large")

    plt.savefig(os.path.join(base_folder, 'selected_over_threshold.pdf'))



def kcv_err_surfaces(kcv_err, exp, base_folder, param_ranges):
    """Generate plot surfaces for training and test error across experiments.

        Parameters
        ----------

        kcv_err : list of arrays

        exp : string
            Either 'regular' or 'permutation'

        base_folder: string
            Path to base output folder.

        param_ranges : list
            list containing all the hyperparameter ranges. When using l1l2Classifier
            this is [tau_range, lambda_range]
    """
    def most_common(lst):
        """Return the most common element in a list."""
        return max(set(lst), key=lst.count)

    # average errors dictionary
    avg_err = dict()

    # iterate over tr and ts
    for k in kcv_err.keys():
        # it may happen that for a certain experiment a solution is not
        # provided for each values of tau. we want to exclude such situations
        mode = most_common([e.shape for e in kcv_err[k]]) # get the most common size of the matrix
        kcv_err_k = filter(lambda x: x.shape == mode, kcv_err[k])
        # get the number of experiment where everything worked fine
        n_exp = len(kcv_err_k)
        # perform reduce operation
        avg_err[k] = sum(kcv_err_k) / float(n_exp)
        # this is like avg_err = reduce(lambda x,y: x+y, kcv_err_k) / float(n_exp)

    ### PLOT SECTION
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cmaps = [cm.Oranges, cm.Blues]
    fc = {'tr': colorsHex['lightBlue'], 'ts': colorsHex['lightOrange']}

    # legends
    legend_handles = []
    legend_labels = ['Test Error', 'Train Error']

    # error surface
    for k, c in zip(avg_err.keys(), cmaps):
        ZZ = avg_err[k]

        # xx = np.arange(0,mode[0]) # FIX THIS!!! we need the actual values for tau and lambda
        # yy = np.arange(0,mode[1])
        xx = np.log10(param_ranges[0]) # tau
        yy = np.log10(param_ranges[1]) # lambda
        XX, YY = np.meshgrid(xx, yy)

        surf = ax.plot_surface(XX, YY, ZZ.T, rstride=1,
                                           cstride=1,
                                           linewidth=0,
                                           antialiased=False,
                                           cmap=c)

        legend_handles.append(Rectangle((0, 0), 1, 1, fc=fc[k])) # proxy handle

    # plot minimum
    ZZ = avg_err['ts']
    x_min_idxs, y_min_idxs = np.where(ZZ == np.min(ZZ))
    ax.plot(xx[x_min_idxs], yy[y_min_idxs],
            ZZ[x_min_idxs, y_min_idxs], 'o', c=colorsHex['darkBlue'])

    # fig.colorbar()
    ax.set_title('average KCV error of '+exp+' experiments')
    ax.set_ylabel(r"$log_{10}(\lambda)$")
    ax.set_xlabel(r"$log_{10}(\tau)$")
    ax.set_zlabel("avg kcv err")
    ax.legend(legend_handles, legend_labels[:len(legend_handles)], loc='best')

    # plt.legend()
    plt.savefig(os.path.join(base_folder, 'kcv_err_'+exp+'.pdf'))





















############################################################################
