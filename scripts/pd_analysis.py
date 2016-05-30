#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import os, sys, imp

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from scipy import stats

import cPickle as pkl

# def analyze_experiment(exp_folder, config):
#     
#     with open(os.path.join(exp_folder, 'result.pkl'), 'r') as f:
#         result = pkl.load(f)
#         
#     Y_ts = result['labels_ts'] # the actual labels
#     
#     analysis_results = dict()
#     analysis_results['summaries'] = list()
#     analysis_results['accuracy'] = list()
#     analysis_results['balanced_accuracy'] = list()
#     
#     for aux_y in result['prediction_ts_list']:
#         
#         ### XXX TODO fix cases in which Y_lr = 0
#         Y_lr = np.sign(aux_y.ravel())
#         # Y_lr = np.sign(Y_lr-0.1)
#         
#         analysis_results['accuracy'].append((Y_ts == Y_lr).sum()/float(len(Y_ts)))
#         
#         TP = np.sum((Y_lr == 1) * (Y_ts == Y_lr))
#         FP = np.sum((Y_lr == 1) * (Y_ts != Y_lr))
#         TN = np.sum((Y_lr == -1) * (Y_ts == Y_lr))
#         FN = np.sum((Y_lr == -1) * (Y_ts != Y_lr))
#         
#         # print TP, TN, FP, FN
#         
#         balanced_accuracy = 0.5 * ( (TP / float(TP + FN)) + (TN / float(TN + FP)) )
#         
#         analysis_results['balanced_accuracy'].append(balanced_accuracy)
#         
#         analysis_results['selected_list'] = result['selected_list']
#         
#         # summary['MCC']
#         
#     return analysis_results

def analyze_experiment(exp_folder, config):
    
    with open(os.path.join(exp_folder, 'result.pkl'), 'r') as f:
        result = pkl.load(f)
        
    Y_ts = result['labels_ts'] # the actual labels
    
    analysis_results = dict()
    # analysis_results['summary']
    # analysis_results['accuracy']
    # analysis_results['balanced_accuracy']
    
    aux_y = result['prediction_ts_list']
        
    ### XXX TODO fix cases in which Y_lr = 0
    Y_lr = np.sign(aux_y.ravel())
    # Y_lr = np.sign(Y_lr-0.1)
    
    analysis_results['accuracy'] = (Y_ts == Y_lr).sum()/float(len(Y_ts))
    
    TP = np.sum((Y_lr == 1) * (Y_ts == Y_lr))
    FP = np.sum((Y_lr == 1) * (Y_ts != Y_lr))
    TN = np.sum((Y_lr == -1) * (Y_ts == Y_lr))
    FN = np.sum((Y_lr == -1) * (Y_ts != Y_lr))
    
    balanced_accuracy = 0.5 * ( (TP / float(TP + FN)) + (TN / float(TN + FP)) )
    
    analysis_results['balanced_accuracy'] = balanced_accuracy
    
    analysis_results['selected_list'] = result['selected_list']
        
    return analysis_results

def analyze_experiments(base_folder, config):
    """
    Perform a preliminar analysis on all experiments producing a summary
    for each experiment that will be subsequently used for the aggregate analysis
    
    Parameters
    ----------
    
    base_folder : string, the folder containing ALL experiments (regular and permutations)
    """
    
    # Local imports, in order to select backend on startup
    from matplotlib import pyplot as plt
    from l1l2signature import internals as l1l2_core
    from l1l2signature import utils as l1l2_utils
    
    print("#--------------------------------------------------------------------")
    print('Reading data... ')

    data_path = os.path.join(base_folder, 'data_file')
    labels_path = os.path.join(base_folder, 'labels_file')
    
    pd_data = pd.read_csv(data_path)
    
    if config.samples_on == 'col':
        pd_data.index = pd_data[pd_data.columns[0]] # Correctly use the first column as index 
        pd_data =  pd_data.iloc[:,1:] # and remove it from the actual data
    
    if not config.data_preprocessing is None:
        config.data_preprocessing.load_data(pd_data)
        pd_data = config.data_preprocessing.process()
        
    probeset_names = pd_data.index
    
    pd_labels = pd.read_csv(labels_path)
    pd_labels.index = pd_labels[pd_labels.columns[0]] # Correctly use the first column as index 
    pd_labels = pd_labels.iloc[:,1:] # and remove it from labels
    
    if not config.positive_label is None:
        poslab = config.positive_label
    else:
        uv = np.sort(np.unique(pd_labels.values))
        
        if len(uv) != 2:
            raise Exception("More than two unique values in the labels array")
        
        poslab = uv[0]
    
    def _toPlusMinus(x) :
        """
        Converts the values in the labels
        """
        if x == poslab:
            return +1.0
        else:
            return -1.0
    
    pd_labels_mapped = pd_labels.applymap(_toPlusMinus)
    
    data = pd_data.as_matrix().T
    labels = pd_labels_mapped.as_matrix().ravel()
    
    # print len(probeset_names)
    
    MCC_regular = list()
    MCC_permutation = list()
    
    acc_regular = list()
    acc_permutation = list()
    
    selected_regular = dict(zip(probeset_names, np.zeros((len(probeset_names),))))
    selected_permutation = dict(zip(probeset_names, np.zeros((len(probeset_names),))))
    
    for exp_folder in [os.path.join(base_folder, x) for x in os.listdir(base_folder)]:
        
        if os.path.isdir(exp_folder):
            
            analysis_result = analyze_experiment(exp_folder, config)
            
            selected_probesets = probeset_names[analysis_result['selected_list']]
            
            if exp_folder.split('/')[1].startswith('regular'):
                
                acc_regular.append(analysis_result['balanced_accuracy'])
                
                for p in selected_probesets:
                    selected_regular[p] += 1
                
            elif exp_folder.split('/')[1].startswith('permutation'):
                
                acc_permutation.append(analysis_result['balanced_accuracy'])
                
                for p in selected_probesets:
                    selected_permutation[p] += 1
                
            else:
                print "error"
            
    out = {
        'v_regular' : np.array(acc_regular),
        'v_permutation' : np.array(acc_permutation),
        'selected_regular' : selected_regular,
        'selected_permutation' : selected_permutation,
    }
    
    return out

def plot_distributions(v_regular, v_permutation, base_folder):
    """
    Create a plot of the distributions of accuracies for both "regular" experiments and permutation tests
    
    Parameters
    ----------
    
    v_regular : numpy.array
        The accuracy values for all regular experiments
       
    v_permutation : numpy.array
        The accuracy values for permutation tests
       
    base_folder : string
        The folder where the plot will be saved
    """
    
    # nbins = 15
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    args = {
        'norm_hist' : False,
        'kde' : False,
        # 'bins' : np.arange(0,1.05,0.05)
        'bins' : np.arange(0,105,5),
        # hist_kws : {'alpha' : 0.8}
    }
    
    reg_mean = np.mean(v_regular)
    reg_std = np.std(v_regular)
    
    perm_mean = np.mean(v_permutation)
    perm_std = np.std(v_permutation)
    
    sns.distplot(v_permutation*100, label = "Permutation tests \nMean = {0:.2f}, STD = {1:.2f}".format(perm_mean, perm_std), color = 'r', ax = ax, hist_kws = {'alpha' : 0.8}, **args)
    sns.distplot(v_regular*100, label = "Regular experiments \nMean = {0:.2f}, STD = {1:.2f}".format(reg_mean, reg_std), color = '#99cc00', ax = ax, hist_kws = {'alpha' : 0.8}, **args)
    
    ### Fit a gaussian with permutation data
    (mu, sigma) = stats.norm.fit(v_permutation*100)
    
    # print (mu, sigma)
    
    kstest = stats.kstest(v_regular*100, 'norm', args=(mu, sigma))
    rstest = stats.ranksums(v_regular, v_permutation)
    
    
    
    with open(os.path.join(base_folder, 'stats.txt'), 'w')  as f:
        
        f.write("Kolmogorov-Smirnov test p-value: {0:.3e}\n".format(kstest[1]))
        f.write("Wilcoxon Rank-Sum test p-value: {0:.3e}\n".format(rstest[1]))
    
    print("Kolmogorov-Smirnov test: {}".format(kstest))
    print("Wilcoxon Rank-Sum test: {}".format(rstest))
    
    
    plt.xlabel("Balanced Accuracy (%)", fontsize="large")
    plt.ylabel("Absolute Frequency", fontsize="large")
    
    plt.title("Distribution of accuracies", fontsize = 20)
    
    
    ### Determine limits for the x axis
    x_min = v_permutation.min() - v_permutation.mean()/10
    x_max = v_regular.max() + v_regular.mean()/10
    
    plt.xlim([x_min*100,x_max*100])
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper right',
           ncol=2, mode="expand", borderaxespad=0., fontsize="large")
    
    # fig.text(0.1, 0.01, "Kolmogorov-Smirnov test p-value: {0:.3e}\n".format(kstest[1]), fontsize=18)
    fig.text(0.1, 0.005, "Wilcoxon Rank-Sum test p-value: {0:.3e}\n".format(rstest[1]), fontsize=18)
    
    plt.savefig(os.path.join(base_folder, 'permutation_acc_distribution.pdf'))

def features_manhattan_plot(sorted_keys, frequencies_true, frequencies_perm, base_folder, threshold = 75):
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
    
    s_t = plt.scatter(fake_x, y_true, marker = 'h', alpha = 0.8, s = 10, color = '#99cc00')
    s_p = plt.scatter(fake_x, y_perm, marker = 'h', alpha = 0.8, s = 10, color = 'r')
    threshold_line = plt.axhline(y=threshold, ls = '--', lw = 0.5, color = 'k')
    
    plt.xlim([-5,len(sorted_keys) + 5])
    plt.ylim([-5,105])
    
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
    plt.ylabel('Absolute frequencies')
    
    plt.savefig(os.path.join(base_folder, 'manhattan_plot.pdf'))
    
def plot_feature_frequencies(sorted_keys, frequencies, base_folder, threshold = 75):
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
    
    plt.figure(figsize=(18, 10))
    
    plt.title("Manhattan plot - top features detail", fontsize = 20)
    
    ax = sns.barplot(x = x, y = y, color = '#99cc00', alpha = 0.9)
    
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
    plt.axvline(x=mid, ls = '-', lw = 1, color = 'r')
    
    plt.xlabel("Feature names", fontsize="large")
    plt.ylabel("Absolute Frequency", fontsize="large")
    
    plt.savefig(os.path.join(base_folder, 'signature_frequencies.pdf'))
    
def main():
    
    base_folder = sys.argv[1]
    
    config = imp.load_source('config', os.path.join(base_folder, 'config.py'))
    
    threshold = int(config.N_jobs_regular * config.frequency_threshold)
    
    out = analyze_experiments(base_folder, config)
    
    v_regular, v_permutation = out['v_regular'], out['v_permutation']
    selected_regular, selected_permutation = out['selected_regular'], out['selected_permutation']
    
    ### Manually sorting stuff
    sorted_keys_regular = sorted(selected_regular, key=selected_regular.__getitem__)
    sorted_keys_permutation = sorted(selected_permutation, key=selected_permutation.__getitem__)
    
    
    
    with open(os.path.join(base_folder, 'signature_regular.txt'), 'w') as f:
        line_drawn = False
        for k in reversed(sorted_keys_regular):
            if not line_drawn and selected_regular[k] < threshold:
                line_drawn = True
                f.write("="*40)
                f.write("\n")
            f.write("{} : {}\n".format(k,selected_regular[k]))
            
    with open(os.path.join(base_folder, 'signature_permutation.txt'), 'w') as f:
        line_drawn = False
        for k in reversed(sorted_keys_permutation):
            if not line_drawn and selected_permutation[k] < threshold:
                line_drawn = True
                f.write("="*40)
                f.write("\n")
            f.write("{} : {}\n".format(k,selected_permutation[k]))
            
    plot_distributions(v_regular, v_permutation, base_folder)
    
    plot_feature_frequencies(sorted_keys_regular, selected_regular, base_folder, threshold = threshold)
    
    features_manhattan_plot(sorted_keys_regular, selected_regular, selected_permutation, base_folder, threshold = threshold)
    
if __name__ == '__main__':
    main()