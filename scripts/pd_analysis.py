#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
import os, sys, imp

# import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from scipy import stats

import cPickle as pkl

from palladio._core import load_data
from palladio import plotting

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

    # Data paths
    data_path = os.path.join(base_folder, 'data_file')
    labels_path = os.path.join(base_folder, 'labels_file')
    config_path = os.path.join(base_folder, 'config.py')

    ### Read data, labels, variables names
    data, labels, probeset_names = load_data(config_path, config, data_path, labels_path)

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

            if exp_folder.split('/')[-1].startswith('regular'):

                acc_regular.append(analysis_result['balanced_accuracy'])

                for p in selected_probesets:
                    selected_regular[p] += 1

            elif exp_folder.split('/')[-1].startswith('permutation'):

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

    plotting.distributions(v_regular, v_permutation, base_folder)

    plotting.feature_frequencies(sorted_keys_regular, selected_regular, base_folder, threshold = threshold)

    plotting.features_manhattan(sorted_keys_regular, selected_regular, selected_permutation, base_folder, threshold = threshold)

    plotting.selected_over_threshold(selected_regular, selected_permutation,
                                     config.N_jobs_regular, config.N_jobs_permutation,
                                     base_folder, threshold = threshold)

if __name__ == '__main__':
    main()
