#!/usr/bin/env python
import os
import sys
import imp
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pkl

from scipy import stats

from palladio import plotting


def analyze_experiment(exp_folder, config):

    with open(os.path.join(exp_folder, 'result.pkl'), 'r') as f:
        result = pkl.load(f)

    with open(os.path.join(exp_folder, 'in_split.pkl'), 'r') as f:
        in_split = pkl.load(f)

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

    # save balanced accuracy
    analysis_results['balanced_accuracy'] = balanced_accuracy

    # save selected_list
    analysis_results['selected_list'] = result['selected_list']

    # save kcv errors
    analysis_results['kcv_err_ts'] = result['kcv_err_ts']
    analysis_results['kcv_err_tr'] = result['kcv_err_tr']

    # save params ranges
    analysis_results['param_ranges'] = in_split['param_ranges']

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

    dataset = config.dataset_class(
        config.dataset_files,
        config.dataset_options,
        is_analysis=True
    )

    data, labels, feature_names  = dataset.load_dataset(base_folder)

    feature_names = np.array(feature_names)

    MCC_regular = list() # not used so far
    MCC_permutation = list()  # not used so far

    # accuracy containers
    acc_regular = list()
    acc_permutation = list()

    # selection containers
    selected_regular = dict(zip(feature_names, np.zeros((len(feature_names),))))
    selected_permutation = dict(zip(feature_names, np.zeros((len(feature_names),))))

    # kcv error containers
    kcv_err_regular = {'tr': list(), 'ts': list()}
    kcv_err_permutation = {'tr': list(), 'ts': list()}

    for exp_folder in [os.path.join(base_folder, x) for x in os.listdir(base_folder)]:
        if os.path.isdir(exp_folder):
            analysis_result = analyze_experiment(exp_folder, config)

            selected_probesets = feature_names[analysis_result['selected_list']]

            if exp_folder.split('/')[-1].startswith('regular'):

                # update accuracy
                acc_regular.append(analysis_result['balanced_accuracy'])
                # update selection
                for p in selected_probesets:
                    selected_regular[p] += 1
                # update kcv errors
                kcv_err_regular['tr'].append(analysis_result['kcv_err_tr'])
                kcv_err_regular['ts'].append(analysis_result['kcv_err_ts'])

            elif exp_folder.split('/')[-1].startswith('permutation'):

                # update accuracy
                acc_permutation.append(analysis_result['balanced_accuracy'])
                # update selection
                for p in selected_probesets:
                    selected_permutation[p] += 1
                # update kcv errors
                kcv_err_permutation['tr'].append(analysis_result['kcv_err_tr'])
                kcv_err_permutation['ts'].append(analysis_result['kcv_err_ts'])

            else:
                print "error"

    # store the actual parameters ranges
    param_ranges = analysis_result['param_ranges']
    out = {'v_regular': np.array(acc_regular),
           'v_permutation': np.array(acc_permutation),
           'selected_regular': selected_regular,
           'selected_permutation': selected_permutation,
           'kcv_err_regular': kcv_err_regular,
           'kcv_err_permutation': kcv_err_permutation,
           'param_ranges': param_ranges
           }

    return out


def main():

    base_folder = sys.argv[1]

    config = imp.load_source('config', os.path.join(base_folder, 'config.py'))

    threshold = int(config.N_jobs_regular * config.frequency_threshold)

    out = analyze_experiments(base_folder, config)

    v_regular, v_permutation = out['v_regular'], out['v_permutation']
    selected_regular, selected_permutation = out['selected_regular'], out['selected_permutation']
    kcv_err_regular, kcv_err_permutation = out['kcv_err_regular'], out['kcv_err_permutation']
    param_ranges = out['param_ranges']

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

    for kcv_err, exp in zip([kcv_err_regular, kcv_err_permutation], ['regular', 'permutation']):
        plotting.kcv_err_surfaces(kcv_err, exp, base_folder, param_ranges)

if __name__ == '__main__':
    main()
