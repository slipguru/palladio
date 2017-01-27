#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import imp
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pkl

from scipy import stats

from palladio import plotting


def analyze_experiment(exp_folder, config, poslab):
    """TODO."""
    with open(os.path.join(exp_folder, 'result.pkl'), 'r') as f:
        result = pkl.load(f)

    with open(os.path.join(exp_folder, 'in_split.pkl'), 'r') as f:
        in_split = pkl.load(f)

    Y_ts = result['labels_ts']  # the actual labels

    analysis_results = dict()
    # analysis_results['summary']
    # analysis_results['accuracy']
    # analysis_results['balanced_accuracy']

    aux_y = result['prediction_ts_list']

    # ## XXX TODO fix cases in which Y_lr = 0
    Y_lr = np.sign(aux_y.ravel())
    # Y_lr = np.sign(Y_lr-0.1)

    # evaluate performance metrics
    print(result)
    TP = np.sum((Y_lr == 1) * (Y_ts == Y_lr))
    FP = np.sum((Y_lr == 1) * (Y_ts != Y_lr))
    TN = np.sum((Y_lr == -1) * (Y_ts == Y_lr))
    FN = np.sum((Y_lr == -1) * (Y_ts != Y_lr))

    if float(TP + FP + FN + TN) == 0:
        raise ValueError("Sum of TP, FP FN and TN is zero. Why?")
    accuracy = (TP + TN) / float(TP + FP + FN + TN)
    balanced_accuracy = 0.5 * ((TP / float(TP + FN)) + (TN / float(TN + FP)))

    den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (((TP * TN) - (FP * FN)) / (1.0 if den == 0 else np.sqrt(den)))

    if poslab is not None:
        precision = TP / float(TP + FP) if TP + FP != 0 else None
        recall = TP / float(TP + FN) if TP + FN != 0 else None
        if precision is None or recall is None or precision + recall == 0:
            F1 = None
        else:
            F1 = 2.0 * ((precision * recall) / (precision + recall))
    else:
        precision = recall = F1 = None

    # save performance metrics
    analysis_results['accuracy'] = accuracy
    analysis_results['balanced_accuracy'] = balanced_accuracy
    analysis_results['MCC'] = MCC
    analysis_results['precision'] = precision
    analysis_results['recall'] = recall
    analysis_results['F1'] = F1

    # save selected_list
    analysis_results['selected_list'] = result['selected_list']

    # save kcv errors
    analysis_results['kcv_err_ts'] = result['kcv_err_ts']
    analysis_results['kcv_err_tr'] = result['kcv_err_tr']

    # save params ranges
    analysis_results['param_ranges'] = in_split['param_ranges']

    return analysis_results


def analyze_experiments(base_folder, config):
    """Perform a preliminar analysis on all experiments.

    Produce a summary for each experiment that will be used for
    aggregate analysis.

    Parameters
    ----------
    base_folder : string
        Folder containing ALL experiments (regular and permutations).
    """
    # Local imports, in order to select backend on startup
    from matplotlib import pyplot as plt
    dataset = config.dataset_class(
        config.dataset_files,
        config.dataset_options,
        is_analysis=True
    )
    data, labels, feature_names = dataset.load_dataset(base_folder)
    feature_names = np.array(feature_names)

    # performance metrics containers
    acc_regular = list()
    acc_permutation = list()
    balanced_acc_regular = list()
    balanced_acc_permutation = list()
    MCC_regular = list()
    MCC_permutation = list()
    # I am not checking config.positive_label on purpose.
    # This comes in handy afterwards
    precision_regular = list()
    precision_permutation = list()
    recall_regular = list()
    recall_permutation = list()
    F1_regular = list()
    F1_permutation = list()

    # selection containers
    selected_regular = dict(zip(feature_names, np.zeros((len(feature_names),))))
    selected_permutation = dict(zip(feature_names, np.zeros((len(feature_names),))))

    # kcv error containers
    kcv_err_regular = {'tr': list(), 'ts': list()}
    kcv_err_permutation = {'tr': list(), 'ts': list()}

    experiments_folder = os.path.join(base_folder, 'experiments')
    for exp_folder in os.listdir(experiments_folder):
        exp_folder = os.path.join(experiments_folder, exp_folder)
        if os.path.isdir(exp_folder):
            analysis_result = analyze_experiment(
                exp_folder, config, dataset.get_positive_label())

            selected_probesets = feature_names[analysis_result['selected_list']]

            if exp_folder.split('/')[-1].startswith('regular'):
                # update performance metrics
                acc_regular.append(analysis_result['accuracy'])
                balanced_acc_regular.append(analysis_result['balanced_accuracy'])
                MCC_regular.append(analysis_result['MCC'])
                # postiive labels stuff
                precision_regular.append(analysis_result['precision'])
                recall_regular.append(analysis_result['recall'])
                F1_regular.append(analysis_result['F1'])

                # update selection
                for p in selected_probesets:
                    selected_regular[p] += 1
                # update kcv errors
                kcv_err_regular['tr'].append(analysis_result['kcv_err_tr'])
                kcv_err_regular['ts'].append(analysis_result['kcv_err_ts'])

            elif exp_folder.split('/')[-1].startswith('permutation'):
                # update performance metrics
                acc_permutation.append(analysis_result['accuracy'])
                balanced_acc_permutation.append(analysis_result['balanced_accuracy'])
                MCC_permutation.append(analysis_result['MCC'])
                # postiive labels stuff
                precision_permutation.append(analysis_result['precision'])
                recall_permutation.append(analysis_result['recall'])
                F1_permutation.append(analysis_result['F1'])

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
    out = {'v_regular': np.array(balanced_acc_regular),
           'v_permutation': np.array(balanced_acc_permutation),
           'acc_regular': np.array(acc_regular),
           'acc_permutation': np.array(acc_permutation),
           'MCC_regular': np.array(MCC_regular),
           'MCC_permutation': np.array(MCC_permutation),
           'precision_regular': np.array(precision_regular),
           'precision_permutation': np.array(precision_permutation),
           'recall_regular': np.array(recall_regular),
           'recall_permutation': np.array(recall_permutation),
           'F1_regular': np.array(F1_regular),
           'F1_permutation': np.array(F1_permutation),
           'selected_regular': selected_regular,
           'selected_permutation': selected_permutation,
           'kcv_err_regular': kcv_err_regular,
           'kcv_err_permutation': kcv_err_permutation,
           'param_ranges': param_ranges
           }

    return out


def main(base_folder):
    """Main for pd_analysis.py."""
    config = imp.load_source('config', os.path.join(base_folder, 'config.py'))
    _positive_label = config.dataset_options['positive_label']
    _N_jobs_regular = config.N_jobs_regular
    _N_jobs_permutation = config.N_jobs_permutation
    _learner = config.learner_class(None)
    param_names = _learner.param_names

    threshold = int(config.N_jobs_regular * config.frequency_threshold)

    out = analyze_experiments(base_folder, config)

    # balanced acc
    v_regular, v_permutation = out['v_regular'], out['v_permutation']
    acc_regular, acc_permutation = out['acc_regular'], out['acc_permutation']
    MCC_regular, MCC_permutation = out['MCC_regular'], out['MCC_permutation']

    if _positive_label is not None:
        precision_regular = out['precision_regular']
        precision_permutation = out['precision_permutation']

        recall_regular = out['recall_regular']
        recall_permutation = out['recall_permutation']

        F1_regular = out['F1_regular']
        F1_permutation = out['F1_permutation']

    selected_regular = out['selected_regular']
    selected_permutation = out['selected_permutation']

    kcv_err_regular = out['kcv_err_regular']
    kcv_err_permutation = out['kcv_err_permutation']

    param_ranges = out['param_ranges']

    # Manually sorting stuff
    sorted_keys_regular = sorted(selected_regular,
                                 key=selected_regular.__getitem__)
    sorted_keys_permutation = sorted(selected_permutation,
                                     key=selected_permutation.__getitem__)

    # create a new folder for the analysis, called 'analysis'
    base_folder = os.path.join(base_folder, 'analysis')
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # firstly, if exists, copy there the report.txt
    shutil.copy(
        os.path.abspath(os.path.join(base_folder, os.pardir, 'report.txt')),
        base_folder)

    # Dump the selected features in a pkl as pandas DataFrame
    selected_todf = list()
    for k in sorted_keys_regular:
        selected_todf.append(selected_regular[k])
    df_selected = pd.DataFrame(data=selected_todf, index=sorted_keys_regular,
                               columns=['Selection frequency'])
    df_selected.sort_values(['Selection frequency'], ascending=False,
                            inplace=True)
    df_selected.to_pickle(os.path.join(base_folder, 'signature_regular.pkl'))
    # print os.path.join(base_folder, 'signature_regular.pkl')

    with open(os.path.join(base_folder, 'signature_regular.txt'), 'w') as f:
        line_drawn = False
        for k in reversed(sorted_keys_regular):
            if not line_drawn and selected_regular[k] < threshold:
                line_drawn = True
                f.write("=" * 40)
                f.write("\n")
            f.write("{} : {}\n".format(k, selected_regular[k]))

    with open(os.path.join(base_folder, 'signature_permutation.txt'), 'w') as f:
        line_drawn = False
        for k in reversed(sorted_keys_permutation):
            if not line_drawn and selected_permutation[k] < threshold:
                line_drawn = True
                f.write("=" * 40)
                f.write("\n")
            f.write("{} : {}\n".format(k, selected_permutation[k]))

    # Plotting section
    plotting.distributions(acc_regular, acc_permutation, base_folder,
                           'Accuracy', first_run=True)
    plotting.distributions(v_regular, v_permutation, base_folder,
                           'Balanced Accuracy')
    plotting.distributions(MCC_regular, MCC_permutation, base_folder, 'MCC')
    if _positive_label is not None:
        plotting.distributions(precision_regular, precision_permutation,
                               base_folder, 'Precision')
        plotting.distributions(recall_regular, recall_permutation, base_folder,
                               'Recall')
        plotting.distributions(F1_regular, F1_permutation, base_folder, 'F1')

    plotting.feature_frequencies(sorted_keys_regular, selected_regular,
                                 base_folder, threshold=threshold)

    plotting.features_manhattan(sorted_keys_regular, selected_regular,
                                selected_permutation, base_folder,
                                _N_jobs_regular, _N_jobs_permutation,
                                threshold=threshold)

    plotting.selected_over_threshold(selected_regular, selected_permutation,
                                     config.N_jobs_regular,
                                     config.N_jobs_permutation,
                                     base_folder, threshold=threshold)

    for kcv_err, exp in zip([kcv_err_regular, kcv_err_permutation],
                            ['regular', 'permutation']):
        plotting.kcv_err_surfaces(
            kcv_err, exp, base_folder, param_ranges, param_names)


if __name__ == '__main__':
    from palladio import __version__
    parser = argparse.ArgumentParser(description='palladio script for '
                                                 'analysing results.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("result_folder", help="specify results directory")
    args = parser.parse_args()
    main(args.result_folder)
