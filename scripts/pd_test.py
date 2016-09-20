#!/usr/bin/env python
"""Test the obtained palladio model on an independent test set."""

import matplotlib; matplotlib.use('Agg')

import os
import sys
import imp
import argparse
import numpy as np
import pandas as pd
import cPickle as pkl
import seaborn as sns
from palladio import utils

# TODO: add more learning machines
__learners__ = ['ols', 'leastquares', 'ls', 'rls', 'tikhonov', 'ridge']


def main(args):
    """Load the signature and perform incremental tests."""
    root = args.result_folder
    analysis_path = os.path.join(root, 'analysis')

    # Load the config file
    config = imp.load_source('config', os.path.join(root, 'config.py'))

    # Load the training data
    tr_data = pd.read_csv(os.path.join(root, 'data'),
                          header=0, index_col=0)
    print("* Training data loaded: {}x{}".format(*tr_data.shape))
    tr_labels = pd.read_csv(os.path.join(root, 'labels'),
                            header=0, index_col=0)
    print("* Training labels loaded: {}x{}".format(*tr_labels.shape))

    # Load the test data
    # WORKAROUND: TODO use the same loading facilities as palladio
    ts_data = pd.read_csv(args.data, header=0, index_col=0)
    print("* Test data loaded: {}x{}".format(*ts_data.shape))
    ts_labels = pd.read_csv(args.labels, header=0, index_col=0)
    print("* Test labels loaded: {}x{}".format(*ts_labels.shape))

    # From string to +1/-1
    poslab = config.dataset_options['positive_label']
    if poslab is None:
        # if no positive labe is specified, pick the first one
        poslab = tr_labels.values[0]
    topm = lambda x: +1.0 if x == poslab else -1.0
    tr_labels = np.array(map(topm, tr_labels.values))
    ts_labels = np.array(map(topm, ts_labels.values))
    print("* Positive label: {}".format(poslab))

    # Load the signature
    with open(os.path.join(analysis_path, 'signature_regular.pkl'), 'r') as f:
        signature = pkl.load(f)

    # Define the list of thresholds
    var_names = signature.index
    thresh = np.unique(signature.values)
    print("* Creating {} test sets with increasing "
          "dimensions.".format(len(thresh)))

    # Define the learner class
    if args.learner.lower() in ['rls', 'tikhonov', 'ridge']:
        # from palladio.wrappers.extensions import RLSCV
        # model = RLSCV()
        pass
    elif args.learner.lower() in ['ols', 'leastquares', 'ls']:
        import l1l2py
        model = lambda x, y: l1l2py.algorithms.ridge_regression(x, y, mu=0)
    else:
        print("Only least squares methods implemented so far.")
        sys.exit(-1)
    print("* Using {} learning algorithm".format(args.learner))

    # Build incremental test sets
    ts_err = list()
    beta_list = list()
    for t in thresh:
        idx = np.where(signature.values >= t)[0]
        selected = var_names[idx]
        X_tr = tr_data[selected].as_matrix()
        y_tr = tr_labels
        X_ts = ts_data[selected].as_matrix()
        y_ts = ts_labels

        # Training
        beta = model(X_tr, y_tr)
        beta_list.append(beta)

        # Test
        y_pred = np.sign(np.dot(X_ts, beta)).ravel()

        # Test error assessment
        confusion_matrix = utils.confusion_matrix(y_ts, y_pred)
        summary = utils.classification_measures(confusion_matrix)
        ts_err.append(summary['balanced_accuracy'])

    # Get the optimal threshold
    opt_thresh = thresh[np.where(ts_err == np.max(ts_err))[0][-1]]
    print("* The suggested selection frequency "
          "threshold is {}%".format(int(opt_thresh)))

    # Store the optimal model
    opt_beta = beta_list[np.where(ts_err == np.max(ts_err))[0][-1]]
    with open(os.path.join(analysis_path, 'beta.pkl'),'w') as f:
        pkl.dump(opt_beta, f)


    # Make plot
    sns.plt.plot(thresh, ts_err, '-o', label='Balanced accuracy')
    sns.plt.axvline(opt_thresh, linestyle='dashed',
                    label='suggested threshold: '+str(int(opt_thresh))+'%',
                    color=sns.xkcd_rgb['dark brown'])
    sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
    sns.plt.xlabel('Selection frequency threshold (%)')
    sns.plt.ylabel('Score')
    sns.plt.ylim([np.min(ts_err)-0.01, np.max(ts_err)+0.01])
    sns.plt.savefig(os.path.join(analysis_path, 'test_errors.png'))

    # Save selected variables
    idx = np.where(signature.values >= opt_thresh)[0]
    selected = var_names[idx]
    df_selected = pd.DataFrame(data=signature.values[idx], index=selected,
                               columns=signature.columns)
    df_selected.to_csv(os.path.join(analysis_path,
                       'suggestion_selected.txt'))
    df_selected.to_pickle(os.path.join(analysis_path,
                          'suggestion_selected.pkl'))

######################################################################

if __name__ == '__main__':
    from palladio import __version__
    parser = argparse.ArgumentParser(description='palladio script for '
                                                 'model testing on an '
                                                 'independent test set.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("result_folder", help="specify results directory")
    parser.add_argument('--data', action="store",
                        dest="data", type=str, help='path to your test data')
    parser.add_argument('--labels', action="store",
                        dest="labels", type=str,
                        help='path to your test labels')
    parser.add_argument('--learner', action="store",
                        dest="learner", type=str,
                        default='OLS',
                        help='the learner algorithm')
    args = parser.parse_args()

    if not args.data:
        sys.stderr.write("Missing test data file.\n")
        sys.exit(-1)
    if not args.labels:
        sys.stderr.write("Missing test labels file.\n")
        sys.exit(-1)
    if args.learner.lower() not in __learners__:
        sys.stderr.write("Unknown learning "
                         "algorithm {}.\n".format(args.learner))
        sys.exit(-1)

    main(args)
