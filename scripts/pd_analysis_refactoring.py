#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Palladio script for summaries and plot generation."""

import argparse
import imp
import os
import pandas as pd

from six.moves import cPickle as pkl
from sklearn.base import is_regressor
from sklearn.utils.multiclass import type_of_target

from palladio.metrics import __REGRESSION_METRICS__
from palladio.utils import build_cv_results


def regression_analysis(cv_results, config):
    """."""
    test_index = cv_results['test_index']
    yts_pred = cv_results['yts_pred']
    yts_true = [config.labels[i] for i in test_index]

    # Evaluate all the metrics on the results



    return 0


def classification_analysis(cv_results):
    return 0


def load_results(base_folder):
    """Load pd_run.py results in pickle format.

    Parameters
    ----------
    base_folder : string
        Folder containing ALL experiments (regular and permutations).

    Returns
    -------
    cv_results : dictionary
        As in `palladio.ModelAssessment.cv_results_`
    """
    experiments_folder = os.path.join(base_folder, 'experiments')
    pkls = [f for f in os.listdir(experiments_folder) if f.endswith('.pkl')]
    assert(len(pkls) > 0), "no pkl files found in %s" % base_folder

    cv_results = {}  # dictionary of results as in ModelAssessment
    for pkl_file in pkls:
        row = pkl.load(open(os.path.join(experiments_folder, pkl_file), 'rb'))
        build_cv_results(cv_results, **row)

    return cv_results


def main():
    """Summary and plot generation."""
    parser = parse_args()
    base_folder = parser.result_folder
    config = imp.load_source('config', os.path.join(base_folder, 'config.py'))

    # Load results from pkl
    cv_results = load_results(base_folder)

    # learning_task follows the convention of
    # sklearn.utils.multiclass.type_of_target
    learning_task = config.learning_task if hasattr(config, 'learning_task') \
        else None
    if learning_task is None:
        if is_regressor(config.estimator):
            learning_task = 'continuous'
        else:
            learning_task = type_of_target(config.labels)

    # Run the appropriate analysis according to the learning_task
    if learning_task.lower() in ['continuous', 'regression']:
        # Perform regression analysis
        out = regression_analysis(cv_results, config)
    else:
        # Perform classification analysis
        pass


def parse_args():
    """Parse arguments."""
    from palladio import __version__
    parser = argparse.ArgumentParser(
        description='palladio script for analysing results.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-o", "--output", dest="output_folder", action="store",
                        help="specify a name for the analysis folder",
                        default='analysis')
    parser.add_argument("result_folder", help="specify results directory")
    return parser.parse_args()


if __name__ == '__main__':
    main()
