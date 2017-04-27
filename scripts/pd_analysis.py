#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Palladio script for summaries and plot generation."""

import argparse
import gzip
import imp
import os

from six.moves import cPickle as pkl
from six.moves import filter

from palladio.analysis import analyse_results
from palladio.utils import build_cv_results
from palladio.utils import set_module_defaults


def load_results(base_folder):
    """Load pd_run.py results in pickle format.

    Parameters
    ----------
    base_folder : string
        Folder containing ALL experiments (regular and permutations).

    Returns
    -------
    regular_cv_results : dictionary
        As in `palladio.ModelAssessment.cv_results_`
    permutation_cv_results : dictionary
        As in `palladio.ModelAssessment.cv_results_`
    """
    experiments_folder = os.path.join(base_folder, 'experiments')
    pkls = [f for f in os.listdir(experiments_folder) if f.endswith('.pkl')]
    assert(len(pkls) > 0), "no pkl files found in %s" % base_folder

    # Separate regular VS permutation batches
    regular_batch = filter(lambda x: 'regular' in x, pkls)
    permutation_batch = filter(lambda x: 'permutation' in x, pkls)

    # Regular batch
    regular_cv_results = {}  # dictionary of results as in ModelAssessment
    for pkl_file in regular_batch:
        row = pkl.load(gzip.open(os.path.join(experiments_folder, pkl_file), 'rb'))
        build_cv_results(regular_cv_results, **row)

    # Regular batch
    permutation_cv_results = {}  # dictionary of results as in ModelAssessment
    for pkl_file in permutation_batch:
        row = pkl.load(gzip.open(os.path.join(experiments_folder, pkl_file), 'rb'))
        build_cv_results(permutation_cv_results, **row)

    return regular_cv_results, permutation_cv_results


def main():
    """Summary and plot generation."""
    parser = parse_args()
    base_folder = parser.result_folder

    # # Load previously dumped configuration object
    # with gzip.open(os.path.join(base_folder, 'config.py'), 'rb') as f:
    #   config = pkl.load(f)

    # config = imp.load_source('config', os.path.join(base_folder, 'config.py'))

    with gzip.open(os.path.join(base_folder, 'pd_session.pkl.gz'), 'r') as f:
        pd_session_object = pkl.load(f)



    # Load results from pkl
    regular_cv_results, permutation_cv_results = load_results(base_folder)

    # score_surfaces_options = config.score_surfaces_options if hasattr(
    #     config, 'score_surfaces_options') else {}

    # TODO use attributes
    score_surfaces_options = pd_session_object._score_surfaces_options

    if len(set(score_surfaces_options.keys()).difference(set([
            'indep_vars', 'pivoting_var', 'logspace', 'plot_errors']))) > 0:
        raise ValueError("Attribute 'score_surfaces_options' contains "
                         "extra attributes. Values allowed are in "
                         "'indep_vars', 'pivot_var', 'logspace', 'plot_errors'")

    # set_module_defaults(config, {
    #     'feature_names': None,
    #     'learning_task': None,
    #     'vs_analysis': None,
    #     'frequency_threshold': .75,
    #     'ma_options': None,
    #     'analysis_folder': 'analysis'
    # })

    analyse_results(
        regular_cv_results, permutation_cv_results, pd_session_object._labels,
        pd_session_object._estimator,
        base_folder=base_folder,
        feature_names=pd_session_object._feature_names, learning_task=pd_session_object._learning_task,
        vs_analysis=pd_session_object._vs_analysis, threshold=pd_session_object._frequency_threshold,
        model_assessment_options=pd_session_object._ma_options,
        analysis_folder=pd_session_object._analysis_folder,
        score_surfaces_options=score_surfaces_options)


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
