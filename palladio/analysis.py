#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Palladio script for summaries and plot generation."""

import numpy as np
import os

from sklearn.base import is_regressor
from sklearn.utils.multiclass import type_of_target

from palladio import plotting
from palladio.metrics import __REGRESSION_METRICS__
from palladio.metrics import __CLASSIFICATION_METRICS__
from palladio.utils import get_selected_list
from palladio.utils import save_signature


def regression_analysis(cv_results, labels):
    """Evaluate the regression metrics on the external splits results.

    Parameters
    ----------
    cv_results : dictionary
        As in `palladio.ModelAssessment.cv_results_`
    config : module
        Palladio config of the current experiment

    Returns
    -------
    performance_metrics : dictionary
        Regression metrics evaluated on the external splits results
    """
    test_index = cv_results['test_index']
    yts_pred = cv_results['yts_pred']
    yts_true = [labels[i] for i in test_index]

    # Evaluate all the metrics on the results
    performance_metrics = {}
    for metric in __REGRESSION_METRICS__:
        performance_metrics[metric.__name__] = [
            metric(*yy) for yy in zip(yts_true, yts_pred)]

    return performance_metrics


def classification_analysis(cv_results, labels):
    """Evaluate the classification metrics on the external splits results.

    Parameters
    ----------
    cv_results : dictionary
        As in `palladio.ModelAssessment.cv_results_`
    config : module
        Palladio config of the current experiment

    Returns
    -------
    performance_metrics : dictionary
        Regression metrics evaluated on the external splits results
    """
    test_index = cv_results['test_index']
    yts_pred = cv_results['yts_pred']
    yts_true = [labels[i] for i in test_index]

    # Evaluate all the metrics on the results
    performance_metrics = {}
    for metric in __CLASSIFICATION_METRICS__:
        performance_metrics[metric.__name__] = [
            metric(*yy) for yy in zip(yts_true, yts_pred)]

    return performance_metrics


def analyse_results(
    regular_cv_results, permutation_cv_results, labels, estimator,
    n_splits_regular, n_splits_permutation,
    base_folder, feature_names=None, learning_task=None, vs_analysis=None,
        threshold=.75, model_assessment_options=None,
        score_surfaces_options=None):
    """Summary and plot generation."""
    # Get feature names
    if feature_names is None:
        # what follows creates [feat_0, feat_1, ..., feat_d]
        feature_names = 'feat_' + np.arange(
            labels.size).astype(str).astype(object)

    # learning_task follows the convention of
    # sklearn.utils.multiclass.type_of_target
    if learning_task is None:
        if is_regressor(estimator):
            learning_task = 'continuous'
        else:
            learning_task = type_of_target(labels)

    # Run the appropriate analysis according to the learning_task
    is_regression = learning_task.lower() in ('continuous', 'regression')
    if is_regression:
        # Perform regression analysis
        performance_regular = regression_analysis(regular_cv_results, labels)
        performance_permutation = {}  # for consistency only
    else:
        # Perform classification analysis
        performance_regular = classification_analysis(
            regular_cv_results, labels)
        performance_permutation = classification_analysis(
            permutation_cv_results, labels)

    if model_assessment_options is None:
        model_assessment_options = {}
    # Handle variable selection step
    if vs_analysis is not None:
        if threshold is None:
            threshold = .75
        selected = {}
        # Init variable selection containers
        selected['regular'] = dict(zip(feature_names,
                                       np.zeros(len(feature_names))))
        selected['permutation'] = selected['regular'].copy()

        n_jobs = {'regular': n_splits_regular,
                  'permutation': n_splits_permutation}
        names_ = ('regular', 'permutation')
        cv_results_ = (regular_cv_results, permutation_cv_results)
        for batch_name, cv_result in zip(names_, cv_results_):
            # cv_result['estimator'] is a list containing
            # the grid-search estimators
            estimators = cv_result.get('estimator', None)
            if estimators is None:
                continue  # in case of no permutations skip this iter
            for estimator in estimators:
                selected_list = get_selected_list(
                    estimator, vs_analysis)
                selected_variables = feature_names[selected_list]

                for var in selected_variables:
                    selected[batch_name][var] += 1. / n_jobs[batch_name]

            # Save selected variables textual summary
            filename = os.path.join(
                base_folder, 'signature_%s.txt' % batch_name)
            save_signature(filename, selected[batch_name], threshold)

        sorted_keys_regular = sorted(
            selected['regular'], key=selected['regular'].__getitem__)

        # # Save graphical summary
        plotting.feature_frequencies(
            sorted_keys_regular, selected['regular'], base_folder,
            threshold=threshold)

        plotting.features_manhattan(
            sorted_keys_regular, selected['regular'],
            selected['permutation'], base_folder,
            threshold=threshold)

        plotting.selected_over_threshold(
            selected['regular'], selected['permutation'],
            base_folder, threshold=threshold)

    # Generate distribution plots
    first_run = True
    for metric in performance_regular:
        plotting.distributions(
            v_regular=performance_regular[metric],
            v_permutation=performance_permutation.get(metric, []),
            base_folder=base_folder, metric=metric,
            first_run=first_run,
            is_regression=is_regression)
        if first_run:
            first_run = False

    # Generate surfaces
    # This has meaning only if the estimator is an istance of GridSearchCV
    from sklearn.model_selection._search import BaseSearchCV
    if isinstance(estimator, BaseSearchCV):
        plotting.score_surfaces(
            param_grid=estimator.param_grid,
            results=regular_cv_results,
            base_folder=base_folder,
            is_regression=is_regression,
            **score_surfaces_options)
