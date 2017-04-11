#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Palladio script for summaries and plot generation."""

import numpy as np
import os

from sklearn.base import is_regressor
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.multiclass import type_of_target
from six import iteritems

from palladio import plotting
from palladio.metrics import __REGRESSION_METRICS__
from palladio.metrics import __CLASSIFICATION_METRICS__
from palladio.metrics import __MULTICLASS_CLASSIFICATION_METRICS__
from palladio.utils import get_selected_list
from palladio.utils import save_signature


def performance_metrics(cv_results, labels, target='regression', metric=None):
    """Evaluate metrics on the external splits results.

    Appropriate metrics are chosen based on the "target" parameter.

    Parameters
    ----------
    cv_results : dictionary
        As in `palladio.ModelAssessment.cv_results_`
    config : module
        Palladio config of the current experiment
    target : ('regression', 'classification', 'multiclass')
        Type of metrics to use.

    Returns
    -------
    performance_metrics : dictionary
        Regression metrics evaluated on the external splits results
    """
    if len(cv_results) < 1:
        return {}

    metrics_ = dict(regression=__REGRESSION_METRICS__,
                    classification=__CLASSIFICATION_METRICS__,
                    multiclass=__MULTICLASS_CLASSIFICATION_METRICS__)
    test_index = cv_results['test_index']
    yts_pred = cv_results['yts_pred']
    yts_true = [labels[i] for i in test_index]

    # Guessing positive label
    pos_label = np.unique(yts_true)[0]

    # Evaluate all the metrics on the results
    performance_metrics_ = {}
    for metric_ in metrics_[target]:
        if metric is not None and metric_.__name__ != metric:
            # restrict the analysis to 'metric'
            continue

        if metric_.__name__ in ['f1_score', 'precision_score', 'recall_score']:
            performance_metrics_[metric_.__name__] = [
                metric_(*yy, pos_label=pos_label) for yy in zip(yts_true, yts_pred)]
        else:
            performance_metrics_[metric_.__name__] = [
                metric_(*yy) for yy in zip(yts_true, yts_pred)]

    if metric is not None and len(performance_metrics_) < 1:
        raise ValueError(
            "No '%s' metric found in available '%s' metrics. "
            "Possible values are %s. Alternatively, "
            "Try to specify another type of target. "
            % (metric, target, [x.__name__ for x in metrics_[target]]))
    return performance_metrics_


def analyse_results(
        regular_cv_results, permutation_cv_results, labels, estimator,
        base_folder=None, analysis_folder='analysis', feature_names=None,
        learning_task=None, vs_analysis=None,
        threshold=.75, model_assessment_options=None,
        score_surfaces_options=None):
    """Summary and plot generation."""
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
        target = 'regression'
    elif learning_task.lower() == 'multiclass':
        target = 'multiclass'
    else:
        # Perform classification analysis
        target = 'classification'

    # Support for empty regular or permutation tests
    performance_regular = performance_metrics(
        regular_cv_results, labels, target)
    performance_permutation = performance_metrics(
        permutation_cv_results, labels, target)
    if base_folder is not None and analysis_folder is not None:
        analysis_folder = os.path.join(base_folder, analysis_folder)
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)
    else:
        analysis_folder = None

    if model_assessment_options is None:
        model_assessment_options = {}
    # Handle variable selection step
    if vs_analysis is not None:
        # Get feature names
        if feature_names is None:
            # what follows creates [feat_0, feat_1, ..., feat_d]
            # feature_names = 'feat_' + np.arange(
            #     labels.size).astype(str).astype(object)
            raise ValueError(
                "Variable selection analysis was specified, but no feature "
                "names were provided.")

        feature_names = np.array(feature_names)  # force feature names to array
        if threshold is None:
            threshold = .75
        selected = {}
        # Init variable selection containers
        selected['regular'] = dict(zip(feature_names,
                                       np.zeros(len(feature_names))))
        selected['permutation'] = selected['regular'].copy()

        n_splits_regular = len((regular_cv_results.values() or [[]])[0])
        n_splits_permutation = len((permutation_cv_results.values() or [[]])[0])
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
                if len(selected_list) < 1:
                    continue
                selected_variables = feature_names[selected_list]

                for var in selected_variables:
                    selected[batch_name][var] += 1. / n_jobs[batch_name]

            # Save selected variables textual summary
            if analysis_folder is not None:
                save_signature(os.path.join(
                    analysis_folder, 'signature_%s.txt' % batch_name),
                    selected[batch_name], threshold)

        # sorted_keys_regular = sorted(
        #     selected['regular'], key=selected['regular'].__getitem__)

        feat_arr_r = np.array(list(iteritems(selected['regular'])), dtype=object)
        feat_arr_p = np.array(list(iteritems(selected['permutation'])), dtype=object)

        # sort by name
        feat_arr_r = feat_arr_r[feat_arr_r[:, 0].argsort()]
        feat_arr_p = feat_arr_p[feat_arr_p[:, 0].argsort()]

        # Save graphical summary
        plotting.feature_frequencies(
            feat_arr_r, analysis_folder,
            threshold=threshold)

        plotting.features_manhattan(
            feat_arr_r, feat_arr_p, analysis_folder,
            threshold=threshold)

        plotting.select_over_threshold(
            feat_arr_r, feat_arr_p, analysis_folder,
            threshold=threshold)

    # Generate distribution plots
    for i, metric in enumerate(performance_regular):
        plotting.distributions(
            v_regular=performance_regular[metric],
            v_permutation=performance_permutation.get(metric, []),
            base_folder=analysis_folder,
            metric=metric,
            first_run=i == 0,
            is_regression=is_regression)

    # Generate surfaces
    # This has meaning only if the estimator is an istance of GridSearchCV
    if isinstance(estimator, BaseSearchCV):
        if score_surfaces_options is None:
            score_surfaces_options = {}
        plotting.score_surfaces(
            param_grid=estimator.param_grid,
            results=regular_cv_results,
            base_folder=analysis_folder,
            is_regression=is_regression,
            **score_surfaces_options)
