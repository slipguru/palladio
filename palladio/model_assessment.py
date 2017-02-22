"""Nested Cross-Validation for scikit-learn using MPI.

This package provides nested cross-validation similar to scikit-learn's
GridSearchCV but uses the Message Passing Interface (MPI)
for parallel computing.
"""
from __future__ import print_function

import logging
# import numpy
import numbers
# import pandas
import os
import warnings

try:
    import cPickle as pkl
except ImportError:  # python 3 compatibility
    import pickle as pkl

from collections import deque
from collections import Iterable
from sklearn.base import BaseEstimator, clone
# from sklearn.model_selection import check_cv as _check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.base import is_classifier
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.model_selection._validation import _shuffle
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils import check_X_y, check_random_state
from sklearn.utils.multiclass import type_of_target

from palladio.utils import build_cv_results as _build_cv_results

__all__ = ('ModelAssessment',)

LOG = logging.getLogger(__package__)

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()
    IS_MPI_JOB = COMM.Get_size() > 1

except ImportError:
    warnings.warn("mpi4py module not found. "
                  "PALLADIO cannot run on multiple machines.")
    COMM = None
    RANK = 0
    NAME = 'localhost'
    IS_MPI_JOB = False

MAX_RESUBMISSIONS = 0  # resubmissions disabled
DO_WORK = 100
EXIT = 200


def _check_cv(cv=3, y=None, classifier=False, **kwargs):
    """Input checker utility for building a cross-validator.

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    y : array-like, optional
        The target variable for supervised learning problems.

    classifier : boolean, optional, default False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    kwargs : dict
        Other parameters for StratifiedShuffleSplit or ShuffleSplit.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """
    if cv is None:
        cv = kwargs.pop('n_splits', 0) or 10

    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return StratifiedShuffleSplit(cv, **kwargs)
        else:
            return ShuffleSplit(cv, **kwargs)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


class ModelAssessment(BaseEstimator):
    """Cross-validation with nested parameter search for each training fold.

    The data is first split into ``cv`` train and test sets. For each training
    set a grid search over the specified set of parameters is performed
    (inner cross-validation). The set of parameters that achieved the highest
    average score across all inner folds is used to re-fit a model on the
    entire training set of the outer cross-validation loop. Finally, results on
    the test set of the outer loop are reported.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        See sklearn.metrics.get_scorer for details.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    cv : integer or cross-validation generator, default=3
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    inner_cv : integer or callable, default=3
        If an integer is passed, it is the number of folds.
        If callable, the function must have the signature
        ``inner_cv_func(X, y)`` and return a cross-validation object,
        see sklearn.model_selection module for the list of possible objects.

    multi_output : boolean, default=False
        Allow multi-output y, as for multivariate regression.

    Attributes
    ----------
    best_params_ : pandas.DataFrame
        Contains selected parameter settings for each fold.
        The validation score refers to average score across all folds of the
        inner cross-validation, the test score to the score on the test set
        of the outer cross-validation loop.

    grid_scores_ : list of pandas.DataFrame
        Contains full results of grid search for each training set of the
        outer cross-validation loop.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    cv_results_ : dictionary
        Result of the fit. The dictionary is pandas.DataFrame-able. Each row is
        the results of an external split.
        Columns are:
        'split_i', 'learn_score', 'test_score', 'cv_results_', 'ytr_pred',
        'yts_pred', 'test_index', 'train_index', 'estimator'

        Example:
        >>> pd.DataFrame(cv_results_)
        split_i | learn_score | test_score | cv_results_         | ...
              0 |       0.987 |      0.876 | {<internal splits>} | ...
              1 |       0.846 |      0.739 | {<internal splits>} | ...
              2 |       0.956 |      0.630 | {<internal splits>} | ...
              3 |       0.964 |      0.835 | {<internal splits>} | ...
    """

    def __init__(self, estimator, cv=None, scoring=None, fit_params=None,
                 multi_output=False, shuffle_y=False,
                 n_splits=10, test_size=0.1, train_size=None,
                 random_state=None, groups=None, experiments_folder=None):
        self.estimator = estimator
        self.scoring = scoring
        self.fit_params = fit_params
        self.cv = cv
        self.multi_output = multi_output
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.groups = groups
        self.experiments_folder = experiments_folder

        # Shuffle training labels
        self.shuffle_y = shuffle_y

    def _fit_single_job(self, job_list, X, y):
        cv_results_ = {}
        for i, (train_index, test_index) in job_list:
            LOG.info("Training fold %d", i + 1)

            slave_result_ = self._worker(
                i, X, y, train_index, test_index)

            _build_cv_results(cv_results_, **slave_result_)

        self.cv_results_ = cv_results_

    def _fit_master(self, X, y):
        cv = _check_cv(
            self.cv, y, classifier=is_classifier(self.estimator),
            n_splits=self.n_splits, test_size=self.test_size,
            train_size=self.train_size, random_state=self.random_state)

        job_list = list(enumerate(cv.split(X, y, self.groups)))
        if not IS_MPI_JOB:
            self._fit_single_job(
                job_list, X, y)
            return

        count = 0
        nprocs = COMM.Get_size()
        queue = deque(job_list)
        n_pipes = len(queue)
        cv_results_ = {}  # updated by _build_cv_results

        # seed the slaves by sending work to each processor
        for rankk in range(1, min(nprocs, n_pipes)):
            pipe_tuple = queue.popleft()
            COMM.send(pipe_tuple, dest=rankk, tag=DO_WORK)

        while queue:
            pipe_tuple = queue.popleft()
            # receive result from slave
            status = MPI.Status()
            slave_result_ = COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            # send to the same slave new work
            COMM.send(pipe_tuple, dest=status.source, tag=DO_WORK)

            _build_cv_results(cv_results_, **slave_result_)
            count += 1

        # No more work to do, so receive all the results from slaves
        for rankk in range(1, min(nprocs, n_pipes)):
            status = MPI.Status()
            slave_result_ = COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            _build_cv_results(cv_results_, **slave_result_)
            count += 1

        # tell all slaves to exit by sending an empty message with the EXIT_TAG
        for rankk in range(1, nprocs):
            COMM.send(0, dest=rankk, tag=EXIT)

            # max_performance = _get_best_parameters(grid_results, param_names)
            # LOG.info("Best performance for fold %d:\n%s", i + 1,
            #          max_performance)
            # max_performance['fold'] = i + 1
            # best_parameters.append(max_performance)

        # best_parameters = pandas.DataFrame(best_parameters)
        # best_parameters.set_index('fold', inplace=True)
        # best_parameters['score (Test)'] = 0.0
        # best_parameters.rename(columns={'score': 'score (Validation)'},
        #                        inplace=True)

        # scores = _fit_and_score_with_parameters(
        #     X, y, cv, best_parameters.loc[:, param_names])
        # best_parameters['score (Test)'] = scores

        # self.best_params_ = best_parameters
        self.cv_results_ = cv_results_

    def _fit_slave(self, X, y):
        """Pipeline evaluation.

        Parameters
        ----------
        X : array of float, shape : n_samples x n_features, default : ()
            The input data matrix.
        """
        try:
            while True:
                status_ = MPI.Status()
                received = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status_)

                # check the tag of the received message
                if status_.tag == EXIT:
                    return
                # do the work
                i, (train_index, test_index) = received

                print("[{} {}]: Performing experiment {}".format(NAME, RANK, i))

                cv_results_ = self._worker(i, X, y, train_index, test_index)
                print("[{} {}]: Experiment {} completed".format(NAME, RANK, i))
                COMM.send(cv_results_, dest=0, tag=0)

        except StandardError as exc:
            warnings.warn("Quitting ... TB:", str(exc))

    def _worker(self, i, X, y, train_index, test_index):
        """Implement the worker resubmission in case of errors."""
        # custom_name = "{}_p_{}_i_{}".format(
        #     ("permutation" if is_permutation_test else "regular"), RANK, i)
        # tmp_name_base = 'tmp_' + custom_name
        experiment_resubmissions = 0
        experiment_completed = False
        while not experiment_completed and \
                experiment_resubmissions <= MAX_RESUBMISSIONS:
            try:
                # tmp_name = tmp_name_base + '_submission_{}'.format(
                #     experiment_resubmissions + 1)
                # run_experiment(data, labels, None, config,
                #                is_permutation_test, experiments_folder_path,
                #                tmp_name)
                Xtr = X[train_index, :]
                Xts = X[test_index, :]
                ytr = y[train_index]
                yts = y[test_index]

                # TODO necessary?
                estimator = clone(self.estimator)
                if self.shuffle_y:
                    random_state = check_random_state(self.random_state)
                    ytr = _shuffle(ytr, self.groups, random_state)
                estimator.fit(Xtr, ytr)

                yts_pred = estimator.predict(Xts)
                ytr_pred = estimator.predict(Xtr)
                lr_score = self.scorer_(estimator, Xtr, ytr)
                ts_score = self.scorer_(estimator, Xts, yts)
                # lr_score = estimator.score(Xtr, ytr)
                # ts_score = estimator.score(Xts, yts)

                if hasattr(estimator, 'cv_results_'):
                    # In case in which the estimator is a CV object
                    cv_results = estimator.cv_results_
                else:
                    cv_results = None

                cv_results_ = {
                    'split_i': i,
                    'learn_score': lr_score,
                    'test_score': ts_score,
                    'cv_results_': cv_results,
                    'ytr_pred': ytr_pred,
                    'yts_pred': yts_pred,
                    'test_index': test_index,
                    'train_index': train_index,
                    'estimator': estimator
                }

                experiment_completed = True

                # ### Dump partial results
                if self.experiments_folder is not None:
                    pkl_name = (
                        'permutation' if self.shuffle_y else 'regular') + \
                        '_%d.pkl' % i

                    # TODO use gzip?
                    with open(os.path.join(
                            self.experiments_folder, pkl_name), 'wb') as ff:
                        # pkl.dump(partial_result, ff)
                        pkl.dump(cv_results_, ff)

            except StandardError as e:
                # If somethings out of the ordinary happens,
                # resubmit the job
                experiment_resubmissions += 1
                warnings.warn(
                    "[{}_{}] failed experiment {}, resubmission #{}\n"
                    "Exception raised: {}".format(
                        NAME, RANK, i, experiment_resubmissions, e))

        if not experiment_completed:
            warnings.warn(
                "[{}_{}] failed to complete experiment {}, "
                "max resubmissions limit reached".format(NAME, RANK, i))
            return {}
        else:
            return cv_results_

    def fit(self, X, y):
        """Fit the model to the training data."""
        X, y = check_X_y(X, y, force_all_finite=False,
                         multi_output=self.multi_output)

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if RANK == 0:
            self._fit_master(X, y)
        else:
            self._fit_slave(X, y)

        return self
