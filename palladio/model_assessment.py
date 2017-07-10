"""Nested Cross-Validation for scikit-learn using MPI.

This package provides nested cross-validation similar to scikit-learn's
GridSearchCV but uses the Message Passing Interface (MPI)
for parallel computing.
"""
from __future__ import print_function

import errno
import gzip
import logging
import joblib as jl
import numbers
import os
import warnings

from collections import deque
from collections import Iterable
from six.moves import cPickle as pkl
from sklearn.base import BaseEstimator, clone
# from sklearn.model_selection import check_cv as _check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.base import is_classifier
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.model_selection._validation import _shuffle
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils import check_X_y, check_random_state
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import type_of_target

from palladio.utils import build_cv_results as _build_cv_results

__all__ = ('ModelAssessment',)

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()
    IS_MPI_JOB = COMM.Get_size() > 1

except ImportError:
    # warnings.warn("mpi4py module not found. "
    #               "PALLADIO cannot run on multiple machines.")
    COMM = None
    RANK = 0
    NAME = 'localhost'
    IS_MPI_JOB = False

MAX_RESUBMISSIONS = 0  # resubmissions disabled
DO_WORK = 100
EXIT = 200


def assert_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def _worker(estimator_, i, X, y, train, test):
    """Implement the worker resubmission in case of errors."""
    # custom_name = "{}_p_{}_i_{}".format(
    #     ("permutation" if is_permutation_test else "regular"), RANK, i)
    # tmp_name_base = 'tmp_' + custom_name

    worker_logger = logging.getLogger('worker')

    experiment_resubmissions = 0
    experiment_completed = False

    worker_logger.info("{}{} executing job {}".format(NAME, RANK, i))

    while not experiment_completed and \
            experiment_resubmissions <= MAX_RESUBMISSIONS:
        try:

            if experiment_resubmissions > 0:
                worker_logger.warning("{}{} resubmitting experiment {}".format(NAME, RANK, i))

            # tmp_name = tmp_name_base + '_submission_{}'.format(
            #     experiment_resubmissions + 1)
            # run_experiment(data, labels, None, config,
            #                is_permutation_test, experiments_folder_path,
            #                tmp_name)
            # TODO necessary?
            estimator = clone(estimator_.estimator)

            # need to get the deepest estimator to use _safe_split
            estimator__ = clone(estimator)
            while hasattr(estimator__, 'estimator'):
                estimator__ = clone(estimator__.estimator)

            X_train, y_train = _safe_split(estimator__, X, y, train)
            X_test, y_test = _safe_split(estimator__, X, y, test, train)

            if estimator_.shuffle_y:
                random_state = check_random_state(estimator_.random_state)
                y_train = _shuffle(y_train, estimator_.groups, random_state)

            worker_logger.info("{}{} fitting experiment {} - starting".format(NAME, RANK, i))
            estimator.fit(X_train, y_train)
            worker_logger.info("{}{} fitting experiment {} - completed".format(NAME, RANK, i))

            worker_logger.debug("{}{} scoring experiment {} - starting".format(NAME, RANK, i))
            yts_pred = estimator.predict(X_test)
            ytr_pred = estimator.predict(X_train)
            lr_score = estimator_.scorer_(estimator, X_train, y_train)
            ts_score = estimator_.scorer_(estimator, X_test, y_test)
            worker_logger.debug("{}{} scoring experiment {} - complete".format(NAME, RANK, i))

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
                'test_index': test,
                'train_index': train,
                'estimator': estimator
            }

            experiment_completed = True

            # ### Dump partial results
            if estimator_.experiments_folder is not None:
                worker_logger.debug("{}{} saving results for experiment {}".format(NAME, RANK, i))
                pkl_name = (
                    'permutation' if estimator_.shuffle_y else 'regular') + \
                    '_%d.pkl' % i

                pkl.dump(cv_results_, gzip.open(os.path.join(
                    estimator_.experiments_folder, pkl_name), 'wb'))

        except StandardError as error:
            # If somethings out of the ordinary happens,
            # resubmit the job
            experiment_resubmissions += 1
            warnings.warn(
                "[{}_{}] failed experiment {}, resubmission #{}\n"
                "Exception raised: {}".format(
                    NAME, RANK, i, experiment_resubmissions, error))

    if not experiment_completed:
        warnings.warn(
            "[{}_{}] failed to complete experiment {}, "
            "max resubmissions limit reached".format(NAME, RANK, i))
        return {}
    else:
        if not IS_MPI_JOB and estimator_.verbose:

            worker_logger.info("[{}{}]: {} job {} completed".format(NAME, RANK, ('permutation' if estimator_.shuffle_y else 'regular'), i))

        return cv_results_


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

    cv : integer or cross-validation generator, optional, default: 3
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects


    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        See sklearn.metrics.get_scorer for details.

    fit_params : dict, optional, default: None
        Parameters to pass to the fit method.

    multi_output : boolean, default: False
        Allow multi-output y, as for multivariate regression.

    shuffle_y : bool, optional, default=False
        When True, the object is used to perform permutation test.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the Monte Carlo runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. Ignored when using MPI.

     n_splits: int, optional, default: 10
        The number of cross-validation splits (folds/iterations).

    test_size : float (default 0.1), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

     random_state : int or RandomState, optional, default: None
        Pseudo-random number generator state used for random sampling.

     groups : array-like, with shape (n_samples,), optional, default: None
            Group labels for the samples used while splitting the dataset into
            train/test set.

    experiments_folder : string, optional, default: None
        The path to the folder used to save the results.

    verbose : bool, optional, default: False
        Print debug messages.

    Attributes
    ----------
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
                 multi_output=False, shuffle_y=False, n_jobs=1,
                 n_splits=10, test_size=0.1, train_size=None,
                 random_state=None, groups=None, experiments_folder=None,
                 verbose=False):
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
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Shuffle training labels
        self.shuffle_y = shuffle_y

    def _fit_single_job(self, job_list, X, y):
        cv_results_ = {}
        # for i, (train_index, test_index) in job_list:
        #     LOG.info("Training fold %d", i + 1)
        #
        #     slave_result_ = self._worker(
        #         i, X, y, train_index, test_index)
        #
        #     _build_cv_results(cv_results_, **slave_result_)
        slave_results = jl.Parallel(n_jobs=self.n_jobs) \
            (jl.delayed(_worker)(
                self, i, X, y, train_index, test_index) for i, (
                    train_index, test_index) in job_list)
        for slave_result_ in slave_results:
            _build_cv_results(cv_results_, **slave_result_)

        self.cv_results_ = cv_results_

    def _fit_master(self, X, y):

        master_logger = logging.getLogger('master')

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
            master_logger.debug("Submitting job to process #{}".format(rankk))
            COMM.send(pipe_tuple, dest=rankk, tag=DO_WORK)

        while queue:
            pipe_tuple = queue.popleft()
            # receive result from slave
            status = MPI.Status()
            slave_result_ = COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            master_logger.debug("Received results from process #{}, submitting another".format(status.source))

            # send to the same slave new work
            COMM.send(pipe_tuple, dest=status.source, tag=DO_WORK)

            _build_cv_results(cv_results_, **slave_result_)
            count += 1

        # No more work to do, so receive all the results from slaves
        for rankk in range(1, min(nprocs, n_pipes)):
            status = MPI.Status()
            slave_result_ = COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            master_logger.debug("Received results from process #{}".format(status.source))

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
                # if self.verbose:
                #     print("[{} {}]: Performing experiment {}".format(
                #         NAME, RANK, i))

                cv_results_ = _worker(self, i, X, y, train_index, test_index)
                # if self.verbose:
                #     print("[{} {}]: Experiment {} completed".format(
                #         NAME, RANK, i))
                COMM.send(cv_results_, dest=0, tag=0)

        except StandardError as exc:
            warnings.warn("Quitting ... TB:", str(exc))

    def fit(self, X, y):
        """Fit the model to the training data."""
        X, y = check_X_y(X, y, force_all_finite=False,
                         multi_output=self.multi_output)

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if RANK == 0:
            if self.experiments_folder is not None:
                assert_path(self.experiments_folder)

            self._fit_master(X, y)
        else:
            self._fit_slave(X, y)

        return self
