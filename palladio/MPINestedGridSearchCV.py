"""Nested Cross-Validation for scikit-learn using MPI.

This package provides nested cross-validation similar to scikit-learn's
GridSearchCV but uses the Message Passing Interface (MPI)
for parallel computing.
"""

import logging
import numpy
from mpi4py import MPI
import pandas
from sklearn.base import BaseEstimator
from sklearn.model_selection import check_cv as _check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.base import is_classifier
from sklearn.model_selection._search import _check_param_grid
from sklearn.utils import check_X_y

from palladio.MPIGridSearchCV import (MPIGridSearchCVMaster,
                                      MPIGridSearchCVSlave)


__all__ = ('NestedGridSearchCV')

LOG = logging.getLogger(__package__)

MPI_TAG_RESULT = 3

MPI_MSG_TERMINATE = 0
MPI_MSG_CV = 1
MPI_MSG_TEST = 2
MPI_TAG_TRAIN_TEST_DATA = 5

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def _get_best_parameters(fold_results, param_names):
    """Get best setting of parameters from grid search.

    Parameters
    ----------
    fold_results : pandas.DataFrame
        Contains performance measures as well as hyper-parameters
        as columns. Must contain a column 'fold'.

    param_names : list
        Names of the hyper-parameters. Each name should be a column
        in ``fold_results``.

    Returns
    -------
    max_performance : pandas.Series
        Maximum performance and its hyper-parameters
    """
    if pandas.isnull(fold_results.loc[:, 'score']).all():
        raise ValueError("Results are all NaN")

    # average across inner folds
    grouped = fold_results.drop('fold', axis=1).groupby(param_names)
    mean_performance = grouped.mean()
    # highest average across performance measures
    max_idx = mean_performance.loc[:, 'score'].idxmax()

    # best parameters
    max_performance = pandas.Series({'score':
                                     mean_performance.loc[max_idx, 'score']})
    if len(param_names) == 1:
        key = param_names[0]
        max_performance[key] = max_idx
    else:
        # index has multiple levels
        for i, name in enumerate(mean_performance.index.names):
            max_performance[name] = max_idx[i]

    return max_performance


def _fit_and_score_with_parameters(X, y, cv, best_parameters):
    """Distribute work of non-nested cross-validation across slave nodes."""
    # tell slaves testing phase is next
    _task_desc = numpy.empty(2, dtype=int)
    _task_desc[1] = MPI_MSG_TEST

    comm.Bcast([_task_desc, MPI.INT], root=0)
    comm.bcast((X, y), root=0)

    # Compability with sklearn > 0.18 TODO
    _splitted_cv = [(a, b) for a, b in cv.split(X, y)]

    assert comm_size >= len(_splitted_cv)

    for i, (train_index, test_index) in enumerate(_splitted_cv):
        fold_id = i + 1
        LOG.info("Testing fold %d", fold_id)

        parameters = best_parameters.loc[fold_id, :].to_dict()
        work_item = (fold_id, train_index, test_index, parameters)

        comm.send(work_item, dest=fold_id, tag=MPI_TAG_TRAIN_TEST_DATA)

    scores = {}
    for i in range(len(_splitted_cv)):
        fold_id, test_result = comm.recv(source=MPI.ANY_SOURCE,
                                         tag=MPI_TAG_RESULT)
        scores[fold_id] = test_result

    # Tell all nodes to terminate
    for i in range(len(_splitted_cv), comm_size):
        comm.send((0, None), dest=i, tag=MPI_TAG_TRAIN_TEST_DATA)

    return pandas.Series(scores)


class NestedGridSearchCV(BaseEstimator):
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
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 cv=None, inner_cv=None, multi_output=False):
        self.scoring = scoring
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.fit_params = fit_params
        self.cv = cv
        self.inner_cv = inner_cv
        self.multi_output = multi_output

    def _grid_search(self, train_X, train_y):
        if callable(self.inner_cv):
            # inner_cv = self.inner_cv(train_X, train_y)
            inner_cv = self.inner_cv.split(train_X, train_y)
        else:
            # inner_cv = _check_cv(self.inner_cv, train_X, train_y,
            #                      classifier=is_classifier(self.estimator))
            inner_cv = _check_cv(self.inner_cv, train_y,
                                 classifier=is_classifier(
                                    self.estimator)).split(train_X, train_y)

        master = MPIGridSearchCVMaster(self.param_grid, inner_cv,
                                       self.estimator, self.scorer_,
                                       self.fit_params)
        return master.run(train_X, train_y)

    def _fit_master(self, X, y, cv):
        param_names = list(self.param_grid.keys())

        best_parameters = []
        grid_search_results = []
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            LOG.info("Training fold %d", i + 1)

            train_X = X[train_index, :]
            train_y = y[train_index]

            grid_results = self._grid_search(train_X, train_y)
            grid_search_results.append(grid_results)

            max_performance = _get_best_parameters(grid_results, param_names)
            LOG.info("Best performance for fold %d:\n%s", i + 1,
                     max_performance)
            max_performance['fold'] = i + 1
            best_parameters.append(max_performance)

        best_parameters = pandas.DataFrame(best_parameters)
        best_parameters.set_index('fold', inplace=True)
        best_parameters['score (Test)'] = 0.0
        best_parameters.rename(columns={'score': 'score (Validation)'},
                               inplace=True)

        scores = _fit_and_score_with_parameters(
            X, y, cv, best_parameters.loc[:, param_names])
        best_parameters['score (Test)'] = scores

        self.best_params_ = best_parameters
        self.grid_scores_ = grid_search_results

    def _fit_slave(self):
        slave = MPIGridSearchCVSlave(
            self.estimator, self.scorer_, self.fit_params)
        slave.run()

    def fit(self, X, y):
        """Fit the model to the training data."""
        X, y = check_X_y(X, y, force_all_finite=False,
                         multi_output=self.multi_output)
        _check_param_grid(self.param_grid)

        # cv = _check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))
        cv = _check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if comm_rank == 0:
            self._fit_master(X, y, cv)
        else:
            self._fit_slave()

        return self
