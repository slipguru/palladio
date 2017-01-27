"""Nested Cross-Validation for scikit-learn using MPI.

This package provides nested cross-validation similar to scikit-learn's
GridSearchCV but uses the Message Passing Interface (MPI)
for parallel computing.
"""

import logging
import numpy
from mpi4py import MPI
import pandas
from sklearn.base import clone
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import ParameterGrid

LOG = logging.getLogger(__package__)

MPI_TAG_RESULT = 3

MPI_MSG_TERMINATE = 0
MPI_MSG_CV = 1
MPI_MSG_TEST = 2
MPI_TAG_TRAIN_TEST_DATA = 5

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


class MPIBatchWorker(object):
    """Base class to fit and score an estimator."""

    def __init__(self, estimator, scorer, fit_params, verbose=False):
        self.estimator = estimator
        self.scorer = scorer
        self.verbose = verbose
        self.fit_params = fit_params

        # first item denotes ID of fold, and second item encodes
        # a message that tells slaves what to do
        self._task_desc = numpy.empty(2, dtype=int)
        # stores data that the root node broadcasts
        self._data_X = None
        self._data_y = None

    def process_batch(self, work_batch):
        fit_params = self.fit_params if self.fit_params is not None else {}

        LOG.debug("Node %d received %d work items", comm_rank, len(work_batch))

        results = []
        for fold_id, train_index, test_index, parameters in work_batch:
            ret = _fit_and_score(clone(self.estimator),
                                 self._data_X, self._data_y,
                                 self.scorer, train_index, test_index,
                                 self.verbose, parameters, fit_params,
                                 return_n_test_samples=True,
                                 return_times=True)

            result = parameters.copy()
            result['score'] = ret[0]
            result['n_samples_test'] = ret[1]
            result['scoring_time'] = ret[2]
            result['fold'] = fold_id
            results.append(result)

        LOG.debug("Node %d is done with fold %d", comm_rank, fold_id)
        return results


class MPIGridSearchCVSlave(MPIBatchWorker):
    """Receives task from root node and sends results back."""

    def __init__(self, estimator, scorer, fit_params):
        super(MPIGridSearchCVSlave, self).__init__(estimator, scorer, fit_params)

    def _run_grid_search(self):
        # get data
        self._data_X, self._data_y = comm.bcast(None, root=0)
        # get batch
        work_batch = comm.scatter(None, root=0)

        results = self.process_batch(work_batch)
        # send result
        comm.gather(results, root=0)

    def _run_train_test(self):
        # get data
        self._data_X, self._data_y = comm.bcast(None, root=0)

        work_item = comm.recv(None, source=0, tag=MPI_TAG_TRAIN_TEST_DATA)
        fold_id = work_item[0]
        if fold_id == MPI_MSG_TERMINATE:
            return

        LOG.debug("Node %d is running testing for fold %d", comm_rank, fold_id)

        test_results = self.process_batch([work_item])

        comm.send((fold_id, test_results[0]['score']),
                  dest=0, tag=MPI_TAG_RESULT)

    def run(self):
        """Wait for new data until node receives a terminate or a test message.

        In the beginning, the node is waiting for new batches distributed by
        :class:`MPIGridSearchCVMaster._scatter_work`.
        After the grid search has been completed, the node either receives data
        from :func:`_fit_and_score_with_parameters` to evaluate the estimator
        given the parameters determined during grid-search, or is asked
        to terminate. Stop messages are: MPI_MSG_TERMINATE or MPI_MSG_TEST.
        """
        task_desc = self._task_desc

        while True:
            comm.Bcast([task_desc, MPI.INT], root=0)
            if task_desc[1] == MPI_MSG_TERMINATE:
                LOG.debug("Node %d received terminate message", comm_rank)
                return
            if task_desc[1] == MPI_MSG_CV:
                self._run_grid_search()
            elif task_desc[1] == MPI_MSG_TEST:
                self._run_train_test()
                break
            else:
                raise ValueError('unknown task with id %d' % task_desc[1])

        LOG.debug("Node %d is terminating", comm_rank)


class MPIGridSearchCVMaster(MPIBatchWorker):
    """Running on the root node and distributes work across slaves."""

    def __init__(self, param_grid, cv_iter, estimator, scorer, fit_params):
        super(MPIGridSearchCVMaster, self).__init__(estimator,
                                                    scorer, fit_params)
        self.param_grid = param_grid
        self.cv_iter = cv_iter

    def _create_batches(self):
        param_iter = ParameterGrid(self.param_grid)

        # divide work into batches equal to the communicator's size
        work_batches = [[] for _ in range(comm_size)]
        i = 0
        for fold_id, (train_index, test_index) in enumerate(self.cv_iter):
            for parameters in param_iter:
                work_batches[i % comm_size].append((fold_id + 1, train_index,
                                                    test_index, parameters))
                i += 1

        return work_batches

    def _scatter_work(self):
        work_batches = self._create_batches()

        LOG.debug("Distributed items into %d batches of size %d", comm_size,
                  len(work_batches[0]))

        # Distribute batches across all nodes
        root_work_batch = comm.scatter(work_batches, root=0)
        # The root node also does receive one batch it has to process
        root_result_batch = self.process_batch(root_work_batch)
        return root_result_batch

    def _gather_work(self, root_result_batch):
        # collect results: list of list of dict of parameters and performance
        # measures
        result_batches = comm.gather(root_result_batch, root=0)

        out = []
        for result_batch in result_batches:
            if result_batch is None:
                continue
            for result_item in result_batch:
                out.append(result_item)
        LOG.debug("Received %d valid results", len(out))

        return pandas.DataFrame(out)

    def run(self, train_X, train_y):
        # tell slave that it should do hyper-parameter search
        self._task_desc[0] = 0
        self._task_desc[1] = MPI_MSG_CV

        comm.Bcast([self._task_desc, MPI.INT], root=0)
        comm.bcast((train_X, train_y), root=0)

        self._data_X = train_X
        self._data_y = train_y

        root_result_batch = self._scatter_work()
        return self._gather_work(root_result_batch)
