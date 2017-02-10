"""Master slave."""
from __future__ import print_function
import os
import imp
import shutil
import time

from palladio.utils import sec_to_timestring
from palladio.model_assessment import ModelAssessment
from palladio.datasets import copy_files

from sklearn.model_selection import GridSearchCV

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()
    IS_MPI_JOB = COMM.Get_size() > 1

except ImportError:
    print("mpi4py module not found. PALLADIO cannot run on multiple machines.")
    COMM = None
    RANK = 0
    NAME = 'localhost'
    IS_MPI_JOB = False

MAX_RESUBMISSIONS = 2
DO_WORK = 100
EXIT = 200


def main(config_path):
    """Main function.

    The main function, performs initial tasks such as creating the main session
    folder and copying files inside it,
    as well as distributing the jobs on all available machines.

    Parameters
    ----------
    config_path : string
        A path to the configuration file containing all required information
        to run a **PALLADIO** session.
    """
    if RANK == 0:
        t0 = time.time()

    # Load config
    config_dir = os.path.dirname(config_path)

    # For some reason, it must be atomic
    imp.acquire_lock()
    config = imp.load_source('config', config_path)
    imp.release_lock()

    # Load dataset
    if RANK == 0:
        print("Loading dataset...")

    # dataset = config.dataset_class(
    #     config.dataset_files,
    #     config.dataset_options
    # )
    #
    # data, labels, _ = dataset.load_dataset(config_dir)

    data, labels = config.data, config.labels

    # Session folder
    result_path = os.path.join(config_dir, config.result_path)
    experiments_folder_path = os.path.join(result_path, 'experiments')

    # Create base session folder
    # Also copy dataset files inside it
    if RANK == 0:
        # Create main session folder
        if os.path.exists(result_path):  # TODO: why [:-1]?
            shutil.move(result_path, result_path[:-1] + '_old')
            # raise Exception("Session folder {} already exists, aborting."
            #                 .format(result_path))

        os.mkdir(result_path)
        # Create experiments folder (where all experiments sub-folders will
        # be created)
        os.mkdir(experiments_folder_path)

        shutil.copy(config_path, os.path.join(result_path, 'config.py'))

        # CREATE HARD LINK IN SESSION FOLDER
        copy_files(config.data_path, config.target_path,
                   config_dir, result_path)

    if IS_MPI_JOB:
        # Wait for the folder to be created and files to be copied
        COMM.barrier()

    if RANK == 0:
        print('  * Data shape:', data.shape)
        print('  * Labels shape:', labels.shape)

        # master(config, data, labels, experiments_folder_path)
    else:
        pass
        # slave(data, labels, config, experiments_folder_path)

    # HERE ALL STUFF

    # Prepare estimator for internal loops (GridSearchCV)

    # The internal estimator (e.g. Elastic Net Classifier)
    # internal_estimator = config.learner(**config.learner_options)
    internal_estimator = config.estimator

    # Grid search estimator
    internal_gridsearch = GridSearchCV(internal_estimator, **config.cv_options)

    # Perform "regular" experiments
    external_estimator = ModelAssessment(
        internal_gridsearch,
        scoring=config.final_scoring,
        shuffle_y=False,
        n_splits=config.N_jobs_regular,
        test_size=0.25,
        train_size=None,
        experiments_folder=experiments_folder_path)
    external_estimator.fit(data, labels)

    # Perform "permutation" experiments
    external_estimator = ModelAssessment(
        internal_gridsearch,
        scoring=config.final_scoring,
        shuffle_y=True,
        n_splits=config.N_jobs_permutation,
        test_size=0.25,
        train_size=None,
        experiments_folder=experiments_folder_path)
    external_estimator.fit(data, labels)

    if IS_MPI_JOB:
        # Wait for all jobs to end
        COMM.barrier()

    if RANK == 0:
        t100 = time.time()
        with open(os.path.join(result_path, 'report.txt'), 'w') as rf:
            rf.write("Total elapsed time: {}".format(
                sec_to_timestring(t100 - t0)))
