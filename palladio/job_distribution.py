"""Master slave."""
from __future__ import print_function
import os
import imp
import numbers
import shutil
import time

try:
    import cPickle as pkl
except:
    import pickle as pkl

from palladio.utils import sec_to_timestring
from palladio.utils import objectify_config
from palladio.model_assessment import ModelAssessment
from palladio.datasets import copy_files

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


def main(config=None, config_path=None):
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


    if config is None and config_path is None:
        raise Exception("Both config and config_path are None")

    if config_path is not None:
        # Load config
        config_dir = os.path.dirname(config_path)

        # For some reason, it must be atomic
        imp.acquire_lock()
        config = imp.load_source('config', config_path)
        imp.release_lock()

    # Load dataset
    if RANK == 0:
        print("Loading dataset...")

    data, labels = config.data, config.labels

    # Session folder
    # Depends whether the configuration path is specified
    # or the object itself
    if config_path is not None:
        session_folder = os.path.join(config_dir, config.session_folder)
    else:
        session_folder = config.session_folder

    experiments_folder_path = os.path.join(session_folder, 'experiments')

    # Create base session folder
    # Also copy dataset files inside it
    if RANK == 0:
        # Create main session folder
        if os.path.exists(session_folder):  # TODO: why [:-1]?
            # shutil.move(session_folder, session_folder[:-1] + '_old')
            # raise Exception("Session folder {} already exists, aborting."
            #                 .format(session_folder))
            pass

        os.mkdir(session_folder)
        # Create experiments folder (where all experiments sub-folders will
        # be created)
        os.mkdir(experiments_folder_path)

        # If a config file was provided, make a copy in the session folder
        if config_path is not None:
            shutil.copy(config_path, os.path.join(session_folder, 'config.py'))
        else:
            # Dump the configuration object using pickle
            with open(os.path.join(session_folder, 'config.pkl'), 'wb') as f:
                pkl.dump(config, f)

        # CREATE HARD LINK IN SESSION FOLDER
        if hasattr(config, 'data_path') and hasattr(config, 'target_path'):
            copy_files(config.data_path, config.target_path,
                       config_dir, session_folder)

    if IS_MPI_JOB:
        # Wait for the folder to be created and files to be copied
        COMM.barrier()

    if RANK == 0:
        print('  * Data shape:', data.shape)
        print('  * Labels shape:', labels.shape)

    internal_estimator = config.estimator
    ma_options = config.ma_options if hasattr(config, 'ma_options') else {}
    ma_options['experiments_folder'] = experiments_folder_path

    # XXX these depends on regular or permutation
    ma_options.pop('shuffle_y', None)

    external_estimator = ModelAssessment(internal_estimator, **ma_options)
    external_estimator.fit(data, labels)

    # Perform "permutation" experiments
    ma_options.pop('n_splits', None)
    n_splits_permutation = int(config.n_splits_permutation) if hasattr(
        config, 'n_splits_permutation') and isinstance(
        config.n_splits_permutation, numbers.Number) else None
    if n_splits_permutation is not None:
        external_estimator = ModelAssessment(
            internal_estimator,
            n_splits=n_splits_permutation,
            shuffle_y=True,
            **ma_options
        )
        external_estimator.fit(data, labels)

    if IS_MPI_JOB:
        # Wait for all jobs to end
        COMM.barrier()

    if RANK == 0:
        t100 = time.time()
        with open(os.path.join(session_folder, 'report.txt'), 'w') as rf:
            rf.write("Total elapsed time: {}".format(
                sec_to_timestring(t100 - t0)))
