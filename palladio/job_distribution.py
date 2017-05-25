"""Master slave."""
from __future__ import print_function
import os
import imp
import numbers
import shutil
import time
import gzip
import logging

try:
    import cPickle as pkl
except:
    import pickle as pkl

from palladio.utils import sec_to_timestring
from palladio.model_assessment import ModelAssessment
from palladio.datasets import copy_files
from palladio.session import Session

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()
    IS_MPI_JOB = COMM.Get_size() > 1

except ImportError:
    # print("mpi4py module not found. PALLADIO cannot run on multiple machines.")
    COMM = None
    RANK = 0
    NAME = 'localhost'
    IS_MPI_JOB = False


MAX_RESUBMISSIONS = 2
DO_WORK = 100
EXIT = 200

def busy_wait(f):
    """
    Wait for the creation of a folder/file

    Workaround for distributed architectures
    """

    while True:
        if os.path.exists(f):
            break
        else:
            time.sleep(1)

def main(pd_session_object, base_folder):
    """Main function.

    The main function, performs initial tasks such as creating the main session
    folder and copying files inside it,
    as well as distributing the jobs on all available machines.

    Parameters
    ----------
    pd_session_object : string
        A `palladio.session.Session` object, containing all information
        required to run a **PALLADIO** session
    """

    if RANK == 0:
        t0 = time.time()

    # if config is None and config_path is None:
    #     raise Exception("Both config and config_path are None")

    # if config_path is not None:
    #     # Load config
    #     config_dir = os.path.dirname(config_path)
    #
    #     # For some reason, it must be atomic
    #     imp.acquire_lock()
    #     config = imp.load_source('config', config_path)
    #     imp.release_lock()

    # Load dataset
    if RANK == 0:
        print("Loading dataset...")

    # data, labels = config.data, config.labels

    # TODO use properties to access attributes
    data, labels = pd_session_object._data, pd_session_object._labels



    # # Session folder
    # # Depends whether the configuration path is specified
    # or the object itself
    # if config_path is not None:
    #     session_folder = os.path.join(config_dir, config.session_folder)
    # else:
    #     session_folder = config.session_folder


    # TODO use properties to access attributes
    session_folder = os.path.join(
        base_folder,
        pd_session_object._session_folder
    )
    # session_folder = pd_session_object._session_folder

    experiments_folder_path = os.path.join(session_folder, 'experiments')
    logs_folder_path = os.path.join(session_folder, 'logs')

    # Create base session folder
    # Also copy dataset files inside it
    if RANK == 0:

        # Create main session folder
        os.mkdir(session_folder)

        # Create experiments folder (where all experiments sub-folders will
        # be created)
        os.mkdir(experiments_folder_path)

        # Create folder which will contain all logs
        os.mkdir(logs_folder_path)

        # TODO use properties to access attributes
        # If a config file was provided, make a copy in the session folder
        if pd_session_object._config_path is not None:
            shutil.copy(
                pd_session_object._config_path,
                os.path.join(session_folder, 'config.py')
            )

        # Then delete it before dumping TODO explain why
        pd_session_object._config_path = None
        pd_session_object._session_folder = None

        # CREATE HARD LINK IN SESSION FOLDER
        # if hasattr(config, 'data_path') and hasattr(config, 'target_path'):
        #     copy_files(config.data_path, config.target_path,
        #                config_dir, session_folder)

        # Dump session object
        with gzip.open(
                os.path.join(session_folder, 'pd_session.pkl.gz'), 'w') as f:
            pkl.dump(pd_session_object, f)

    # Busy wait for folder creation
    # Workaround to cope with filesystem consistency issues on
    # distributed architectures
    busy_wait(session_folder)
    busy_wait(experiments_folder_path)

    # ### Initialize logging object
    # It will log stuff on the same file for all processes
    # And an individual copy

    # Create the main logging object
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # create console handler with INFO log level
    # DEBUG level messages won't be shown here
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)


    # create file handler which logs even debug messages
    # This logger writes on a shared file

    fh_shared = logging.FileHandler(os.path.join(logs_folder_path, 'pd_session.log'))
    fh_shared.setLevel(logging.DEBUG)

    # A handler for each individual process
    fh_single = logging.FileHandler(os.path.join(logs_folder_path, 'worker_{}{}.log'.format(NAME, RANK)))
    fh_single.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh_shared.setFormatter(formatter)
    fh_single.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh_shared)
    logger.addHandler(fh_single)

    logger.debug('Logger initialization completed')

    if IS_MPI_JOB:
        # Wait for the folder to be created and files to be copied
        COMM.barrier()

    if RANK == 0:
        print('  * Data shape:', data.shape)
        print('  * Labels shape:', labels.shape)

    internal_estimator = pd_session_object._estimator

    # ma_options = config.ma_options if hasattr(config, 'ma_options') else {}
    ma_options = pd_session_object._ma_options
    ma_options['experiments_folder'] = experiments_folder_path

    # TODO XXX these depends on regular or permutation
    ma_options.pop('shuffle_y', None)

    # n_splits_regular = ma_options.pop('n_splits', None)
    # n_splits_regular = int(n_splits_regular) if (
    #     n_splits_regular is not None) and isinstance(
    #     n_splits_regular, numbers.Number) else None

    n_splits_regular = pd_session_object._n_splits_regular



    if n_splits_regular is not None and n_splits_regular > 0:
        logger.info('Performing regular experiments')
        ma_regular = ModelAssessment(
            internal_estimator, **ma_options).fit(
                data, labels)
        logger.info('Regular experiments completed')
    else:
        ma_regular = None

    # Perform "permutation" experiments
    ma_options.pop('n_splits', None)

    # n_splits_permutation = int(config.n_splits_permutation) if hasattr(
    #     config, 'n_splits_permutation') and isinstance(
    #     config.n_splits_permutation, numbers.Number) else None

    n_splits_permutation = pd_session_object._n_splits_permutation

    if n_splits_permutation is not None and n_splits_permutation > 0:
        logger.info('Performing permutation experiments')
        ma_permutation = ModelAssessment(
            internal_estimator,
            n_splits=n_splits_permutation,
            shuffle_y=True,
            **ma_options
        ).fit(data, labels)
        logger.info('Permutation experiments completed')
    else:
        ma_permutation = None

    if IS_MPI_JOB:
        # Wait for all jobs to end
        COMM.barrier()

    if RANK == 0:
        t100 = time.time()
        with open(os.path.join(session_folder, 'report.txt'), 'w') as rf:
            rf.write("Total elapsed time: {}".format(
                sec_to_timestring(t100 - t0)))

    return ma_regular, ma_permutation
