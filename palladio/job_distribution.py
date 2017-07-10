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
# from palladio.session import Session

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
    data, labels = pd_session_object.data, pd_session_object.labels

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
        pd_session_object.session_folder
    )
    # session_folder = pd_session_object.session_folder

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
        if pd_session_object.config_path is not None:
            shutil.copy(
                pd_session_object.config_path,
                os.path.join(session_folder, 'config.py')
            )

        # Then delete it before dumping TODO explain why
        pd_session_object.config_path = None
        pd_session_object.session_folder = None

        # Dump session object
        with gzip.open(
                os.path.join(session_folder, 'pd_session.pkl.gz'), 'w') as f:
            pkl.dump(pd_session_object, f)

    # Busy wait for folder creation
    # Workaround to cope with filesystem consistency issues on
    # distributed architectures
    busy_wait(session_folder)
    busy_wait(experiments_folder_path)

    # create formatter for loggers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if RANK == 0:
        # ### Initialize master logger
        master_logger = logging.getLogger('master')
        master_logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(
            logs_folder_path, 'master_{}.log'.format(NAME)))
        fh.setLevel(logging.DEBUG)

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to the logger
        master_logger.addHandler(ch)
        # logger.addHandler(fh_shared)
        master_logger.addHandler(fh)

    # Create the main logging object
    worker_logger = logging.getLogger('worker')
    worker_logger.setLevel(logging.DEBUG)

    # create console handler with INFO log level
    # DEBUG level messages won't be shown here
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    # This logger writes on a shared file

    # fh_shared = logging.FileHandler(os.path.join(logs_folder_path, 'pd_session.log'))
    # fh_shared.setLevel(logging.DEBUG)

    # A handler for each individual process
    fh_single = logging.FileHandler(os.path.join(
        logs_folder_path, 'worker_{}{}.log'.format(NAME, RANK)))
    fh_single.setLevel(logging.DEBUG)

    # Assigning formatter to handlers
    ch.setFormatter(formatter)
    # fh_shared.setFormatter(formatter)
    fh_single.setFormatter(formatter)

    # add the handlers to the logger
    worker_logger.addHandler(ch)
    # logger.addHandler(fh_shared)
    worker_logger.addHandler(fh_single)

    if RANK == 0:

        master_logger.debug('Logger initialization completed')

        master_logger.info('Data shape: {}'.format(data.shape))
        master_logger.info('Labels shape: {}'.format(labels.shape))

    if IS_MPI_JOB:
        # Wait for the folder to be created and files to be copied
        COMM.barrier()

    internal_estimator = pd_session_object.estimator

    ma_options = pd_session_object.ma_options if hasattr(
        pd_session_object, 'ma_options') else {}
    # ma_options = pd_session_object.ma_options
    ma_options['experiments_folder'] = experiments_folder_path

    # TODO XXX these depends on regular or permutation
    ma_options.pop('shuffle_y', None)

    n_splits_regular = ma_options.pop('n_splits', None)
    if n_splits_regular is None and hasattr(
            pd_session_object, 'n_splits_regular'):
        n_splits_regular = int(pd_session_object.n_splits_regular) if \
            pd_session_object.n_splits_regular is not None else None

    # if n_splits_regular is not specified and is not in ma_options
    # n_splits_regular is None
    # then get the default of ModelAssessment
    n_splits_regular = int(n_splits_regular) if (
        n_splits_regular is not None) and isinstance(
        n_splits_regular, numbers.Number) else 10

    # n_splits_regular = pd_session_object.n_splits_regular

    if n_splits_regular is not None and n_splits_regular > 0:
        worker_logger.info('Performing regular experiments')
        ma_regular = ModelAssessment(
            internal_estimator,
            n_splits=n_splits_regular,
            **ma_options).fit(
                data, labels)
        worker_logger.info('Regular experiments completed')
    else:
        ma_regular = None

    # Perform "permutation" experiments
    # ma_options.pop('n_splits', None)

    n_splits_permutation = int(pd_session_object.n_splits_permutation) if hasattr(
        pd_session_object, 'n_splits_permutation') and isinstance(
        pd_session_object.n_splits_permutation, numbers.Number) else None

    # n_splits_permutation = pd_session_object.n_splits_permutation

    if n_splits_permutation is not None and n_splits_permutation > 0:
        worker_logger.info('Performing permutation experiments')
        ma_permutation = ModelAssessment(
            internal_estimator,
            n_splits=n_splits_permutation,
            shuffle_y=True,
            **ma_options
        ).fit(data, labels)
        worker_logger.info('Permutation experiments completed')
    else:
        ma_permutation = None

    if IS_MPI_JOB:
        # Wait for all jobs to end
        COMM.barrier()

    if RANK == 0:
        t100 = time.time()
        master_logger.info('Session complete, elapsed time: {}'.format(
            sec_to_timestring(t100 - t0)))

    return ma_regular, ma_permutation
