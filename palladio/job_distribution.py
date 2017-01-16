"""Master slave."""
from __future__ import print_function
import os
import imp
import logging
import shutil
import time

from collections import deque

from palladio.core import run_experiment, generate_job_list
from palladio.utils import sec_to_timestring

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()

    IS_MPI_JOB = COMM.Get_size() > 1

except ImportError:
    print("mpi4py module not found. Adenine cannot run on multiple machines.")
    COMM = None
    RANK = 0
    NAME = 'localhost'

    IS_MPI_JOB = False

# MAX_RESUBMISSIONS = 2
# constants to use as tags in communications
DO_WORK = 100
EXIT = 200

# VERBOSITY
# VERBOSITY = 1


def master_single_machine(pipes, X):
    """Fit and transform/predict some pipelines on some data (single machine).

    This function fits each pipeline in the input list on the provided data.
    The results are dumped into a pkl file as a dictionary of dictionaries of
    the form {'pipe_id': {'stepID' : [alg_name, level, params, data_out,
    data_in, model_obj, voronoi_suitable_object], ...}, ...}. The model_obj is
    the sklearn model which has been fit on the dataset, the
    voronoi_suitable_object is the very same model but fitted on just the first
    two dimensions of the dataset. If a pipeline fails for some reasons the
    content of the stepID key is a list of np.nan.

    Parameters
    -----------
    pipes : list of list of tuples
        Each tuple contains a label and a sklearn Pipeline object.
    X : array of float, shape : n_samples x n_features, default : ()
        The input data matrix.

    Returns
    -----------
    pipes_dump : dict
        Dictionary with the results of the computation.
    """
    import multiprocessing as mp
    jobs = []
    manager = mp.Manager()
    pipes_dump = manager.dict()

    # Submit jobs
    for i, pipe in enumerate(pipes):
        pipe_id = 'pipe' + str(i)
        proc = mp.Process(target=pipe_worker,
                          args=(pipe_id, pipe, pipes_dump, X))
        jobs.append(proc)
        proc.start()
        logging.info("Job: %s submitted", pipe_id)

    # Collect results
    count = 0
    for proc in jobs:
        proc.join()
        count += 1
    logging.info("%d jobs collected", count)

    return dict(pipes_dump)


def master(config):
    """Distribute pipelines with mpi4py or multiprocessing."""
    # Pipeline definition
    # pipes = define_pipeline.parse_steps(
    #     [config.step0, config.step1,
    #      config.step2, config.step3])
    #
    # if not IS_MPI_JOB:
    #     return master_single_machine(pipes, config.X)

    # RUN PIPELINES
    nprocs = COMM.Get_size()
    # print(NAME + ": start running slaves", nprocs, NAME)
    queue = deque(list(enumerate(generate_job_list(
        config.N_jobs_regular, config.N_jobs_permutation))))

    pipe_dump = dict()
    count = 0
    n_pipes = len(queue)

    # seed the slaves by sending work to each processor
    for rankk in range(1, min(nprocs, n_pipes)):
        pipe_tuple = queue.popleft()
        COMM.send(pipe_tuple, dest=rankk, tag=DO_WORK)
        # print(NAME + ": send to rank", rankk)

    # loop until there's no more work to do. If queue is empty skips the loop.
    while queue:
        pipe_tuple = queue.popleft()
        # receive result from slave
        status = MPI.Status()
        COMM.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        # pipe_dump[pipe_id] = step_dump
        count += 1
        # send to the same slave new work
        COMM.send(pipe_tuple, dest=status.source, tag=DO_WORK)

    # there's no more work to do, so receive all the results from the slaves
    for rankk in range(1, min(nprocs, n_pipes)):
        # print(NAME + ": master - waiting from", rankk)
        status = MPI.Status()
        COMM.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        # pipe_dump[pipe_id] = step_dump
        count += 1

    # tell all the slaves to exit by sending an empty message with the EXIT_TAG
    for rankk in range(1, nprocs):
        # print(NAME + ": master - killing", rankk)
        COMM.send(0, dest=rankk, tag=EXIT)

    # print(NAME + ": terminating master")
    # return pipe_dump


def slave(data, labels, config, experiments_folder_path):
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
            i, is_permutation_test = received
            # print(NAME + ": slave received", RANK, i)
            custom_name = "{}_p_{}_i_{}".format(
                ("permutation" if is_permutation_test else "regular"), RANK, i)
            tmp_name_base = 'tmp_' + custom_name

            experiment_resubmissions = 0
            experiment_completed = False
            MAX_RESUBMISSIONS = 2
            while not experiment_completed and \
                    experiment_resubmissions <= MAX_RESUBMISSIONS:
                try:
                    tmp_name = tmp_name_base + '_submission_{}'.format(
                        experiment_resubmissions + 1)
                    run_experiment(data, labels, None, config,
                                   is_permutation_test, experiments_folder_path,
                                   tmp_name)
                    experiment_completed = True

                    shutil.move(
                        os.path.join(experiments_folder_path, tmp_name),
                        os.path.join(experiments_folder_path, custom_name),
                    )
                    print("[{}_{}] finished experiment {}".format(NAME, RANK, i))

                except Exception as e:
                    raise
                    # If somethings out of the ordinary happens,
                    # resubmit the job
                    experiment_resubmissions += 1
                    print("[{}_{}] failed experiment {}, resubmission #{}\n"
                          "Exception raised: {}".format(
                              NAME, RANK, i, experiment_resubmissions, e))

            if not experiment_completed:
                print("[{}_{}] failed to complete experiment {}, "
                      "max resubmissions limit reached".format(NAME, RANK, i))

            COMM.send(0, dest=0, tag=0)

    except StandardError as exc:
        print("Quitting ... TB:", str(exc))


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

    dataset = config.dataset_class(
        config.dataset_files,
        config.dataset_options
    )

    data, labels, _ = dataset.load_dataset(config_dir)

    # Session folder
    result_path = os.path.join(config_dir, config.result_path)
    experiments_folder_path = os.path.join(result_path, 'experiments')

    # Create base session folder
    # Also copy dataset files inside it
    if RANK == 0:
        # Create main session folder
        if os.path.exists(result_path):
            shutil.move(result_path, result_path[:-1] + '_old')
            # raise Exception("Session folder {} already exists, aborting."
            #                 .format(result_path))

        os.mkdir(result_path)
        # Create experiments folder (where all experiments sub-folders will
        # be created)
        os.mkdir(experiments_folder_path)

        shutil.copy(config_path, os.path.join(result_path, 'config.py'))

        # CREATE HARD LINK IN SESSION FOLDER
        dataset.copy_files(config_dir, result_path)

    if IS_MPI_JOB:
        # Wait for the folder to be created and files to be copied
        COMM.barrier()

    if RANK == 0:
        print('  * Data shape:', data.shape)
        print('  * Labels shape:', labels.shape)

        master(config)
    else:
        slave(data, labels, config, experiments_folder_path)

    if IS_MPI_JOB:
        # Wait for all jobs to end
        COMM.barrier()

    if RANK == 0:
        t100 = time.time()
        with open(os.path.join(result_path, 'report.txt'), 'w') as rf:
            rf.write("Total elapsed time: {}".format(
                sec_to_timestring(t100 - t0)))
