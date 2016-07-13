import os
import imp
import shutil
import cPickle as pkl
import numpy as np
import time

from hashlib import sha512

from .utils import sec_to_timestring

# Initialize GLOBAL MPI variables (or dummy ones for the single process case)
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    IS_MPI_JOB = True

except:

    comm = None
    size = 1
    rank = 0
    name = 'localhost'

    IS_MPI_JOB = False


def generate_job_list(N_jobs_regular, N_jobs_permutation):
    """Generate a vector used to distribute jobs among nodes.

    Given the total number of processes, generate a list of jobs distributing
    the load, so that each process has approximately the same amount of work
    to do (i.e., the same number of regular and permutated instances of the
    experiment).

    Parameters
    ----------
    N_jobs_regular : int
        The number of *regular* jobs, i.e. experiments where the labels
        have not been randomly shuffled.

    N_jobs_permutation : int
        The number of experiments for the permutation test, i.e. experiments
        where the labels *in the training set* will be randomly shuffled in
        order to disrupt any relationship between data and labels.

    Returns
    -------
    type_vector : numpy.ndarray
        A vector whose entries are either 0 or 1, representing respectively a
        job where a *regular* experiment is performed and one where an
        experiment where labels *in the training set* are randomly shuffled
        is performed.
    """
    # The total number of jobs
    N_jobs_total = N_jobs_permutation + N_jobs_regular

    # A vector representing the type of experiment: 1 for permutation, 0 for regular
    type_vector = np.ones((N_jobs_total,))
    type_vector[N_jobs_permutation:] = 0
    np.random.shuffle(type_vector)

    return type_vector


def run_experiment(data, labels, config_dir, config, is_permutation_test, custom_name):
    """Run a single independent experiment.

    Perform a single experiment, which is divided in three main stages:

    * Dataset Splitting
    * Model Selection
    * Model Assessment

    The details of the process actually depend on the algorithms used.

    Parameters
    ----------
    data : ndarray
        A :math:`n \\times p` matrix describing the input data.

    labels : ndarray
        A :math:`n`-dimensional vector containing the labels indicating the
        class of the samples.

    config_dir : string
        The path to the folder containing the configuration file, which will
        also contain the main session folder.

    config : object
        The object containing all configuration parameters for the session.

    is_permutation_test : bool
        A flag indicatin whether the experiment is part of the permutation test
        (and therefore has had its training labels randomly shuffled) or not.

    custom_name : string
        The name of the subfolder where the experiments' results will be
        stored. It is a combination of a prefix which is either ``regular`` or
        ``permutation`` depending on the nature of the experiment, followed
        by two numbers which can be used to identify the experiment, for
        debugging purposes.
    """
    result_path = os.path.join(config_dir, config.result_path)  # result base dir

    # Create experiment folders
    result_dir = os.path.join(result_path, custom_name)
    os.mkdir(result_dir)

    # Produce a seed for the pseudo random generator
    rseed = 0
    aux = sha512(name).digest()  # hash the machine's name
    # extract integers from the sha
    for c in aux:
        try:
            rseed += int(c)
        except ValueError:
            pass
    rseed = time.time()
    rseed += rank

    # Split the dataset in learning and test set
    # Use a trick to keep the original splitting strategy
    aux_splits = config.cv_splitting(labels, int(round(1/(config.test_set_ratio))), rseed=rseed)
    # aux_splits = config.cv_splitting(labels, int(round(1/(config.test_set_ratio))))

    # idx_lr = aux_splits[0][0]
    # idx_ts = aux_splits[0][1]

    # A quick workaround to get just the first couple of the list;
    # one iteration, then break the loop
    for idx_tr, idx_ts in aux_splits:
        idx_lr = idx_tr
        idx_ts = idx_ts
        break

    data_lr = data[idx_lr, :]
    labels_lr = labels[idx_lr]

    data_ts = data[idx_ts, :]
    labels_ts = labels[idx_ts]

    # Compute the ranges of the parameters using only the learning set
    # TODO FIX
    if is_permutation_test:
        labels_perm = labels_lr.copy()
        np.random.shuffle(labels_perm)

    Xtr = data_lr
    Ytr = labels_perm if is_permutation_test else labels_lr
    Xts = data_ts
    Yts = labels_ts

    # Setup the internal splitting for model selection
    int_k = config.internal_k
    ms_split = config.cv_splitting(Ytr, int_k, rseed=time.clock())  # since it requires the labels, it can't be done before those are loaded
    config.learner_params['ms_split'] = ms_split

    # Create the object that will actually perform the classification/feature selection
    clf = config.learner_class(config.learner_params)

    # Set the actual data and perform additional steps such as rescaling parameters etc.
    clf.setup(Xtr, Ytr, Xts, Yts)

    ### Workaround: this is gonna work only if clf is an l1l2Classifier
    try:
        param_1_range = clf._tau_range
        param_2_range = clf._lambda_range
    except:
        param_1_range = None
        param_2_range = None
    ###

    result = clf.run()

    # result = out['result']
    result['labels_ts'] = labels_ts  # also save labels

    # save results
    with open(os.path.join(result_dir, 'result.pkl'), 'w') as f:
        pkl.dump(result, f, pkl.HIGHEST_PROTOCOL)

    in_split = {
        'ms_split': ms_split,
        # 'outer_split': aux_splits[0]
        'outer_split': (idx_lr, idx_ts),
        'param_ranges': (param_1_range, param_2_range)
    }

    with open(os.path.join(result_dir, 'in_split.pkl'), 'w') as f:
        pkl.dump(in_split, f, pkl.HIGHEST_PROTOCOL)

    return

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
    if rank == 0:
        t0 = time.time()

    ##########################
    ### LOAD CONFIGURATION ###
    ##########################

    config_dir = os.path.dirname(config_path)

    ### For some reason, it must be atomic
    imp.acquire_lock()
    config = imp.load_source('config', config_path)
    imp.release_lock()
    
    ####################
    ### LOAD DATASET ###
    ####################
    
    if rank == 0:
        print("Loading dataset...")
    
    dataset = config.dataset_class(
        config.dataset_files,
        config.dataset_options
    )
    
    data, labels, _  = dataset.load_dataset(config_dir)
    
    ### Create base results dir if it does not already exist
    ### Also copy dataset files inside it
    if rank == 0:
        result_path = os.path.join(config_dir, config.result_path) #result base dir
        
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        shutil.copy(config_path, os.path.join(result_path, 'config.py'))

        ### CREATE HARD LINK IN SESSION FOLDER
        dataset.copy_files(config_dir, result_path)
    
    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if IS_MPI_JOB:
        ### Wait for the folder to be created and files to be copied
        comm.barrier()

    if rank == 0:
        print('  * Data shape:', data.shape)
        print('  * Labels shape:', labels.shape)

    # Experimental design
    N_jobs_regular = config.N_jobs_regular
    N_jobs_permutation = config.N_jobs_permutation

    if rank == 0:
        job_list = generate_job_list(N_jobs_regular, N_jobs_permutation)
    else:
        job_list = None

    ### XXX Fix for single machine!!!
    ### Distribute job list with broadcast
    job_list = comm.bcast(job_list, root=0)

    ### Compute which jobs each process has to handle
    N_jobs_total = N_jobs_permutation + N_jobs_regular

    jobs_per_proc = N_jobs_total/size
    exceeding_jobs = N_jobs_total%size

    ### compute the local offset
    heavy_jobs = min(rank, exceeding_jobs) # jobs that make one extra iteration
    light_jobs = rank - heavy_jobs
    offset = heavy_jobs*(jobs_per_proc + 1) + light_jobs*jobs_per_proc
    idx = np.arange(offset, offset + jobs_per_proc + int(rank < exceeding_jobs))

    ### The jobs handled by this process
    local_jobs = job_list[idx]

    # print("Job {}, handling jobs {}".format(rank, idx))

    for i, is_permutation_test in enumerate([bool(x) for x in local_jobs]):

        # print("Job {}, is permutation test? {}".format(rank, is_permutation_test))

        ### Create a custom name for the experiment based on whether it is a permutation test,
        ### the process' rank and a sequential number
        custom_name = "{}_p_{}_i_{}".format(("permutation" if is_permutation_test else "regular"), rank, i)

        run_experiment(data, labels, config_dir, config, is_permutation_test, custom_name)

        print("[{}_{}] finished experiment {}".format(name, rank, i))

    if IS_MPI_JOB:
        ### Wait for all experiments to finish before taking the time
        comm.barrier()

    if rank == 0:
        t100 = time.time()

    if rank == 0:
        with open(os.path.join(result_path, 'report.txt'), 'w') as rf:

            rf.write("Total elapsed time: {}".format(sec_to_timestring(t100-t0)))

    return
