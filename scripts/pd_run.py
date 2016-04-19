#!/usr/bin/python -u
'''
#!/usr/bin/env python -u
'''
# -*- coding: utf-8 -*-
import os, sys
import imp
import shutil
import cPickle as pkl
import random

import numpy as np

from mpi4py import MPI

import l1l2py

from l1l2signature import internals as l1l2_core
from l1l2signature import utils as l1l2_utils

### Initialize MPI variables
### THESE ARE GLOBALS
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

def generate_job_list(N_jobs_regular, N_jobs_permutation):
    """
    Given the total number of processes, generate a list of jobs distributing the load,
    so that each process has approximately the same amount of work to do
    (i.e., the same number of regular and permutated instances of the experiment)
    """
    
    # The total number of jobs
    N_jobs_total = N_jobs_permutation + N_jobs_regular
    
    # A vector representing the type of experiment: 1 for permutation, 0 for regular
    type_vector = np.ones((N_jobs_total,))
    type_vector[N_jobs_permutation:] = 0
    np.random.shuffle(type_vector)
    
    return type_vector

def modelselection_job(Xtr, Ytr, Xts, Yts, tau_range, mu_range, lambda_range, config, result_dir, random_seed = None):
    """
    A single model selection job
    """

    sparse, regularized, return_predictions = (True, False, True)
    
    # Parameters
    
    int_k = config.internal_k
    ms_split = config.cv_splitting(Ytr, int_k) # args[3]=k -> splits
    
    # Execution
    result = l1l2py.model_selection(
                    Xtr, Ytr, Xts, Yts, 
                    mu_range, tau_range, lambda_range,
                    ms_split, config.cv_error, config.error,
                    config.data_normalizer, config.labels_normalizer,
                    sparse=sparse, regularized=regularized, return_predictions=return_predictions
                    )


    out = {
        'result' : result,
        'ms_split' : ms_split
    }

    return out

    
def run_experiment(data, labels, config_dir, config, is_permutation_test, custom_name):
    
    result_path = os.path.join(config_dir, config.result_path) #result base dir
    
    ### Create experiment folders
    result_dir = os.path.join(result_path, custom_name)
    os.mkdir(result_dir)
    
    ### Split the dataset in learning and test set
    ### Use a trick to keep the original splitting strategy
    aux_splits = config.cv_splitting(labels, int(round(1/(config.test_set_ratio))), rseed = None)
    
    idx_lr = aux_splits[0][0]
    idx_ts = aux_splits[0][1]
    
    data_lr = data[idx_lr, :]
    labels_lr = labels[idx_lr]
    
    data_ts = data[idx_ts, :]
    labels_ts = labels[idx_ts]
    
    ### Compute the ranges of the parameters using only the learning set
    if is_permutation_test:
        labels_perm = labels_lr.copy()
        np.random.shuffle(labels_perm)
        rs = l1l2_utils.RangesScaler(data_lr, labels_perm, config.data_normalizer,
                                               config.labels_normalizer)
    else:
        rs = l1l2_utils.RangesScaler(data_lr, labels_lr, config.data_normalizer,
                                               config.labels_normalizer)
    
    tau_range = rs.tau_range(config.tau_range)
    mu_range = rs.mu_range(config.mu_range)
    lambda_range = np.sort(config.lambda_range)
    
    if is_permutation_test:
        out = modelselection_job(data_lr, labels_perm, data_ts, labels_ts, tau_range, mu_range, lambda_range, config, result_dir, random_seed = None)
    else:
        out = modelselection_job(data_lr, labels_lr, data_ts, labels_ts, tau_range, mu_range, lambda_range, config, result_dir, random_seed = None)
    result = out['result']
    result['labels_ts'] = labels_ts ### also save labels
    
    # save results 
    with open(os.path.join(result_dir, 'result.pkl'), 'w') as f:
        pkl.dump(result, f, pkl.HIGHEST_PROTOCOL)
        
    in_split = {
        'ms_split': out['ms_split'],
        'outer_split': aux_splits[0]
    }
        
    with open(os.path.join(result_dir, 'in_split.pkl'), 'w') as f:
        pkl.dump(in_split, f, pkl.HIGHEST_PROTOCOL)
    
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def main(config_path, custom_name = None):
def main(config_path):
    
    
    
    # Configuration File
    config_dir = os.path.dirname(config_path)


    imp.acquire_lock()
    config = imp.load_source('config', config_path)
    imp.release_lock()
    
    # Data paths
    data_path = os.path.join(config_dir, config.data_matrix)
    labels_path = os.path.join(config_dir, config.labels)
    
    ### Create base results dir if it does not already exist
    if rank == 0:
        result_path = os.path.join(config_dir, config.result_path) #result base dir
        if not os.path.exists(result_path):
            os.mkdir(result_path)
            
        shutil.copy(config_path, os.path.join(result_path, 'config.py'))
        os.link(data_path, os.path.join(result_path, 'data_file'))
        os.link(labels_path, os.path.join(result_path, 'labels_file'))
        
            
    ### Wait for the folder to be created and files to be copied
    comm.barrier()
    
    # Experimental design
    N_jobs_regular = config.N_jobs_regular
    N_jobs_permutation = config.N_jobs_permutation
    
    if rank == 0:
        print("") #--------------------------------------------------------------------
        print('Reading data... ')
        
    br = l1l2_utils.BioDataReader(data_path, labels_path,
                                  config.sample_remover,
                                  config.variable_remover,
                                  config.delimiter,
                                  config.samples_on,
                                  config.positive_label)
    data = br.data
    labels = br.labels

    if rank == 0:
        print('  * Data shape:', data.shape)
        print('  * Labels shape:', labels.shape)
        
    if rank == 0:
        job_list = generate_job_list(N_jobs_regular, N_jobs_permutation)
    else:
        job_list = None
    
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
        
        pass
        


    






















    return    




# def main2():
#     
#     import numpy.distutils.system_info as sysinfo
#     
#     import time
#     
#     # print sysinfo.get_info('atlas')
#     # print sysinfo.get_info('blas')
#     # print sysinfo.get_info('openblas')
#     
#     
#     print os.environ['PYTHONPATH']
#     
#     A = np.random.normal(size = (5000, 5000))
#     
#     tic = time.time()
#     
#     A.dot(A)
#     
#     tac = time.time()
#     
#     dt = tac - tic
#     
#     print("Elapsed time = {}".format(dt))
    
    

        

# Script entry ----------------------------------------------------------------
if __name__ == '__main__':
    
    # main2()
    
    
    if len(sys.argv) != 2:
        parser.error('incorrect number of arguments')
    config_file_path = sys.argv[1]
    
    main(os.path.abspath(config_file_path))
