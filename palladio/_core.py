import numpy as np

def generate_job_list(N_jobs_regular, N_jobs_permutation):
    """Generates a vector used to distribute jobs among nodes
    
    Given the total number of processes, generate a list of jobs distributing the load,
    so that each process has approximately the same amount of work to do
    (i.e., the same number of regular and permutated instances of the experiment).
    
    Parameters
    ----------
    
    N_jobs_regular : int
        The number of *regular* jobs, i.e. experiments where the labels
        have not been randomly shuffled.
        
    N_jobs_permutation : int
        The number of experiments for the permutation test, i.e. experiments where the labels
        *in the training set* will be randomly shuffled in order to disrupt any relationship
        between data and labels.
    
    Returns
    -------
    
    type_vector : numpy.ndarray
        A vector whose entries are either 0 or 1, representing respectively a job
        where a *regular* experiment is performed and one where an experiment where
        labels *in the training set* are randomly shuffled is performed.
    
    """
    
    # The total number of jobs
    N_jobs_total = N_jobs_permutation + N_jobs_regular
    
    # A vector representing the type of experiment: 1 for permutation, 0 for regular
    type_vector = np.ones((N_jobs_total,))
    type_vector[N_jobs_permutation:] = 0
    np.random.shuffle(type_vector)
    
    return type_vector