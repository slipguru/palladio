# Configuration file example for PALLADIO
# version: 0.5

import numpy as np
from palladio.wrappers import ElasticNetClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score

from palladio.datasets import DatasetCSV

#####################
#   DATASET PATHS ###
#####################

# * All the path are w.r.t. config file path

dataset_class = DatasetCSV

# The list of all files required for the experiments
dataset_files = {
    'data': 'data/gedm.csv',
    'labels': 'data/labels.csv',
}

dataset_options = {
    'samples_on': 'row',  # or 'row': samples on cols or rows
    'positive_label': 1,  # positive class in case of 2-class task
}

#######################
#   SESSION OPTIONS ###
#######################

result_path = 'palladio_test_enet'

# The number of "regular" experiment
N_jobs_regular = 100

# The number of instances for the permutation tests
# (labels in the training sets are randomly shuffled)
N_jobs_permutation = 100

# The ratio of the dataset held out for model assessment
# It should be of the form 1/M
test_set_ratio = 1 / 4.

#######################
#  LEARNER OPTIONS  ###
#######################

learner = ElasticNetClassifier
force_classifier = False

learner_options = {
    'fit_intercept': True
}


# ~~ Elastic-Net Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
param_grid = {
    'l1_ratio': np.logspace(0.5, 0, 10),
    'alpha': np.logspace(-1, 0, 20)
}

# ~~ Data filtering/normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data_normalizer = l1l2py.tools.center
# labels_normalizer = None

# ~~ Cross validation options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cv_options = {
    'param_grid': param_grid,
    'cv': 3,
    'scoring': 'accuracy',
    'n_jobs': -1
}

final_scoring = accuracy_score

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75
