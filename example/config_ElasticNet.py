# Configuration file example for PALLADIO
# version: '0.4

import numpy as np
from palladio.wrappers.elastic_net import ElasticNetClassifier
from palladio.datasets import DatasetNPY

import l1l2py

#####################
#   DATASET PATHS ###
#####################

# * All the path are w.r.t. config file path

dataset_class = DatasetNPY

# The list of all files required for the experiments
dataset_files = {
    'data': 'data.npy',
    'labels': 'labels.npy',
    'indcol': 'indcols.pkl'
}

dataset_options = {
    'delimiter': ',',
    'samples_on': 'row',  # or 'row': samples on cols or rows
    'positive_label': 1,  # positive class in case of 2-class task
    'header': 0,
    'index_col': 0
}

#######################
#   SESSION OPTIONS ###
#######################

result_path = 'dummy_palladio_test_2step'

# The number of "regular" experiment
N_jobs_regular = 100

# The number of instances for the permutation tests
# (labels in the training sets are randomly shuffled)
N_jobs_permutation = 100

# The ratio of the dataset held out for model assessment
# It should be of the form 1/M
test_set_ratio = float(1) / 4

#######################
#  LEARNER OPTIONS  ###
#######################

learner_class = ElasticNetClassifier

# ~~ Elastic-Net Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
l1_ratio_range = np.logspace(0.5, 0, 10)
alpha_range = np.logspace(-1, 0, 20)  # * CORRELATION_FACTOR

# ~~ Data filtering/normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_normalizer = l1l2py.tools.center
labels_normalizer = None

# ~~ Cross validation options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
internal_k = 3
cv_splitting = l1l2py.tools.stratified_kfold_splits

# ~~ Errors functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * See l1l2py.tools.{regression_error, classification_error,
#                     balanced_classification_error}
cv_error = l1l2py.tools.regression_error
error = l1l2py.tools.balanced_classification_error

learner_params = {
    'l1_ratio_range': l1_ratio_range,
    'alpha_range': alpha_range,
    'data_normalizer': data_normalizer,
    'labels_normalizer': labels_normalizer,
    'cv_error': cv_error,
    'error': error
}

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75
