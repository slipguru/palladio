"""Configuration file example for PALLADIO  version: 0.4."""

from palladio.wrappers.l1l2 import l1l2Classifier
from palladio.datasets import DatasetNPY
import l1l2py

#####################
#  DATASET PATHS ###
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
    'positive_label': None  # positive class in case of 2-class task
}


#######################
#   SESSION OPTIONS ###
#######################

result_path = 'golub_palladio_test_l1l2'

# The number of "regular" experiment
N_jobs_regular = 100

# The number of instances for the permutation tests
# (labels in the training sets are randomly shuffled)
N_jobs_permutation = 100

# The ratio of the dataset held out for model assessment
# It should be of the form 1/M
test_set_ratio = float(1) / 4

#######################
#   LEARNER OPTIONS ###
#######################

learner_class = l1l2Classifier

# ~~ L1l2 Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * Ranges will be sorted from smaller to bigger value!
# * See l1l2py.tools.{geometric_range, linear_range}
tau_range = l1l2py.tools.geometric_range(1e-3, 0.5, 20)  # * MAX_TAU
# mu_range = l1l2py.tools.geometric_range(1e-3, 1.0, 3)   # *
# CORRELATION_FACTOR
mu = 1e-3  # * CORRELATION_FACTOR
lambda_range = l1l2py.tools.geometric_range(1e0, 1e4, 10)

sparse, regularized, return_predictions = (True, False, True)

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
    # 'gpu_processes': [0, 2],
    'gpu_processes': [],
    'mu': mu,
    'tau_range': tau_range,
    'lambda_range': lambda_range,
    'data_normalizer': data_normalizer,
    'labels_normalizer': labels_normalizer,
    'cv_error': cv_error,
    'error': error,
    'sparse': sparse,
    'regularized': regularized,
    'return_predictions': return_predictions
}

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75
