# Configuration file example for PALLADIO
# version: '0.1.1

import l1l2py

from palladio.wrappers.l1l2 import l1l2Classifier

#~~ Data Input/Output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * Data assumed csv with samples and features labels
# * All the path are w.r.t. config file path
data_matrix = 'data/gedm.csv'
labels = 'data/labels.csv'
delimiter = ','
samples_on = 'col' # or 'row': samples on cols or rows
result_path = 'golub_palladio_test_l1l2'

# The number of "regular" experiment
N_jobs_regular = 100

# The number of instances for the permutation tests
# (labels in the test sets are randomly shuffled)
N_jobs_permutation = 100

### The ratio of the dataset held out for model assessment
### It should be of the form 1/M
test_set_ratio = float(1)/4

#~~ Data filtering/normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_normalizer = l1l2py.tools.center
labels_normalizer = None

data_preprocessing = None

#~~ Cross validation options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
internal_k = 3
cv_splitting = l1l2py.tools.stratified_kfold_splits

#~~ Errors functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * See l1l2py.tools.{regression_error, classification_error,
#                     balanced_classification_error}
cv_error = l1l2py.tools.regression_error
error = l1l2py.tools.balanced_classification_error
positive_label = None # Indicates the positive class in case of 2-class task

#~~ L1l2 Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * Ranges will be sorted from smaller to bigger value!
# * See l1l2py.tools.{geometric_range, linear_range}
tau_range = l1l2py.tools.geometric_range(1e-3, 0.5, 20) # * MAX_TAU
mu_range = l1l2py.tools.geometric_range(1e-3, 1.0, 3)   # * CORRELATION_FACTOR
lambda_range = l1l2py.tools.geometric_range(1e0, 1e4, 10)

sparse, regularized, return_predictions = (True, False, True)

learner_class = l1l2Classifier

params = {
    'mu_range' : mu_range,
    'tau_range' : tau_range,
    'lambda_range' : lambda_range,
    'data_normalizer' : data_normalizer,
    'cv_error' : cv_error,
    'error' : error,
    'labels_normalizer' : labels_normalizer,
    'sparse' : sparse,
    'regularized' : regularized,
    'return_predictions' : return_predictions
}

#~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75
