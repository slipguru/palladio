# Configuration file example for L1L2Signature
# version: '0.2.2'

import l1l2py

from palladio.wrappers.l1l2 import l1l2Classifier

#~~ Data Input/Output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * Data assumed csv with samples and features labels
# * All the path are w.r.t. config file path
data_matrix = 'data/gedm.csv'
labels = 'data/Y_MTC_VS_CTRL.csv'
delimiter = ','
samples_on = 'col' # or 'row': samples on cols or rows
result_path = 'snp_mtc_experiments'

N_jobs_regular = 100 # The number of "regular" experiment (actual labels used)
N_jobs_permutation = 100 # The number of instances for the permutation tests (labels are randomly shuffled)

### The ratio of the dataset held out for model assessment
### It should be of the form 1/B
test_set_ratio = float(1)/4

#~~ Data filtering/normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sample_remover = None # removes samples with this label value
variable_remover = 'affx' # remove vars where name starts with (not case-sens.)
data_normalizer = l1l2py.tools.center
labels_normalizer = None

#~~ Cross validation options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * See l1l2py.tools.{kfold_splits, stratified_kfold_splits}
external_k = 4   # (None means Leave One Out)
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
frequency_threshold = 0.5
