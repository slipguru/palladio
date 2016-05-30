# Configuration file example for Lasso/OLS

import numpy as np
from sklearn.cross_validation import StratifiedKFold

from palladio.wrappers.lasso_ols import lasso_olsClassifier

#~~ Data Input/Output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * Data assumed csv with samples and features labels
# * All the path are w.r.t. config file path
data_matrix = 'data/gedm.csv'
labels = 'data/labels'
delimiter = ','
samples_on = 'col' # or 'row': samples on cols or rows
result_path = 'golub_palladio_test_lasso_ols'

N_jobs_regular = 100 # The number of "regular" experiment (actual labels used)
N_jobs_permutation = 100 # The number of instances for the permutation tests (labels are randomly shuffled)

### The ratio of the dataset held out for model assessment
### It should be of the form 1/B
test_set_ratio = float(1)/4

positive_label = None

data_preprocessing = None

#~~ Cross validation options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
internal_k = 3
cv_splitting = StratifiedKFold

#~~ L1 regularization parameter
tau_range = 0.75*np.logspace(-3, 0, num=20)

learner_class = lasso_olsClassifier

params = {
    'tau_range' : tau_range,
    'internal_k' : internal_k
}

#~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75