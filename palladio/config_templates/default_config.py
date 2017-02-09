# Configuration file example for PALLADIO
# version: 2.0

from palldio.wrappers import ElasticNetClassifier

#####################
#   DATASET PATHS ###
#####################

# * All the path are w.r.t. config file path

# The list of all files required for the experiments
dataset_files = {
    'data': 'data/gedm.csv',
    'labels': 'data/labels.csv'
}

dataset_options = {
    'positive_label': 'ALL',  # Indicates the positive class in case of 2-class task
    'samples_on': 'col',  # or 'row': samples on cols or rows
    # 'data_preprocessing' : None,

    # other options for pandas.read_csv
    'delimiter': ',',
    'header': 0,
    'index_col': 0
}

#######################
#   SESSION OPTIONS ###
#######################

result_path = 'palladio_default_results'

# The number of "regular" experiment
N_jobs_regular = 20

# The number of instances for the permutation tests
# (labels in the training sets are randomly shuffled)
N_jobs_permutation = 20

# The ratio of the dataset held out for model assessment
# It should be of the form 1/M
test_set_ratio = 0.25

#######################
#  LEARNER OPTIONS  ###
#######################

estimator = ElasticNetClassifier()

# ~~-Random Forest Classifier parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
param_grid = {
    'l1_ratio': [.1, .3, .6],
}

# ~~ Cross validation options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cv_options = {
    'param_grid': param_grid,
    'cv': 3,
    'scoring': 'accuracy',
}

final_scoring = 'accuracy'

# For the Pipeline object, indicate the name of the step from which to
# retrieve the list of selected features
# For a single estimator which has a `coef_` attributes (e.g., elastic net or
# lasso) set to True
vs_analysis = None

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75
