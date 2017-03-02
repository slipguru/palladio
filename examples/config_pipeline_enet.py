# Configuration file example for PALLADIO
# version: 2.0

import numpy as np

from sklearn.linear_model import RidgeClassifier
from palladio.wrappers import ElasticNetFeatureSelection
from sklearn.pipeline import Pipeline

from palladio.datasets import DatasetCSV as dataset_class
# from palladio.datasets import DatasetNPY as dataset_class

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

# # The list of all files required for the experiments
# dataset_files = {
#     'data': 'data/iris_data.npy',
#     'labels': 'data/iris_labels.npy',
#     'indcol': 'data/iris_indcols.pkl'
# }
#
# dataset_options = {
#     'samples_on': 'row',  # or 'row': samples on cols or rows
#     'positive_label': 1,  # positive class in case of 2-class task
# }

#######################
#   SESSION OPTIONS ###
#######################

result_path = 'palladio_test_golub_pipeline_enet'

# The number of "regular" experiment
N_jobs_regular = 50

# The number of instances for the permutation tests
# (labels in the training sets are randomly shuffled)
N_jobs_permutation = 50

# The ratio of the dataset held out for model assessment
# It should be of the form 1/M
test_set_ratio = float(1) / 4

#######################
#  LEARNER OPTIONS  ###
#######################

# ### PIPELINE ###

# ### STEP 1: VARIABLE SELECTION
vs = ElasticNetFeatureSelection()

# ### STEP 2: CLASSIFICATION VIA RIDGE REGRESSION
clf = RidgeClassifier()

# ### COMPOSE THE PIPELINE
pipe = Pipeline([
        ('enet_vs', vs),
        ('ridge_clf', clf),
        ])

# ### Set the estimator to be the pipeline
estimator = pipe

# ### Parameter grid for both steps
param_grid = {
    'enet_vs__l1_ratio': [0.1],
    'enet_vs__alpha': np.logspace(2, 4, 5),
    'ridge_clf__alpha': np.logspace(-4, 2, 5),
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
vs_analysis = 'enet_vs'

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = 0.75
