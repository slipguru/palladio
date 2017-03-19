# Configuration file example for PALLADIO
# version: 2.0

import numpy as np

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from palladio import datasets

import os

#####################
#   DATASET PATHS ###
#####################

# * All the path are w.r.t. config file path

# The list of all files required for the experiments

data_path = 'data/binclass_data.npy'
target_path = 'data/binclass_target.npy'

try:
    dataset = datasets.load_npy(os.path.join(os.path.dirname(__file__),data_path),
                                os.path.join(os.path.dirname(__file__),target_path),
                                samples_on='row')
except:
    dataset = datasets.load_npy(os.path.join(os.path.dirname(__file__), 'data'),
                                os.path.join(os.path.dirname(__file__), 'labels'),
                                samples_on='row')

data, labels = dataset.data, dataset.target
feature_names = dataset.feature_names

#######################
#   SESSION OPTIONS ###
#######################

session_folder = 'palladio_test_binclass'

# The learning task, if None palladio tries to guess it
# [see sklearn.utils.multiclass.type_of_target]
learning_task = None

# The number of repetitions of 'regular' experiments
n_splits_regular = 100

# The number of repetitions of 'permutation' experiments
n_splits_permutation = 100

#######################
#  LEARNER OPTIONS  ###
#######################

vs = RFE(LinearSVC(loss='hinge'), step=0.3)

param_grid = {
    'n_features_to_select': [10, 20, 50],
    'estimator__C': np.logspace(-4, 0, 5),
}

estimator = GridSearchCV(vs, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=1)

# Set options for ModelAssessment
ma_options = {
    'test_size': 0.25,
    'scoring': 'accuracy',
    'n_jobs': -1,
    'n_splits': n_splits_regular
}


# For the Pipeline object, indicate the name of the step from which to
# retrieve the list of selected features
# For a single estimator which has a `coef_` attributes (e.g., elastic net or
# lasso) set to True
vs_analysis = True

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = None

# ~~ Plotting Options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
score_surfaces_options = {
    'logspace': ['variable_selection__C'],
    'plot_errors': True
}
