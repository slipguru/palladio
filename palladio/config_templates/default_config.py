# Configuration file example for PALLADIO
# version: 2.0

import numpy as np

from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from palladio import datasets

#####################
#   DATASET PATHS ###
#####################

# * All the path are w.r.t. config file path

# The list of all files required for the experiments
data_path = 'data.csv'
target_path = 'labels.csv'

# pandas.read_csv options
data_loading_options = {
    'delimiter': ',',
    'header': 0,
    'index_col': 0
}
target_loading_options = data_loading_options

dataset = datasets.load_csv(data_path, target_path,
                            data_loading_options=data_loading_options,
                            target_loading_options=target_loading_options,
                            samples_on='row')
data, labels = dataset.data, dataset.target

#######################
#   SESSION OPTIONS ###
#######################

result_path = 'results'

# The learning task, if None palladio tries to guess it
# [see sklearn.utils.multiclass.type_of_target]
learning_task = None

#######################
#  LEARNER OPTIONS  ###
#######################

# ### PIPELINE ###

# ### STEP 0: Preprocessing
pp = MinMaxScaler(feature_range=(0, 1))

# ### STEP 1: Variable selection
vs = RFE(LinearSVC(loss='hinge'), step=0.3)

# ### STEP 2: Classification
clf = LinearSVC(loss='hinge')

# ### COMPOSE THE PIPELINE
pipe = Pipeline([
	('preprocessing', pp),
        ('variable_selection', vs),
        ('classification', clf),
        ])


# ### Set the estimator to be assessed a GridSearchCV
param_grid = {
    'variable_selection__n_features_to_select': [10, 20, 50],
    'variable_selection__estimator__C': np.logspace(-4, 0, 5),
    'classification__C': np.logspace(-4, 0, 5),
}

estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring='accuracy')

# ### Set options for ModelAssessment
ma_options = {
    'test_size': 0.25,
    'scoring': 'accuracy',
    'n_jobs': -1,
    'n_splits': 2
}

n_splits_permutation = 3

# For the Pipeline object, indicate the name of the step from which to
# retrieve the list of selected features
# For a single estimator which has a `coef_` attributes (e.g., elastic net or
# lasso) set to True
vs_analysis = 'variable_selection'

# ~~ Signature Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
frequency_threshold = None

# ~~ Plotting Options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logspace = ['variable_selection__C']
plot_errors = True
