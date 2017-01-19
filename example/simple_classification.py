"""NestedGridSearchCV example: fit a simple regression model."""
from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

from palladio.wrappers import ElasticNetClassifier
from palladio.model_assessment import ModelAssessment

data = load_breast_cancer()
X = data['data']
y = data['target'].ravel()

estimator = ElasticNetClassifier()
param_grid = {'alpha': np.logspace(-3, 3, 10)}

mca = ModelAssessment(GridSearchCV(estimator=estimator, param_grid=param_grid))
mca.fit(X, y)

if MPI.COMM_WORLD.Get_rank() == 0:
    print(mca.cv_results_)
