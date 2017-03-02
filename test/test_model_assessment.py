"""Some testing."""
import warnings

from nose import with_setup
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from palladio.wrappers import ElasticNetFeatureSelection
from palladio.model_assessment import ModelAssessment


def setup_function():
    warnings.simplefilter('ignore', category=UserWarning)


@with_setup(setup_function)
def test_model_assessment():
    X, y = make_classification(n_samples=40, n_features=100, n_informative=2,
                               n_classes=2, n_redundant=0)
    pipe = Pipeline([('enet', ElasticNetFeatureSelection()),
                     ('ridge', RidgeClassifier())])

    ma = ModelAssessment(GridSearchCV(pipe, {'enet__l1_ratio': [2]})).fit(X, y)
    assert len(ma.cv_results_) == 0
