"""Feature selection and classification by linear two steps model."""

import numpy as np
from sklearn.linear_model import ElasticNet  # feature selection
from sklearn.linear_model import RidgeClassifier  # overshrink prevention

from .classification import Classification
from l1l2signature import utils as l1l2_utils

# Legacy import
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV


class LinearTwoStep(Classification):
    """Feature selection and learning combined.

    [step 1]: FEATURE SELECTION
    The feature selection step is obtained minimizing the Elastic-Net objective
    function:

    1 / (2 * n_samples) * ||y - Xw||^2_2 +
    + alpha * l1_ratio * ||w||_1 +
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    specifically, if l1_ratio = 1 this is equivalent to the Lasso functional,
    while if alpha = 0 no feature selection is performed and all features are
    passed to the second step.

    [step 2]: LEARNING
    The final model is given training a Ridge classifier on the restricted set
    of features identified at the previous step. The Ridge functional is:

    1 / (2 * n_samples) * ||y - Xw||^2_2 + lambda * ||w||^2_2

    specifically, when alpha = 0 this is equivalent to the ordinary least
    squares solution.
    """

    def setup(self, Xtr, Ytr, Xts, Yts):
        super(Classification, self).setup(Xtr, Ytr, Xts, Yts)

        rs = l1l2_utils.RangesScaler(Xtr, Ytr,
                                     self._params['data_normalizer'],
                                     self._params['labels_normalizer'])

        self._tau_range = rs.tau_range(self._params['tau_range'])
        self._mu_range = rs.mu_range(np.array([self._params['mu']]))

        # The lambda range implies using the l1l2py objective function, so we
        # need to rescale it to:
        self._lambda_range = np.sort(self._params['lambda_range'])

    def run(self):
        mdl = LinearTwoStepTransformer(scoring=self._params['cv_error'],
                                       data_normalizer=self._params['data_normalizer'],
                                       labels_normalizer=self._params['labels_normalizer'])
        clf = GridSearchCV(mdl, param_grid={'tau': self._tau_range,
                                            'mu': self._mu_range,
                                            'lambda': self._lambda_range},
                           cv=self._params['internal_k'])

        clf.fit(self._Xtr, self._Ytr)

class LinearTwoStepTransformer(object):
    """An sklearn-compliant transformer class.

    This transformer class simply wraps the LinearTwoStep classifier.
    """
    def __init__(self, tau, mu, lam, scoring=None, data_normalizer=None,
                 labels_normalizer=None):
        self._coef = None
        self._tau = tau
        self._mu = mu
        self._lambda = lam
        self.data_normalizer = data_normalizer
        self.labels_normalizer = labels_normalizer

    def _to_l1_ratio(tau, mu):
        """Get l1_ratio on-the-fly."""
        return tau / (2.0 * mu) + tau

    def _to_alpha(tau, mu):
        """Get alpha on-the-fly."""
        return mu + 0.5 * tau

    def _to_ridge_alpha(lam):
        """Get alpha on-the-fly."""
        return 0.5 * lam

    def fit(self, X, y):
        # Normalize data (if necessary)
        if self.data_normalizer is not None:
            X = self.data_normalizer(X)

        if self.labels_normalizer is not None:
            y = self.labels_normalizer(y)

        # Transform tau/mu to li_ratio/alpha
        l1_ratio = self._to_l1_ratio(self._tau, self._mu)
        alpha = self._to_alpha(self._tau, self._mu)

        # Perform feature selection
        fs = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
        fs.fit(X, y)
        sel_idx = np.nonzero(fs.coef_)

        # Transoform lambda to alpha
        alpha = self._to_ridge_alpha(self._lambda)

        # Learn model on selected features
        mdl = RidgeClassifier(alpha=alpha)
        mdl.fit(X[:, sel_idx], y)

        # Save wheights
        self.coef_ = mdl.coef_

    def predict(self, X):
        return np.dot(X, self._coef)
