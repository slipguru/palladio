"""Feature selection and classification by linear two steps model."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet  # feature selection
from sklearn.linear_model import RidgeClassifier  # overshrink prevention

from palladio.wrappers.classification import Classification
from l1l2signature import utils as l1l2_utils

# Legacy import
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV


class LinearTwoStepClassifier(Classification):
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
        super(LinearTwoStepClassifier, self).setup(Xtr, Ytr, Xts, Yts)

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
                                            'lam': self._lambda_range},
                           cv=self._params['internal_k'])

        clf.fit(self._Xtr, self._Ytr)

        # Evaluate prediction on test set
        Y_pred = clf.best_estimator_.predict(self._Xts)
        Y_pred_tr = clf.best_estimator_.predict(self._Xtr)

        # Get performance
        err_fun = self._params['error']  # config error function
        ts_err = err_fun(self._Yts, Y_pred)
        tr_err = err_fun(self._Ytr, Y_pred_tr)

        # Save results
        result = dict()
        result['selected_list'] = np.nonzero(clf.best_estimator_.coef_)[0].tolist()
        result['beta_list'] = clf.best_estimator_.coef_.tolist()
        result['prediction_ts_list'] = Y_pred
        result['prediction_tr_list'] = Y_pred_tr
        result['err_ts_list'] = ts_err
        result['err_tr_list'] = tr_err
        # TODO: save the kcv_err_ts and (if possible) training
        # result['kcv_err_ts'] = 

        return result


class LinearTwoStepTransformer(BaseEstimator):
    """An sklearn-compliant transformer class.

    This transformer class simply wraps the LinearTwoStep classifier.
    """
    def __init__(self, tau=None, mu=None, lam=None,
                 scoring=None, data_normalizer=None,
                 labels_normalizer=None):
        self.coef_ = None
        self._tau = tau  # dummy
        self._mu = mu  # dummy
        self._lambda = lam  # dummy
        self.scoring = scoring
        self.data_normalizer = data_normalizer
        self.labels_normalizer = labels_normalizer

    def score(self, X, y):
        """Scorer wrapper for scoring function."""
        # Predict labels
        pred_y = self.predict(X)
        real_y = y

        if self.scoring is not None:
            return self.scoring(real_y, pred_y)
        else:
            from sklearn.metrics.regression import mean_squared_error
            print("No scoring function passed: using mean squared error.")
            return mean_squared_error(real_y, pred_y)

    def _to_l1_ratio(self, tau, mu):
        """Get l1_ratio on-the-fly."""
        return tau / (2.0 * mu) + tau

    def _to_alpha(self, tau, mu):
        """Get alpha on-the-fly."""
        return mu + 0.5 * tau

    def _to_ridge_alpha(self, lam):
        """Get alpha on-the-fly."""
        return 0.5 * lam

    def fit(self, X, y):
        # Normalize data (if necessary)
        if self.data_normalizer is not None:
            out = self.data_normalizer(X, None, True)
            X = out[0]  # normalized training data matrix
            self.data_norm_factors = out[1:]  # normalization factor: mu

        if self.labels_normalizer is not None:
            out = self.labels_normalizer(y, None, True)
            y = out[0]  # normalized training labels vector
            self.labels_norm_factors = out[1:]  # normalization factors: mu std

        # Transform tau/mu to li_ratio/alpha
        l1_ratio = self._to_l1_ratio(self._tau, self._mu)
        alpha = self._to_alpha(self._tau, self._mu)

        # Perform feature selection
        fs = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
        fs.fit(X, y)

        sel_idx = np.nonzero(fs.coef_)
        # check for empty solutions
        if len(sel_idx[0]) != 0:
            # Transoform lambda to alpha
            alpha = self._to_ridge_alpha(self._lambda)

            # Learn model on selected features
            mdl = RidgeClassifier(alpha=alpha)
            mdl.fit(X[:, sel_idx], y)

            # Save wheights
            self.coef_ = mdl.coef_
        else:
            # Empty solution
            self.coef_ = fs.coef_

    def predict(self, X):
        if len(self.data_norm_factors) > 1:
            # trainig data are standardized
            return np.dot((X - self.data_norm_factors[0]) /
                          self.data_norm_factors[1], self.coef_)
        else:
            # trainig data are recentered
            return np.dot(X - self.data_norm_factors, self.coef_)

    def get_params(self, deep=None):  # the keyword argument deep is unused
        return {'tau': self._tau, 'mu': self._mu, 'lam': self._lambda,
                'scoring': self.scoring,
                'data_normalizer': self.data_normalizer,
                'labels_normalizer': self.labels_normalizer}

    def set_params(self, **kwargs):
        self._tau = kwargs['tau']
        self._mu = kwargs['mu']
        self._lambda = kwargs['lam']
        return self
