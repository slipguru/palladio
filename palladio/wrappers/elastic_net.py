# -*- coding: UTF-8 -*-
"""Feature selection and classification by linear two steps model."""

import numpy as np
from sklearn.linear_model import ElasticNetCV
# Legacy import
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV

import l1l2py
from palladio import utils as pd_utils
from palladio.wrappers.classification import Classification


class ElasticNetClassifier(Classification):
    """Feature selection and learning combined.

    Feature selection is obtained minimizing the Elastic-Net objective
    function:

    1 / (2 * n_samples) * ||y - Xw||^2_2 +
    + alpha * l1_ratio * ||w||_1 +
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    specifically, if l1_ratio = 1 this is equivalent to the Lasso functional,
    while if alpha = 0 no feature selection is performed and all features are
    passed to the second step.
    """

    def __init__(self, params=None):
        super(ElasticNetClassifier, self).__init__(params)
        self.param_names = [r'\alpha', r'l_1 ratio']  # BEWARE: use LaTeX names

    def setup(self, Xtr, Ytr, Xts, Yts):
        super(ElasticNetClassifier, self).setup(Xtr, Ytr, Xts, Yts)

        rs = pd_utils.RangesScaler(Xtr, Ytr,
                                   self._params['data_normalizer'],
                                   self._params['labels_normalizer'])

        self.l1_ratio_range = self._params['l1_ratio_range'][::-1]
        self.alpha_range = self._params['alpha_range'] * rs.mu_scaling_factor

    def run(self):
        # Check if the data neeed to be normalized
        _normalize = (self._params['data_normalizer'] ==
                      l1l2py.tools.standardize)

        if self._params['data_normalizer'] == l1l2py.tools.center:
            out = self._params['data_normalizer'](self._Xtr, self._Xts, True)
            self._Xtr = out[0]
            self._Xts = out[1]

        if self._params['labels_normalizer'] == l1l2py.tools.center:
            out = self._params['labels_normalizer'](self._Ytr, self._Yts, True)
            self._Ytr = out[0]
            self._Yts = out[1]

        clf = ElasticNetCV(l1_ratio=self.l1_ratio_range,
                           alphas=self.alpha_range,
                           cv=self._params['internal_k'],
                           normalize=_normalize,
                           fit_intercept=False,
                           n_jobs=-1)  # !!! THIS IS DANGEROUS !!!
        clf.fit(self._Xtr, self._Ytr)

        # Evaluate prediction on test set
        Y_pred = clf.predict(self._Xts)
        Y_pred_tr = clf.predict(self._Xtr)

        # Get performance
        err_fun = self._params['error']  # config error function
        ts_err = err_fun(self._Yts, Y_pred)
        tr_err = err_fun(self._Ytr, Y_pred_tr)

        # Save results
        result = dict()
        result['selected_list'] = np.nonzero(clf.coef_)[0].tolist()
        result['beta_list'] = clf.coef_.tolist()
        result['prediction_ts_list'] = Y_pred
        result['prediction_tr_list'] = Y_pred_tr
        result['err_ts_list'] = ts_err
        result['err_tr_list'] = tr_err
        result['kcv_err_ts'] = np.mean(clf.mse_path_, axis=2)
        # TODO: define a policy for the training error
        result['kcv_err_tr'] = np.zeros((len(self.l1_ratio_range),
                                         len(self.alpha_range)))
        return result
