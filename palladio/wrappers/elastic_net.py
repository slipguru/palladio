# -*- coding: UTF-8 -*-
"""Feature selection and classification by linear two steps model."""
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

import l1l2py
from palladio import utils as pd_utils
from palladio.wrappers import Classification


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
        self.alpha_range = self._params['alpha_range']  # * rs.mu_scaling_factor
        # print(rs.mu_scaling_factor)

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

        estimator = ElasticNet(normalize=_normalize, fit_intercept=False)
        param_grid = {
            'alpha': self.alpha_range,
            'l1_ratio': self.l1_ratio_range
        }
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid,
                          cv=self._params['internal_k'], n_jobs=-1)
        gs.fit(self._Xtr, self._Ytr)

        # Evaluate prediction on test set
        clf = gs.best_estimator_
        Y_pred_ts = clf.predict(self._Xts)
        Y_pred_tr = clf.predict(self._Xtr)

        # Get performance
        err_fun = self._params['error']  # config error function
        ts_err = err_fun(self._Yts, Y_pred_ts)
        tr_err = err_fun(self._Ytr, Y_pred_tr)

        # Save results
        result = dict()
        result['selected_list'] = np.nonzero(clf.coef_)[0].tolist()
        result['beta_list'] = clf.coef_.tolist()
        result['prediction_ts_list'] = Y_pred_ts
        result['prediction_tr_list'] = Y_pred_tr
        result['err_ts_list'] = ts_err
        result['err_tr_list'] = tr_err

        result['kcv_err_ts'] = gs.cv_results_['mean_test_score']
        result['kcv_err_tr'] = gs.cv_results_['mean_train_score']
        result['best_params'] = gs.best_params_
        return result
