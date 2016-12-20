"""Wrapper for Group Lasso."""
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

from lightning.classification import FistaClassifier

import l1l2py
from palladio import utils as pd_utils
from palladio.wrappers.classification import Classification


class GroupLassoClassifier(Classification):
    """GroupLasso classifier.

    The optimization objective for Group Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """

    def __init__(self, params):
        """TODO."""
        self._params = params
        self.param_names = [r'\tau', r'\lambda']

    # def setup(self, Xtr, Ytr, Xts, Yts):
    #     """Setup for the lasso_olsClassifier."""
    #     Classification.setup(self, Xtr, Ytr, Xts, Yts)

    def setup(self, Xtr, Ytr, Xts, Yts):

        self._Xtr = Xtr
        self._Ytr = Ytr

        self._Xts = Xts
        self._Yts = Yts

        rs = pd_utils.RangesScaler(Xtr, Ytr,
                                   self._params['data_normalizer'],
                                   self._params['labels_normalizer'])

        self._tau_range = rs.tau_range(self._params['tau_range'])
        self._mu_range = rs.mu_range(np.array([self._params['mu']]))
        self._lambda_range = np.sort(self._params['lambda_range'])

        # Determine which version of the algorithm must be used (CPU or GPU)
        # based on the process rank and configuration settings
        if self.get_param('process_rank') in self.get_param('gpu_processes'):
            self._algorithm_version = 'GPU'
        else:
            self._algorithm_version = 'CPU'

    def get_l1_bound(self):
        r"""Estimation of an useful maximum bound for the `l1` penalty term.

        For each value of ``tau`` smaller than the maximum bound the solution
        vector contains at least one non zero element.

        .. warning

            That is, bounds are right if you run the `l1l2` regularization
            algorithm with the same data matrices.

        Parameters
        ----------
        data : (N, P) ndarray
            Data matrix.
        labels : (N,)  or (N, 1) ndarray
            Labels vector.

        Returns
        -------
        tau_max : float
            Maximum ``tau``.
        """
        data = self._Xtr
        labels = self._Ytr
        corr = np.abs(np.dot(data.T, labels))
        tau_max = (corr.max() * (2.0 / data.shape[0]))
        return tau_max

    def run(self):
        """Perform run."""
        if self._params['data_normalizer'] == l1l2py.tools.center:
            out = self._params['data_normalizer'](self._Xtr, self._Xts, True)
            self._Xtr = out[0]
            self._Xts = out[1]

        if self._params['labels_normalizer'] == l1l2py.tools.center:
            out = self._params['labels_normalizer'](self._Ytr, self._Yts, True)
            self._Ytr = out[0]
            self._Yts = out[1]

        # Model selection phase
        internal_k = self._params['internal_k']

        TAU_MAX = self.get_l1_bound()

        clf = FistaClassifier(penalty='l1/l2')
        gs = GridSearchCV(clf, {'alpha': self._params['tau_range'] * TAU_MAX})
        gs.fit(self._Xtr, self._Ytr)  # fit the model

        # extract only nonzero coefficients
        coefs = gs.best_estimator_.coef_
        selected_features = np.argwhere(coefs).ravel()

        # predict test
        Y_pred_ts = gs.best_estimator_.predict(self._Xts)
        Y_pred_ts = gs.best_estimator_.predict(self._Xtr)

        # Get performance
        err_fun = self._params['error']  # config error function
        ts_err = err_fun(self._Yts, Y_pred_ts)
        # tr_err = err_fun(self._Ytr, Y_pred_tr)
        tr_err = np.min(gs.cv_results_['mean_train_score'])

        result = {}
        result['selected_list'] = selected_features
        # result['beta_list'] = result['beta_list'][0]
        result['prediction_ts_list'] = Y_pred_ts
        result['prediction_tr_list'] = Y_pred_ts
        result['labels_ts'] = self._Yts

        result['beta_list'] = clf.coef_.tolist()
        result['err_ts_list'] = ts_err
        result['err_tr_list'] = tr_err
        result['kcv_err_ts'] = np.mean(clf.mse_path_, axis=2)
        # TODO: define a policy for the training error
        result['kcv_err_tr'] = np.zeros((len(self.l1_ratio_range),
                                         len(self.alpha_range)))
        return result
