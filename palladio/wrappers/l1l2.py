"""Wrapper for l1l2."""
import numpy as np

import l1l2py
from l1l2py.algorithms import ridge_regression, l1l2_regularization

from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d

# from palladio import utils as pd_utils
# from palladio.wrappers import Classification

# class RangesScaler(object):
#     """Given data and labels helps to scale L1L2 parameters ranges properly.
#
#     This class works on tau and mu ranges passed to the l1l2 selection
#     framework (see also :func:`l1l2py.model_selection` and related
#     function for details).
#
#     Scaling ranges permits to use relative (and not absolute) ranges of
#     parameters.
#
#     Attributes
#     ----------
#     norm_data : :class:`numpy.ndarray`
#         Normalized data matrix.
#     norm_labels : :class:`numpy.ndarray`
#         Normalized labels vector.
#     """
#
#     def __init__(self, data, labels, data_normalizer=None,
#                  labels_normalizer=None):
#
#         self.norm_data = data
#         self.norm_labels = labels
#         self._tsf = self._msf = None
#
#         # Data must be normalized
#         if data_normalizer:
#             self.norm_data = data_normalizer(self.norm_data)
#         if labels_normalizer:
#             self.norm_labels = labels_normalizer(self.norm_labels)
#
#     def tau_range(self, trange):
#         """Return a scaled tau range.
#
#         Tau scaling factor is the maximum tau value to avoid and empty solution
#         (where all variables are discarded).
#         The value is estimated on the maximum correlation between data and
#         labels.
#
#         Parameters
#         ----------
#         trange : :class:`numpy.ndarray`
#             Tau range containing relative values (expected maximum is lesser
#             than 1.0 and minimum greater than 0.0).
#
#         Returns
#         -------
#         tau_range : :class:`numpy.ndarray`
#             Scaled tau range.
#
#         Raises
#         ------
#         PDException
#             If trange values are not in the [0, 1) interval
#             (right extreme excluded).
#         """
#         trange = np.sort(trange)
#
#         if max(trange) >= 1.0 or min(trange) < 0.0:
#             raise PDException('relative tau values have to '
#                               'be in [0, 1)')
#
#         return trange * self.tau_scaling_factor
#
#     def mu_range(self, mrange):
#         """Return a scaled mu range.
#
#         Mu scaling factor is estimated on the maximum eigenvalue of the
#         correlation matrix and is used to simplify the parameters choice.
#
#         Parameters
#         ----------
#         mrange : :class:`numpy.ndarray`
#             Mu range containing relative values (expected maximum is lesser
#             than 1.0 and minimum greater than 0.0).
#
#         Returns
#         -------
#         mu_range : :class:`numpy.ndarray`
#             Scaled mu range.
#
#         Raises
#         ------
#         PDException
#             If mrange values are not all greater than 0.
#         """
#         mrange = np.sort(mrange)
#
#         if min(mrange) < 0.0:
#             raise PDException('relative mu values have to be '
#                               'greater than 0')
#
#         return np.sort(mrange) * self.mu_scaling_factor
#
#     @property
#     def tau_scaling_factor(self):
#         """Tau scaling factor calculated on given data and labels."""
#         if self._tsf is None:
#             self._tsf = self._tau_scaling_factor()
#         return self._tsf
#
#     @property
#     def mu_scaling_factor(self):
#         """Mu scaling factor calculated on given data matrix."""
#         if self._msf is None:
#             self._msf = self._mu_scaling_factor()
#         return self._msf
#
#     def _tau_scaling_factor(self):
#         return l1l2py.algorithms.l1_bound(self.norm_data, self.norm_labels)
#
#     def _mu_scaling_factor(self):
#         n, d = self.norm_data.shape
#
#         if d > n:
#             tmp = np.dot(self.norm_data, self.norm_data.T)
#             num = np.linalg.eigvalsh(tmp).max()
#         else:
#             tmp = np.dot(self.norm_data.T, self.norm_data)
#             evals = np.linalg.eigvalsh(tmp)
#             num = evals.max() + evals.min()
#
#         return (num / (2. * n))

class l1l2Classifier(LinearClassifierMixin):

    # def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
    #              normalize=False, precompute=False, max_iter=1000,
    #              copy_X=True, tol=1e-4, warm_start=False, positive=False,
    #              random_state=None, selection='cyclic'):
    def __init__(self, mu=1e-4, tau=1.0, lambda_ = 1e4, fit_intercept=False
                 ):

        self.mu = mu
        self.tau = tau
        self.lambda_ = lambda_

        self.coef_ = None
        self.fit_intercept = fit_intercept
        # self.normalize = normalize
        # self.precompute = precompute
        # self.max_iter = max_iter
        # self.copy_X = copy_X
        # self.tol = tol
        # self.warm_start = warm_start
        # self.positive = positive
        self.intercept_ = 0.0
        # self.random_state = random_state
        # self.selection = selection


    def fit(self, X, y, check_input=True):



        # rs = RangesScaler(Xtr, Ytr,
        #                                    self._params['data_normalizer'],
        #                                    self._params['labels_normalizer'])
        #
        #         self._tau_range = rs.tau_range(self._params['tau_range'])
        #         self._mu_range = rs.mu_range(np.array([self._params['mu']]))
        #         self._lambda_range = np.sort(self._params['lambda_range'])







        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            y = column_or_1d(y, warn=True)
        else:
            # we don't (yet) support multi-label classification in ENet
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))


        self.coef_ = l1l2_regularization(X, y, self.mu, self.tau, beta=None, kmax=100000,
                                tolerance=1e-5, return_iterations=False,
                                adaptive=False)

        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
        self.coef_ = self.coef_.reshape(ndim, -1)

        ### TODO check this ravel
        self.coef_ = self.coef_.ravel()



        return 6

    @property
    def classes_(self):
        return self._label_binarizer.classes_

#
# class l1l2Classifier(Classification):
#     """Select features using l1l2."""
#
#     def __init__(self, params):
#         super(l1l2Classifier, self).__init__(params)
#         self.param_names = [r'\tau', r'\lambda']
#
#     def setup(self, Xtr, Ytr, Xts, Yts):
#         super(l1l2Classifier, self).setup(Xtr, Ytr, Xts, Yts)
#
#         rs = pd_utils.RangesScaler(Xtr, Ytr,
#                                    self._params['data_normalizer'],
#                                    self._params['labels_normalizer'])
#
#         self._tau_range = rs.tau_range(self._params['tau_range'])
#         self._mu_range = rs.mu_range(np.array([self._params['mu']]))
#         self._lambda_range = np.sort(self._params['lambda_range'])
#
#         # Determine which version of the algorithm must be used (CPU or GPU)
#         # based on the process rank and configuration settings
#         if self.get_param('process_rank') in self.get_param('gpu_processes'):
#             self._algorithm_version = 'GPU'
#         else:
#             self._algorithm_version = 'CPU'
#
#     def get_algorithm_version(self):
#         return self._algorithm_version
#
#     def run(self):
#         # Execution
#         result = l1l2py.model_selection(
#             self._Xtr, self._Ytr, self._Xts, self._Yts,
#             self._mu_range, self._tau_range, self._lambda_range,
#             self.get_param('ms_split'), self.get_param(
#                 'cv_error'), self.get_param('error'),
#             data_normalizer=self.get_param('data_normalizer'),
#             labels_normalizer=self.get_param('labels_normalizer'),
#             sparse=self.get_param('sparse'),
#             regularized=self.get_param('regularized'),
#             return_predictions=self.get_param('return_predictions'),
#             algorithm_version=self.get_algorithm_version()
#         )
#
#         # Return only the first element of the list, which is the one related
#         # to the smallest value of mu
#         result['selected_list'] = result['selected_list'][0]
#         result['beta_list'] = result['beta_list'][0]
#
#         result['prediction_ts_list'] = result['prediction_ts_list'][0].ravel()
#         if 'prediction_tr_list' in result.keys():
#             result['prediction_tr_list'] = result[
#                 'prediction_tr_list'][0].ravel()
#
#         return result
