# -*- coding: UTF-8 -*-
"""Feature selection and classification by linear two steps model."""
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import l1l2py
from palladio import utils as pd_utils
from palladio.wrappers import Classification


# class GenericClassifier(object):
#     def __init__(self, estimator=None, *args, **kwargs):
#
#         self._estimator = type(estimator)(*args, **kwargs)
#
#         self.score = make_scorer(
#             l1l2py.tools.balanced_classification_error,
#             greater_is_better=False)
#
#     def __getattr__(self, name):
#         attr = getattr(self._estimator, name)
#         if not callable(attr):
#             return attr
#
#         def handlerFunction(*args, **kwargs):
#             # print("calling", name,args,kwargs)
#             return attr(*args, **kwargs)
#         return handlerFunction
#
#     def predict(self, *args, **kwargs):
#         y_pred = self._estimator.predict(*args, **kwargs)
#         return y_pred

class GenericClassifier():
    def __init__(self, *args, **kwargs):
        super(GenericClassifier, self).__init__(*args, **kwargs)

        self.score = make_scorer(
            l1l2py.tools.balanced_classification_error,
            greater_is_better=False)

    def predict(self, *args, **kwargs):
        y_pred = self._estimator.predict(*args, **kwargs)
        return y_pred


def make_classifier(estimator):
    """Make a classifier for a possible regressor.

    Parameters
    ----------
    estimator : sklearn-like class
        It must contain at least a fit and predict method.
    """
    # return GenericClassifier(estimator)
    args = dict(estimator.__dict__)
    # args['score'] = make_scorer(
    #     l1l2py.tools.balanced_classification_error, greater_is_better=False)
    from sklearn.base import ClassifierMixin
    return type('GenericClassifier', (type(estimator), ClassifierMixin), args)


class PipelineClassifier(Classification):
    """General pipeline for sklearn-classifiers."""

    def __init__(self, clf, params=None, param_names=None):
        super(PipelineClassifier, self).__init__(params)
        self.clf = clf
        if param_names is not None:
            self.param_names = param_names

    def setup(self, Xtr, Ytr, Xts, Yts):
        super(PipelineClassifier, self).setup(Xtr, Ytr, Xts, Yts)

    def normalize():
        raise NotImplementedError()
        # if self._params['data_normalizer'] == l1l2py.tools.center:
        #     out = self._params['data_normalizer'](self._Xtr, self._Xts, True)
        #     self._Xtr = out[0]
        #     self._Xts = out[1]
        #
        #     if self._params['labels_normalizer'] == l1l2py.tools.center:
        #         out = self._params['labels_normalizer'](self._Ytr, self._Yts, True)
        #         self._Ytr = out[0]
        #         self._Yts = out[1]

    def run(self):

        ### TODO Check if the data need to be normalized
        ### TODO These params must be
        clf = self.clf(**self._params)

        param_grid = {
            'alpha': self.alpha_range,
            'l1_ratio': self.l1_ratio_range
        }

        gs = GridSearchCV(estimator=clf, param_grid=param_grid,
                          cv=self._params['internal_k'], n_jobs=-1)
        gs.fit(self._Xtr, self._Ytr)

        # Evaluate prediction on test set
        clf = gs.best_estimator_
        Y_pred_ts = gs.predict(self._Xts)
        Y_pred_tr = gs.predict(self._Xtr)

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
        result['err_tr_list'] = tr_err  # learning error
        result['err_ts_list'] = ts_err  # test error

        result['kcv_err_tr'] = 1 - np.clip(
            gs.cv_results_['mean_train_score'], 0, 1)  # training score
        result['kcv_err_ts'] = 1 - np.clip(
            gs.cv_results_['mean_test_score'], 0, 1)  # validation score
        result['best_params'] = gs.best_params_
        return result
