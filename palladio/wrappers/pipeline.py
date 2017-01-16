# -*- coding: UTF-8 -*-
"""Feature selection and classification by linear two steps model."""
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


import l1l2py
from palladio import utils as pd_utils
from palladio.wrappers import Classification


def predict(self, *args, **kwargs):
    """Predict method for classifiers.

    It must be defined outside a function to be pickable.
    """
    y_pred = super(type(self), self).predict(*args, **kwargs)
    return np.sign(y_pred)


def make_classifier(estimator, params=None):
    """Make a classifier for a possible regressor.

    Parameters
    ----------
    estimator : sklearn-like class
        It must contain at least a fit and predict method.
    params : dict, optional
        Parameters of the classifier.

    Returns
    -------
    generic_classifier : class
        sklearn-like class that is a subclass of estimator. The predict method
        has been overwritten in order to return only the sign of the results.
        Note: this assumes that labels are 1 and -1.
    """
    if params is None:
        params = {}
    params['predict'] = predict
    params.setdefault('score', accuracy_score)
    return type('GenericClassifier', (estimator,), params)()


class ElasticNetClassifier(ElasticNet):
    """Class to extend elastic-net in case of classification.

    In case in which n_jobs != 1 for GridSearchCV, the estimator class must be
    pickable, therefore statically defined.
    """

    # def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
    #              normalize=False, precompute=False, max_iter=1000, copy_X=True,
    #              tol=0.0001, warm_start=False, positive=False,
    #              random_state=None, selection='cyclic', score=None):
    #     """Init for ElasticNetClassifier."""
    #     super(ElasticNetClassifier, self).__init__()
    predict = predict
    score = accuracy_score


class PipelineClassifier(Classification):
    """General pipeline for sklearn-classifiers."""

    def __init__(self, learner, learner_options=None,
                 cv_options=None,
                 final_scoring='accuracy',
                 data_normalizer=None,
                 label_normalizer=None, force_classifier=False):
        super(PipelineClassifier, self).__init__(
            learner_options=learner_options,
            cv_options=cv_options, final_scoring=final_scoring,
            data_normalizer=data_normalizer, label_normalizer=label_normalizer,
            force_classifier=force_classifier)
        self.learner = learner
        # if param_names is not None:
        #     self.param_names = param_names

    def setup(self, Xtr, Ytr, Xts, Yts):
        super(PipelineClassifier, self).setup(Xtr, Ytr, Xts, Yts)

    def normalize_data(self):
        raise NotImplementedError()

    def normalize_label(self):
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
        # TODO Check if the data need to be normalized
        if self.data_normalizer is not None:
            self.normalize_data()

        if self.label_normalizer is not None:
            self.normalize_label()

        if self.force_classifier:
            clf = make_classifier(self.learner, params=self.learner_options)
        else:
            clf = self.learner(**self.learner_options)

        gs = GridSearchCV(estimator=clf, **self.cv_options)
        gs.fit(self._Xtr, self._Ytr)

        # Evaluate prediction on test set
        clf = gs.best_estimator_
        Y_pred_ts = gs.predict(self._Xts)
        Y_pred_tr = gs.predict(self._Xtr)

        # Get performance
        ts_err = self.final_scoring(self._Yts, Y_pred_ts)
        tr_err = self.final_scoring(self._Ytr, Y_pred_tr)

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
