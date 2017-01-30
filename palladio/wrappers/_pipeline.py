# -*- coding: UTF-8 -*-
"""Feature selection and classification by linear two steps model."""
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.base import ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d

from palladio.wrappers import Classification


def predict(self, *args, **kwargs):
    """Predict method for classifiers.

    It must be defined outside a function to be pickable.
    """
    y_pred = super(type(self), self).predict(*args, **kwargs)
    return np.sign(y_pred)


def make_classifier(estimator, params=None):
    """Make a classifier for a possible regressor.

    .. deprecated:: 0.5

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


class GridSearchCVClassifier(GridSearchCV, Classification):
    """General pipeline for sklearn-classifiers.

    .. deprecated:: 0.5
    """

    def __init__(self, learner, learner_options=None,
                 cv_options=None,
                 final_scoring='accuracy',
                 data_normalizer=None,
                 label_normalizer=None, force_classifier=False):
        super(GridSearchCVClassifier, self).__init__(
            learner_options=learner_options,
            cv_options=cv_options, final_scoring=final_scoring,
            data_normalizer=data_normalizer, label_normalizer=label_normalizer,
            force_classifier=force_classifier)
        self.learner = learner
        # if param_names is not None:
        #     self.param_names = param_names

    def setup(self, Xtr, Ytr, Xts, Yts):
        """Deprecated. Use fit predict instead."""
        super(GridSearchCVClassifier, self).setup(Xtr, Ytr, Xts, Yts)

    def normalize_data(self, X):
        raise NotImplementedError()

    def normalize_label(self, X):
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

    def fit(self, X, y=None):
        """Fitting function on the data."""
        if self.data_normalizer is not None:
            X = self.normalize_data(X)

        if self.label_normalizer is not None:
            y = self.normalize_label(y)

        if self.force_classifier:
            clf = make_classifier(self.learner, params=self.learner_options)
        elif callable(self.learner):
            # self.learner = type(self.learner)
            clf = self.learner(**self.learner_options)
        else:
            clf = self.learner

        self.gs_ = GridSearchCV(estimator=clf, **self.cv_options)
        self.gs_.fit(X, y)

    @property
    def cv_results_(self):
        """Get GridSearchCV results."""
        check_is_fitted(self, 'gs_')
        return self.gs_.cv_results_

    @property
    def best_params_(self):
        """Get GridSearchCV best_params."""
        check_is_fitted(self, 'gs_')
        return self.gs_.best_params_

    def predict(self, X):
        """Predicting function."""
        check_is_fitted(self, "gs_")
        return self.gs_.predict(X)

    def scoring(self, y_true, y_pred):
        """Score the result."""
        return self.final_scoring(y_true, y_pred)

    def run(self):
        """Deprecated. Run fit and predict instead."""
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
        Y_pred_ts = gs.predict(self._Xts)
        Y_pred_tr = gs.predict(self._Xtr)

        # Get performance
        tr_err = 1 - self.final_scoring(self._Ytr, Y_pred_tr)
        ts_err = 1 - self.final_scoring(self._Yts, Y_pred_ts)

        # Save results
        result = dict()
        clf = gs.best_estimator_
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
