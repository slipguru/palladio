"""Wrapper for sklearn Linear Regression."""
import numpy as np

from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import SGDRegressor

# from palladio.preprocessing import Center
from palladio.wrappers.classification import Classification


class OLSClassification(Classification):
    """OLS classification class."""

    def setup(self, Xtr, Ytr, Xts, Yts):
        """Setup for the OLSClassification.

        Parameters
        ----------
        Xtr, Ytr, Xts, Yts : ndarray
            Training samples and labels, test samples and labels.
        """
        Classification.setup(self, Xtr, Ytr, Xts, Yts)

        # Is this mandatory?
        # self._data_normalizer = Center(Xtr)
        # self._labels_normalizer = Center(Ytr)

        # self._clf = LinearRegression(fit_intercept = False)
        self._clf = LinearRegression(fit_intercept=True)


    def run(self):
        n, p = self._Xtr.shape

        # Fit the model, possibly normalizing data and labels
        # self._clf.fit(
        #     self._data_normalizer.center(self._Xtr),
        #     # self._labels_normalizer.center(self._Ytr)
        #     self._Ytr
        # )
        #
        # prediction_ts_list = self._clf.predict(
        #     self._data_normalizer.center(self._Xts)
        # )
        #
        # prediction_tr_list = self._clf.predict(
        #     self._data_normalizer.center(self._Xtr)
        # )

        # Fit the model, possibly normalizing data and labels
        self._clf.fit(self._Xtr, self._Ytr)

        prediction_ts_list = self._clf.predict(
            self._Xts
        )

        prediction_tr_list = self._clf.predict(
            self._Xtr
        )

        beta = self._clf.coef_

        result = {}
        result['selected_list'] = np.ones(p, dtype=bool)
        result['model'] = beta
        result['intercept'] = self._clf.intercept_

        result['prediction_tr_list'] = np.sign(prediction_tr_list)
        result['prediction_ts_list'] = np.sign(prediction_ts_list)
        return result
