import numpy as np

from sklearn.linear_model import SGDClassifier, LinearRegression, SGDRegressor

from .classification import Classification

from ..preprocessing import Center

class OLSClassification(Classification):
    """
    
    """
    
    def setup(self, Xtr, Ytr, Xts, Yts):
        """
        Parameters
        ----------
        
        Xtr : ndarray
            The trainig samples
        """
        
        Classification.setup(self, Xtr, Ytr, Xts, Yts)
        
        ### Is this mandatory?
        self._data_normalizer = Center(Xtr)
        self._labels_normalizer = Center(Ytr)
        
        # self._clf = SGDClassifier(loss = 'squared_loss', penalty = 'none', n_iter = 500, verbose = 250)
        # self._clf = SGDRegressor(loss = 'squared_loss', penalty = 'none', n_iter = 500, verbose = 250)
        self._clf = LinearRegression(fit_intercept = False)
        
        
    def run(self):
        
        n, p = self._Xtr.shape
        
        ### Fit the model, possibly normalizing data and labels
        self._clf.fit(
            self._data_normalizer.center(self._Xtr),
            # self._labels_normalizer.center(self._Ytr)
            self._Ytr
        )
        
        prediction_ts_list = self._clf.predict(
            self._data_normalizer.center(self._Xts)
        )
        
        prediction_tr_list = self._clf.predict(
            self._data_normalizer.center(self._Xtr)
        )
        
        beta = self._clf.coef_
        
        result = {}
        
        selected_list = np.array(p*[True])
        result['selected_list'] = selected_list
        result['model'] = beta
        
        # result['prediction_tr_list'] = prediction_tr_list
        # result['prediction_ts_list'] = prediction_ts_list
        
        result['prediction_tr_list'] = np.sign(prediction_tr_list)
        result['prediction_ts_list'] = np.sign(prediction_ts_list)
        
        return result