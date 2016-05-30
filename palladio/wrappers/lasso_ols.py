import numpy as np

from .classification import Classification

from sklearn.linear_model import Lasso

class lasso_olsClassifier(Classification):
    """
    Selects feature using lasso, then adjust weights with OLS
    
    From http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    
    The optimization objective for Lasso is:
    
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """
    
    def setup(self, Xtr, Ytr, Xts, Yts):
        
        Classification.setup(self, Xtr, Ytr, Xts, Yts)
        
    def run(self):
        
        # Execution
        result = l1l2py.model_selection(
            self._Xtr, self._Ytr, self._Xts, self._Yts, 
            self._mu_range, self._tau_range, self._lambda_range,
            self._params['ms_split'], self._params['cv_error'], self._params['error'],
            self._params['data_normalizer'], self._params['labels_normalizer'],
            self._params['sparse'], self._params['regularized'], self._params['return_predictions']
            )

        ### Return only the first element of the list, which is the one related to the smallest value of mu
        ### BEGIN
        result['selected_list'] = result['selected_list'][0]
        result['beta_list'] = result['beta_list'][0]
        
        result['prediction_ts_list'] = result['prediction_ts_list'][0].ravel()
        if 'prediction_tr_list' in result.keys():
            result['prediction_tr_list'] = result['prediction_tr_list'][0].ravel()
            
        ### END
        
        
        return result
    
    
    pass