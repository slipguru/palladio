import numpy as np

from .classification import Classification

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import Lasso, LinearRegression

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
        
        ##############################
        ### Model selection phase
        ##############################
        
        ### Perform k splits once
        skf = StratifiedKFold(self._Ytr, n_folds = self._params['internal_k'])
        
        acc_list = np.empty((len(self._params['tau_range']),)) # store the mean accuracies for each model
        
        for tau in self._params['tau_range']:
            
            acc = 0
            
            for idx_tr, idx_ts in skf:
                
                Xk_tr = self._Xtr[idx_tr,:]
                Xk_ts = self._Xtr[idx_ts,:]
                
                Yk_tr = self._Ytr[idx_tr]
                Yk_ts = self._Ytr[idx_ts]
                
                clf = Lasso(alpha = tau)
                clf.fit(Xk_tr, Yk_tr) # fit the model
                clf.coef_ # extract only nonzero coefficients
                
            
            
            pass
        
        
        
        
        
        
        
        
        
        
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