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
        
        for i, tau in enumerate(self._params['tau_range']):
            
            acc = 0
            
            for idx_tr, idx_ts in skf:
                
                Xk_tr = self._Xtr[idx_tr,:]
                Xk_ts = self._Xtr[idx_ts,:]
                
                Yk_tr = self._Ytr[idx_tr]
                Yk_ts = self._Ytr[idx_ts]
                
                clf = Lasso(alpha = tau)
                clf.fit(Xk_tr, Yk_tr) # fit the model
                
                selected_features = np.argwhere(clf.coef_)[0] # extract only nonzero coefficients
                
                Xk_tr2 = Xk_tr[:, selected_features]
                Xk_ts2 = Xk_ts[:, selected_features]
                
                clf = LinearRegression()
                clf.fit(Xk_tr2, Yk_tr) # fit the model
                Yk_lr = clf.predict(Xk_ts2) # predict test data
                Yk_lr = np.sign(Yk_lr) # take the sign
                
                acc += accuracy_score(Yk_ts, Yk_lr)
                
            acc_list[i] = acc
            
        ### Final train with the best choice for tau
        best_tau_idx = np.argmax(acc_list)
        best_tau = tau_range[best_tau_idx]
        
        clf = Lasso(alpha = best_tau)
        clf.fit(self._Xtr, self._Ytr) # fit the model
        
        selected_features = np.argwhere(clf.coef_)[0] # extract only nonzero coefficients
        
        X_tr2 = self._Xtr[:, selected_features]
        X_ts2 = self._Xts[:, selected_features]
        
        clf = LinearRegression()
        clf.fit(X_tr2, self._Ytr) # fit the model
        
        Y_lr = clf.predict(X_ts2) # predict test data
        Y_lr = np.sign(Y_lr) # take the sign
        
        Y_lr_tr = clf.predict(X_tr2) # predict training data
        Y_lr_tr = np.sign(Y_lr_tr) # take the sign
        
        result['selected_list'] = selected_features
        #result['beta_list'] = result['beta_list'][0]
        
        result['prediction_ts_list'] = Y_lr
        result['prediction_tr_list'] = Y_lr_tr
            
        result['labels_ts'] = self._Yts
            
        return result
        
    