import numpy as np

from sklearn.linear_model import SGDClassifier

class SGDClassification(Classification):
    
    
    def setup(self, Xtr, Ytr, Xts, Yts):
        
        self._Xtr = Xtr
        self._Ytr = Ytr
        
        self._Xts = Xts
        self._Yts = Yts
        
        
        
        
        
        
        
        
        
        rs = l1l2_utils.RangesScaler(Xtr, Ytr, self._params['data_normalizer'], self._params['labels_normalizer'])
        
        self._tau_range = rs.tau_range(self._params['tau_range'])
        self._mu_range = rs.mu_range(self._params['mu_range'])
        self._lambda_range = np.sort(self._params['lambda_range'])
        