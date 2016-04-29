import l1l2py

import numpy as np

from l1l2signature import utils as l1l2_utils

class l1l2Classifier:
    
    def __init__(self, params):
        
        self._params = params
        
    def setup(self, Xtr, Ytr, Xts, Yts):
        
        self._Xtr = Xtr
        self._Ytr = Ytr
        
        self._Xts = Xts
        self._Yts = Yts
        
        rs = l1l2_utils.RangesScaler(Xtr, Ytr, self._params['data_normalizer'], self._params['labels_normalizer'])
        
        self._tau_range = rs.tau_range(self._params['tau_range'])
        self._mu_range = rs.mu_range(self._params['mu_range'])
        self._lambda_range = np.sort(self._params['lambda_range'])
        
        pass
        
    def run(self):
        
        # Execution
        result = l1l2py.model_selection(
            self._Xtr, self._Ytr, self._Xts, self._Yts, 
            self._mu_range, self._tau_range, self._lambda_range,
            self._params['ms_split'], self._params['cv_error'], self._params['error'],
            self._params['data_normalizer'], self._params['labels_normalizer'],
            self._params['sparse'], self._params['regularized'], self._params['return_predictions']
            )

        return result
    
    
    pass