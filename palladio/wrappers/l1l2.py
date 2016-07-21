import numpy as np

from l1l2signature import utils as l1l2_utils
from .classification import Classification

import l1l2py

class l1l2Classifier(Classification):
    
    def __init__(self, params):

        self._params = params

    def setup(self, Xtr, Ytr, Xts, Yts):

        self._Xtr = Xtr
        self._Ytr = Ytr

        self._Xts = Xts
        self._Yts = Yts

        rs = l1l2_utils.RangesScaler(Xtr, Ytr,
                                     self._params['data_normalizer'],
                                     self._params['labels_normalizer'])

        self._tau_range = rs.tau_range(self._params['tau_range'])
        self._mu_range = rs.mu_range(np.array([self._params['mu']]))
        self._lambda_range = np.sort(self._params['lambda_range'])
        
        # Determine which version of the algorithm must be used (CPU or GPU)
        # based on the process rank and configuration settings
        if self.get_param('process_rank') in self.get_param('gpu_processes'):
            self._algorithm_version = 'GPU'
        else:
            self._algorithm_version = 'CPU'
        
        
    def get_algorithm_version(self):
        
        return self._algorithm_version
        
        
    def run(self):
        
        
        # Execution
        result = l1l2py.model_selection(
            self._Xtr, self._Ytr, self._Xts, self._Yts,
            self._mu_range, self._tau_range, self._lambda_range,
            self.get_param('ms_split'), self.get_param('cv_error'), self.get_param('error'),
            data_normalizer = self.get_param('data_normalizer'),
            labels_normalizer = self.get_param('labels_normalizer'),
            sparse = self.get_param('sparse'),
            regularized = self.get_param('regularized'),
            return_predictions = self.get_param('return_predictions'),
            algorithm_version = self.get_algorithm_version()
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
