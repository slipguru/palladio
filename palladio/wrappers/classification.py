class Classification(object):
    
    def run(self):
        """
        Returns
        ------
        
        result : dict
            Keys of the dictionary:
            
            Output dictionary. According with the parameters the dictionary has
        
            **kcv_err_ts** : (T, L) ndarray
                [STAGE I] Mean cross validation errors on the training set.
            **kcv_err_tr** : (T, L) ndarray
                [STAGE I] Mean cross validation errors on the training set.
            **tau_opt** : float
                Optimal value of tau selected in ``tau_range``.
            **lambda_opt** : float
                Optimal value of lambda selected in ``lambda_range``.
            **beta_list** :  list of M (S,1) ndarray
                [STAGE II] Models calculated for each value in ``mu_range``.
            **selected_list** : list of M (P,) ndarray of boolean
                [STAGE II] Selected variables for each model calculated.
            **err_ts_list** : list of M floats
                [STAGE II] List of Test errors evaluated for the all the models.
            **err_tr_list** : list of M floats
                [STAGE II] List of Training errors evaluated for the all the models.
            **prediction_ts_list** : list of M two-dimensional ndarray, optional
                [STAGE II] Prediction vectors for the models evaluated on the test
                set.
            **prediction_tr_list** : list of M two-dimensional ndarray, optional
                [STAGE II] Prediction vectors for the models evaluated on the
                training set.
            
            ''
        """
        
        raise Exception("Abstract method")