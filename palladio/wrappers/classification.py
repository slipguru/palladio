"""Abstract method for classification wrappers."""
from sklearn.metrics import accuracy_score


class Classification(object):
    """Abstract method for classification wrappers."""

    def __init__(self, learner_options=None, cv_options=None,
                 final_scoring='accuracy', data_normalizer=None,
                 label_normalizer=None, force_classifier=False):

        self.learner_options = {} if learner_options is None \
            else learner_options
        # self.param_grid = {} if param_grid is None else param_grid
        self.cv_options = {} if cv_options is None else cv_options
        self.final_scoring = accuracy_score if final_scoring is 'accuracy' \
            else final_scoring

        self.data_normalizer = data_normalizer
        self.label_normalizer = label_normalizer
        self.force_classifier = force_classifier

    def setup(self, Xtr, Ytr, Xts, Yts):

        self._Xtr = Xtr
        self._Ytr = Ytr

        self._Xts = Xts
        self._Yts = Yts

    # def get_param(self, param_name):
    #     return self._params[param_name]

    def getXtr(self):
        return self._Xtr

    def getXts(self):
        return self._Xts

    def getYtr(self):
        return self._Ytr

    def getYts(self):
        return self._Yts

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
        """
        raise Exception("Abstract method")
