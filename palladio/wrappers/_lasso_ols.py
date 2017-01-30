"""Wrapper for sklearn Lasso and Linear Regression."""
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, LinearRegression

from palladio.wrappers.classification import Classification


class LassoOlsClassifier(Classification):
    """Select features using lasso, then adjust weights with OLS.

    From
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

    The optimization objective for Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """

    def setup(self, Xtr, Ytr, Xts, Yts):
        """Setup for the lasso_olsClassifier."""
        Classification.setup(self, Xtr, Ytr, Xts, Yts)

    def get_l1_bound(self):
        r"""Estimation of an useful maximum bound for the `l1` penalty term.

        For each value of ``tau`` smaller than the maximum bound the solution
        vector contains at least one non zero element.

        .. warning

            That is, bounds are right if you run the `l1l2` regularization
            algorithm with the same data matrices.

        Parameters
        ----------
        data : (N, P) ndarray
            Data matrix.
        labels : (N,)  or (N, 1) ndarray
            Labels vector.

        Returns
        -------
        tau_max : float
            Maximum ``tau``.

        Examples
        --------
        >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
        >>> beta = numpy.array([0.1, 0.1, 0.0])
        >>> Y = numpy.dot(X, beta)
        >>> tau_max = l1l2py.algorithms.l1_bound(X, Y)
        >>> l1l2py.algorithms.l1l2_regularization(X, Y, 0.0, tau_max).T
        array([[ 0.,  0.,  0.]])
        >>> beta = l1l2py.algorithms.l1l2_regularization(X, Y, 0.0, tau_max - 1e-5)
        >>> len(numpy.flatnonzero(beta))
        1

        """
        data = self._Xtr
        labels = self._Ytr

        n = data.shape[0]
        corr = np.abs(np.dot(data.T, labels))

        tau_max = (corr.max() * (2.0 / n))

        return tau_max

    def run(self):
        # Model selection phase
        internal_k = self._params['internal_k']

        # Perform k splits once
        skf = StratifiedKFold(self._Ytr, n_folds=internal_k)

        # store the mean accuracies for each model
        acc_list = np.empty((len(self._params['tau_range']),))

        TAU_MAX = self.get_l1_bound()
        # print("TAU_MAX = {}".format(TAU_MAX))

        for i, tau_scaling in enumerate(self._params['tau_range']):
            tau = TAU_MAX * tau_scaling
            # print("{}-th value of tau ({})".format(i+1, tau))

            acc = 0
            # number of solutions which consisted of only zeros
            # (early stopping for too big tau)
            N_allzeros = 0

            for idx_tr, idx_ts in skf:
                Xk_tr = self._Xtr[idx_tr, :]
                Xk_ts = self._Xtr[idx_ts, :]

                Yk_tr = self._Ytr[idx_tr]
                Yk_ts = self._Ytr[idx_ts]

                clf = Lasso(alpha=tau)
                clf.fit(Xk_tr, Yk_tr)  # fit the model

                # extract only nonzero coefficients
                selected_features = np.argwhere(clf.coef_).ravel()
                # print("Selected {} features".format(len(selected_features)))

                if len(selected_features) == 0:
                    # If no features are selected, just assign all samples to
                    # the most common class (in the training set)
                    N_allzeros += 1
                    Yk_lr = np.ones((len(Yk_ts),)) * np.sign(Yk_tr.sum() + 0.1)

                else:
                    # Else, run OLS and get weights for coefficients NOT
                    # affected by shrinking
                    Xk_tr2 = Xk_tr[:, selected_features]
                    Xk_ts2 = Xk_ts[:, selected_features]

                    clf = LinearRegression(normalize=False)
                    clf.fit(Xk_tr2, Yk_tr)  # fit the model
                    Yk_lr = clf.predict(Xk_ts2)  # predict test data
                    Yk_lr = np.sign(Yk_lr)  # take the sign

                acc += accuracy_score(Yk_ts, Yk_lr)

            acc_list[i] = acc / internal_k

            if N_allzeros == internal_k:
                # All k-fold splits returned empty solutions, stop here as
                # bigger values of tau would return empty solutions as well
                print("The {}-th value of tau ({}) returned only empty "
                      "solutions".format(i + 1, tau))
                break

        # Final train with the best choice for tau
        best_tau_idx = np.argmax(acc_list)
        # best_tau = self._params['tau_range'][best_tau_idx]
        best_tau = self._params['tau_range'][best_tau_idx] * TAU_MAX

        clf = Lasso(alpha=best_tau)
        clf.fit(self._Xtr, self._Ytr)  # fit the model

        # extract only nonzero coefficients
        # selected_features = np.argwhere(clf.coef_)[0]
        selected_features = np.argwhere(clf.coef_).ravel()

        if len(selected_features) == 0:
            print("WARNING: the allegedly best solution (tau = {}) was "
                  " empty".format(best_tau))

            sign = np.sign(np.sum(self._Ytr) + 0.1)
            Y_lr = np.ones((len(self._Yts)),) * sign
            Y_lr_tr = np.ones((len(self._Ytr)),) * sign

        else:
            X_tr2 = self._Xtr[:, selected_features]
            X_ts2 = self._Xts[:, selected_features]

            clf = LinearRegression()
            clf.fit(X_tr2, self._Ytr)  # fit the model

            Y_lr = clf.predict(X_ts2)  # predict test data
            Y_lr = np.sign(Y_lr)  # take the sign

            Y_lr_tr = clf.predict(X_tr2)  # predict training data
            Y_lr_tr = np.sign(Y_lr_tr)  # take the sign

        result = {}
        result['selected_list'] = selected_features
        # result['beta_list'] = result['beta_list'][0]
        result['prediction_ts_list'] = Y_lr
        result['prediction_tr_list'] = Y_lr_tr
        result['labels_ts'] = self._Yts
        return result
