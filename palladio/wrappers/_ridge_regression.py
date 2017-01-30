"""Extension module for palladio (pd_test.py).

.. deprecated:: 0.5
"""
from __future__ import division

import multiprocessing as mp
import numpy as np

from sklearn.model_selection import KFold

from l1l2py.algorithms import ridge_regression


def _mu_scaling_factor(data):
    """Return the mu scaling factor."""
    n, d = data.shape

    if d > n:
        tmp = np.dot(data, data.T)
        num = np.linalg.eigvalsh(tmp).max()
    else:
        tmp = np.dot(data.T, data)
        evals = np.linalg.eigvalsh(tmp)
        num = evals.max() + evals.min()

    return (num / (2. * n))


def RLS_path(X_tr, Y_tr, mu_range):
    """Perform ridge regression path."""
    beta_list = list()
    for mu in mu_range:
        beta = ridge_regression(X_tr, Y_tr, mu)
        beta_list.append(beta)
    return beta_list


def kf_worker(X_tr, Y_tr, mu_range, tr_idx, vld_idx, i, results):
    """Worker for parallel KFold implementation."""
    betas = RLS_path(X_tr, Y_tr, mu_range)
    results[i] = {'betas': betas, 'tr_idx': tr_idx, 'vld_idx': vld_idx}


def regression_error(labels, predictions):
    r"""Returns regression error.

    The regression error is the sum of the quadratic differences between the
    ``labels`` values and the ``predictions`` values, over the number of
    samples.

    Parameters
    ----------
    labels : array_like, shape (N,)
        Regression labels.
    predictions : array_like, shape (N,)
        Regression labels predicted.

    Returns
    -------
    error : float
        Regression error calculated.

    """
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()

    difference = labels - predictions
    return np.dot(difference.T, difference).squeeze() / float(len(labels))


def RLSCV(data, labels, cv_split=5, log_mu_range=(-9, 0, 20)):
    """Ridge regression with built-in cross-validation."""
    n, _ = data.shape
    mu_range = np.logspace(log_mu_range[0], log_mu_range[1],
                           log_mu_range[2]) * _mu_scaling_factor(data)

    # Kfold starts here
    try:
        kf = KFold(n=n, n_folds=cv_split)
    except Exception:
        kf = KFold(n_splits=cv_split).split(data)

    jobs = []  # multiprocessing job list
    results = mp.Manager().dict()  # dictionary shared among processess

    # Submit each kfold job
    for i, (tr_idx, vld_idx) in enumerate(kf):
        X_tr = data[tr_idx, :]
        Y_tr = labels[tr_idx]
        X_vld = data[vld_idx, :]
        Y_vld = labels[vld_idx]

        p = mp.Process(target=kf_worker, args=(X_tr, Y_tr, mu_range,
                                               tr_idx, vld_idx,
                                               i, results))
        jobs.append(p)
        p.start()

    # Collect the results
    for p in jobs:
        p.join()

    # Evaluate the errors
    tr_errors = np.zeros((cv_split, len(mu_range)))
    vld_errors = np.zeros((cv_split, len(mu_range)))
    for i in results.keys():
        betas = results[i]['betas']
        tr_idx = results[i]['tr_idx']
        vld_idx = results[i]['vld_idx']
        X_tr = data[tr_idx, :]
        Y_tr = labels[tr_idx]
        X_vld = data[vld_idx, :]
        Y_vld = labels[vld_idx]
        for j, beta in enumerate(betas):
            Y_pred_tr = np.sign(np.dot(X_tr, beta))
            Y_pred_vld = np.sign(np.dot(X_vld, beta))
            vld_errors[i, j] = regression_error(Y_vld, Y_pred_vld)
            tr_errors[i, j] = regression_error(Y_tr, Y_pred_tr)

    # Once all the training is done, get the best tau
    avg_vld_err = np.mean(vld_errors, axis=0)
    avg_tr_err = np.mean(tr_errors, axis=0)
    opt_mu = mu_range[np.argmin(avg_vld_err)]

    # Refit and return the best model
    beta = ridge_regression(data, labels, opt_mu)

    return beta
