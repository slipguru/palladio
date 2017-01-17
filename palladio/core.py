"""Module to distribute jobs among machines."""

import os
import time
import cPickle as pkl
import numpy as np

from sklearn.model_selection import train_test_split

from palladio.wrappers import l1l2Classifier  # need to check type
from palladio.wrappers import PipelineClassifier
from palladio.utils import set_module_defaults


def generate_job_list(n_regular, n_permutation):
    """Generate a vector used to distribute jobs among nodes.

    Given the total number of processes, generate a list of jobs distributing
    the load, so that each process has approximately the same amount of work
    to do (i.e., the same number of regular and permutated instances of the
    experiment).

    Parameters
    ----------
    n_regular : int
        The number of *regular* jobs, i.e. experiments where the labels
        have not been randomly shuffled.

    n_permutation : int
        The number of experiments for the permutation test, i.e. experiments
        where the labels *in the training set* will be randomly shuffled in
        order to disrupt any relationship between data and labels.

    Returns
    -------
    type_vector : numpy.ndarray
        A vector whose entries are either 0 or 1, representing respectively a
        job where a *regular* experiment is performed and one where an
        experiment where labels *in the training set* are randomly shuffled
        is performed.
    """
    type_vector = np.ones(n_permutation + n_regular, dtype=bool)
    type_vector[n_permutation:] = False
    np.random.shuffle(type_vector)

    return type_vector


def run_experiment(data, labels, config_dir, config, is_permutation_test,
                   experiments_folder_path, custom_name, rank=None):
    r"""Run a single independent experiment.

    Perform a single experiment, which is divided in three main stages:

    * Dataset Splitting
    * Model Selection
    * Model Assessment

    The details of the process actually depend on the algorithms used.

    Parameters
    ----------
    data : ndarray
        A :math:`n \\times p` matrix describing the input data.

    labels : ndarray
        A :math:`n`-dimensional vector containing the labels indicating the
        class of the samples.

    config_dir : string
        The path to the folder containing the configuration file, which will
        also contain the main session folder.

    config : object
        The object containing all configuration parameters for the session.

    is_permutation_test : bool
        A flag indicatin whether the experiment is part of the permutation test
        (and therefore has had its training labels randomly shuffled) or not.

    experiments_folder_path : string
        The path to the folder where all experiments' sub-folders
        will be stored

    custom_name : string
        The name of the subfolder where the experiments' results will be
        stored. It is a combination of a prefix which is either ``regular`` or
        ``permutation`` depending on the nature of the experiment, followed
        by two numbers which can be used to identify the experiment, for
        debugging purposes.
    """
    # Create experiment folders
    result_dir = os.path.join(experiments_folder_path, custom_name)
    os.mkdir(result_dir)

    Xtr, Xts, ytr, yts = train_test_split(
        data, labels, test_size=config.test_set_ratio)

    # Compute the ranges of the parameters using only the learning set
    if is_permutation_test:
        np.random.shuffle(ytr)

    # Setup the internal splitting for model selection
    # TODO: fix this and make it more general
    if config.learner == l1l2Classifier:
        int_k = config.cv_options['cv']
        # since it requires the labels, it can't be done before they are loaded
        ms_split = config.cv_splitting(ytr, int_k, rseed=time.clock())
        config.learner_params['ms_split'] = ms_split

        # Add process rank
        config.learner_params['process_rank'] = rank

        # Create the object that will actually perform
        # the classification/feature selection
        clf = config.learner_class(config.learner_params)
        clf.setup(Xtr, ytr, Xts, yts)
        result = clf.run()
    else:
        set_module_defaults(config, {
            'data_normalizer': None,
            'label_normalizer': None,
        })
        ms_split = None
        clf = PipelineClassifier(
            config.learner, config.learner_options, config.cv_options,
            config.final_scoring, config.data_normalizer,
            config.label_normalizer, config.force_classifier)

        # Set the actual data and perform
        # additional steps such as rescaling parameters etc.
        clf.fit(Xtr, ytr)
        ytr_pred = clf.predict(Xtr)
        yts_pred = clf.predict(Xts)

        # Get performance
        tr_err = 1 - clf.scoring(ytr, ytr_pred)
        ts_err = 1 - clf.scoring(yts, yts_pred)

        # Save results
        result = clf.get_cv_result()
        result['prediction_ts_list'] = yts_pred
        result['prediction_tr_list'] = ytr_pred
        result['err_tr_list'] = tr_err  # learning error
        result['err_ts_list'] = ts_err  # test error
        result['kcv_err_tr'] = 1 - np.clip(
            clf.gs_.cv_results_['mean_train_score'], 0, 1)  # training score
        result['kcv_err_ts'] = 1 - np.clip(
            clf.gs_.cv_results_['mean_test_score'], 0, 1)  # validation score
        result['best_params'] = clf.gs_.best_params_

        try:
            coef_ = clf.gs_.best_estimator_.coef_
            result['selected_list'] = np.nonzero(coef_)[0].tolist()
            result['beta_list'] = coef_.tolist()
        except AttributeError:
            result['selected_list'] = None
            result['beta_list'] = None

    result['labels_ts'] = yts  # also save labels

    # save results
    with open(os.path.join(result_dir, 'result.pkl'), 'w') as f:
        pkl.dump(result, f, pkl.HIGHEST_PROTOCOL)

    in_split = {
        'ms_split': ms_split,
        # 'outer_split': aux_splits[0]
        # 'outer_split': (idx_lr, idx_ts),
        # 'param_ranges': (param_1_range, param_2_range)
    }

    with open(os.path.join(result_dir, 'in_split.pkl'), 'w') as f:
        pkl.dump(in_split, f, pkl.HIGHEST_PROTOCOL)
