"""
This file should contain all information required for a session.
It can be generated either from a configuration file or directly in case it
is used in an interactive session (i.e., jupyter notebook).
"""

import imp

from sklearn.datasets.base import Bunch

# class Session(object):
#     """
#     All configuration required for a session
#
#     Parameters
#     ----------
#
#
#     """
#
#     def __init__(
#             self,
#             data,
#             labels,
#             n_splits_regular,
#             n_splits_permutation,
#             estimator,
#             param_grid={},
#             ma_options=None,
#             learning_task=None,
#             vs_analysis=True,
#             frequency_threshold=0.75,
#             score_surfaces_options=None,
#             feature_names=None,
#             session_folder=None,
#             config_path=None,
#             analysis_folder='analysis'
#         ):
#
#         # Input data and labels
#         self._data = data
#         self._labels = labels
#
#         # Number of experiments for regular and permutation batches
#         self._n_splits_regular = n_splits_regular
#         self._n_splits_permutation = n_splits_permutation
#
#         # The estimator used and its parameters
#         self._estimator = estimator
#         self._param_grid = param_grid
#
#         # Model Assessment options
#         self._ma_options = ma_options
#
#         # The learning task, if None palladio tries to guess it
#         # [see sklearn.utils.multiclass.type_of_target]
#         self._learning_task = learning_task
#
#         # Which object is to be used for the feature analysis
#         # TODO explain
#         self._vs_analysis = vs_analysis
#
#         # Signature Parameters
#         self._frequency_threshold = frequency_threshold
#
#         # Plotting Options
#         self._score_surfaces_options = score_surfaces_options
#
#         # An array-like containing the names of the features
#         self._feature_names = feature_names
#
#         # The folder where the results will be stored
#         self._session_folder = session_folder
#
#         # The path of the configuration file itself, if needed
#         self._config_path = config_path
#
#         self._analysis_folder = analysis_folder


ALLOWED_VARS = ['n_splits_regular', 'frequency_threshold', 'labels',
                'analysis_folder', 'score_surfaces_options', 'data',
                'config_path', 'learning_task', 'estimator', 'vs_analysis',
                'n_splits_permutation', 'ma_options', 'session_folder']


def load_from_config(config_path):
    """Create a PALLADIO session object starting from a configuration file."""
    imp.acquire_lock()
    config = imp.load_source('config', config_path)
    imp.release_lock()

    analysis_folder = config.analysis_folder if hasattr(
        config, 'analysis_folder') else 'analysis'

    variables_in_config = [item for item in dir(config) if item in ALLOWED_VARS]

    # return Session(
    #     config.data,
    #     config.labels,
    #     config.n_splits_regular,
    #     config.n_splits_permutation,
    #     config.estimator,
    #     param_grid=config.param_grid
    #     ma_options=config.ma_options,
    #     learning_task=config.learning_task,
    #     vs_analysis=config.vs_analysis,
    #     frequency_threshold=config.frequency_threshold,
    #     score_surfaces_options=config.score_surfaces_options,
    #     feature_names=config.feature_names
    #     session_folder=config.session_folder,
    #     config_path=config_path,
    #     analysis_folder=analysis_folder
    # )

    bunch = Bunch(**dict(zip(
        variables_in_config, [getattr(config, item) for item
                              in variables_in_config])))
    bunch['config_path'] = config_path
    bunch['analysis_folder'] = analysis_folder
    return bunch
