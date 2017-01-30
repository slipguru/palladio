"""Feature selection using Elastic Net."""

from sklearn.linear_model import ElasticNet
from sklearn.feature_selection.base import SelectorMixin


class ElasticNetFeatureSelection(SelectorMixin, ElasticNet):
    """Class to extend elastic-net in case of classification.

    In case in which n_jobs != 1 for GridSearchCV, the estimator class must be
    pickable, therefore statically defined.
    """

    def _get_support_mask(self):
        # Returns the mask relative to coefficients different from zero
        mask = (self.coef_ != 0)

        return mask

