"""Feature selection and classification by linear two steps model."""

from sklearn.linear_model import ElasticNet
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.preprocessing import LabelBinarizer
# from sklearn.utils import column_or_1d


class ElasticNetClassifier(LinearClassifierMixin, ElasticNet):
    """Class to extend elastic-net in case of classification."""

    def fit(self, X, y, check_input=True):
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if self._label_binarizer.y_type_.startswith('multilabel'):
            # we don't (yet) support multi-label classification in ENet
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        # Y = column_or_1d(Y, warn=True)
        super(ElasticNetClassifier, self).fit(X, Y)
        if self.classes_.shape[0] > 2:
            ndim = self.classes_.shape[0]
        else:
            ndim = 1
        self.coef_ = self.coef_.reshape(ndim, -1)

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_
