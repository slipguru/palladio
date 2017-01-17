"""Module for the metrics used by PALLADIO."""
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef

__all__ = (
    'accuracy_score', 'precision_recall_fscore_support',
    'matthews_corrcoef', 'balanced_accuracy')


def balanced_accuracy(y_true, y_pred):
    """Return a balanced accuracy, supporting also multiclass learning.

    This is computed averaging the balanced accuracy for each class.
    Here
    https://github.com/scikit-learn/scikit-learn/issues/6747
    there is an issue on how to implement this in sklearn.
    """
    perclass_balanced_accuracy = np.zeros(np.unique(y_true).shape[0])
    for i, class_ in enumerate(np.unique(y_true)):
        y_true_class = (y_true == class_).astype(int)
        y_pred_class = (y_pred == class_).astype(int)

        tp = np.sum((y_pred_class == 1) * (y_true_class == y_pred_class))
        tn = np.sum((y_pred_class == 0) * (y_true_class == y_pred_class))
        fp = np.sum((y_pred_class == 1) * (y_true_class != y_pred_class))
        fn = np.sum((y_pred_class == 0) * (y_true_class != y_pred_class))
        sensitivity = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        perclass_balanced_accuracy[i] = (sensitivity + specificity) / 2.
    return np.mean(perclass_balanced_accuracy)
