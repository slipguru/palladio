"""Module for the metrics used by PALLADIO."""
import numpy as np

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.regression import *
from sklearn.metrics.regression import __ALL__ as rmetrics

# List of callables for regression metrics
VARS = locals()
__REGRESSION_METRICS__ = [VARS[m] for m in rmetrics]


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


# List of callables for classification metrics
__CLASSIFICATION_METRICS__ = (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy)


def micro_precision_score(y_true, y_pred, labels=None, pos_label=1,
                          sample_weight=None):
    """Precision score for multiclass problems (micro averaged)."""
    return precision_score(
        y_true, y_pred, labels=labels, pos_label=pos_label,
        sample_weight=sample_weight, average='micro')


def micro_recall_score(y_true, y_pred, labels=None, pos_label=1,
                       sample_weight=None):
    """Precision score for multiclass problems (micro averaged)."""
    return recall_score(
        y_true, y_pred, labels=labels, pos_label=pos_label,
        sample_weight=sample_weight, average='micro')


def micro_f1_score(y_true, y_pred, labels=None, pos_label=1,
                   sample_weight=None):
    """Precision score for multiclass problems (micro averaged)."""
    return f1_score(
        y_true, y_pred, labels=labels, pos_label=pos_label,
        sample_weight=sample_weight, average='micro')


# List of callables for multiclass classification metrics
__MULTICLASS_CLASSIFICATION_METRICS__ = (
    accuracy_score, micro_precision_score,
    micro_recall_score, micro_f1_score,
    balanced_accuracy)
