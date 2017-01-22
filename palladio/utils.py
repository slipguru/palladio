"""Utilities functions and classes."""

# Author: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>,
#         Annalisa Barla <annalisa.barla@disi.unige.it>
# License: new BSD.
# PALLADIO Refactoring: Samuele Fiorini

import numpy as np
# import l1l2py


class PDException(Exception):
    """Exception raised by ``PALLADIO`` classes and functions."""

    pass


class RangesScaler(object):
    """Given data and labels helps to scale L1L2 parameters ranges properly.

    This class works on tau and mu ranges passed to the l1l2 selection
    framework (see also :func:`l1l2py.model_selection` and related
    function for details).

    Scaling ranges permits to use relative (and not absolute) ranges of
    parameters.

    Attributes
    ----------
    norm_data : :class:`numpy.ndarray`
        Normalized data matrix.
    norm_labels : :class:`numpy.ndarray`
        Normalized labels vector.
    """

    def __init__(self, data, labels, data_normalizer=None,
                 labels_normalizer=None):

        self.norm_data = data
        self.norm_labels = labels
        self._tsf = self._msf = None

        # Data must be normalized
        if data_normalizer:
            self.norm_data = data_normalizer(self.norm_data)
        if labels_normalizer:
            self.norm_labels = labels_normalizer(self.norm_labels)

    def tau_range(self, trange):
        """Return a scaled tau range.

        Tau scaling factor is the maximum tau value to avoid and empty solution
        (where all variables are discarded).
        The value is estimated on the maximum correlation between data and
        labels.

        Parameters
        ----------
        trange : :class:`numpy.ndarray`
            Tau range containing relative values (expected maximum is lesser
            than 1.0 and minimum greater than 0.0).

        Returns
        -------
        tau_range : :class:`numpy.ndarray`
            Scaled tau range.

        Raises
        ------
        PDException
            If trange values are not in the [0, 1) interval
            (right extreme excluded).
        """
        trange = np.sort(trange)

        if max(trange) >= 1.0 or min(trange) < 0.0:
            raise PDException('relative tau values have to '
                              'be in [0, 1)')

        return trange * self.tau_scaling_factor

    def mu_range(self, mrange):
        """Return a scaled mu range.

        Mu scaling factor is estimated on the maximum eigenvalue of the
        correlation matrix and is used to simplify the parameters choice.

        Parameters
        ----------
        mrange : :class:`numpy.ndarray`
            Mu range containing relative values (expected maximum is lesser
            than 1.0 and minimum greater than 0.0).

        Returns
        -------
        mu_range : :class:`numpy.ndarray`
            Scaled mu range.

        Raises
        ------
        PDException
            If mrange values are not all greater than 0.
        """
        mrange = np.sort(mrange)

        if min(mrange) < 0.0:
            raise PDException('relative mu values have to be '
                              'greater than 0')

        return np.sort(mrange) * self.mu_scaling_factor

    @property
    def tau_scaling_factor(self):
        """Tau scaling factor calculated on given data and labels."""
        if self._tsf is None:
            self._tsf = self._tau_scaling_factor()
        return self._tsf

    @property
    def mu_scaling_factor(self):
        """Mu scaling factor calculated on given data matrix."""
        if self._msf is None:
            self._msf = self._mu_scaling_factor()
        return self._msf

    def _tau_scaling_factor(self):
        return l1l2py.algorithms.l1_bound(self.norm_data, self.norm_labels)

    def _mu_scaling_factor(self):
        n, d = self.norm_data.shape

        if d > n:
            tmp = np.dot(self.norm_data, self.norm_data.T)
            num = np.linalg.eigvalsh(tmp).max()
        else:
            tmp = np.dot(self.norm_data.T, self.norm_data)
            evals = np.linalg.eigvalsh(tmp)
            num = evals.max() + evals.min()

        return (num / (2. * n))


def signatures(splits_results, frequency_threshold=0.0):
    """Return (almost) nested signatures for each correlation value.

    The function returns 3 lists where each item refers to a signature
    (for increasing value of linear correlation).
    Each signature is orderer from the most to the least selected variable
    across KCV splits results.

    Parameters
    ----------
    splits_results : iterable
        List of results from L1L2Py module, one for each external split.
    frequency_threshold : float
        Only the variables selected more (or equal) than this threshold are
        included into the signature.

    Returns
    -------
    sign_totals : list of :class:`numpy.ndarray`.
        Counts the number of times each variable in the signature is selected.
    sign_freqs : list of :class:`numpy.ndarray`.
        Frequencies calculated from ``sign_totals``.
    sign_idxs : list of :class:`numpy.ndarray`.
        Indexes of the signatures variables .

    Examples
    --------
    >>> from l1l2signature.utils import signatures
    >>> splits_results = [{'selected_list':[[True, False], [True, True]]},
    ...                   {'selected_list':[[True, False], [False, True]]}]
    >>> sign_totals, sign_freqs, sign_idxs = signatures(splits_results)
    >>> print sign_totals
    [array([ 2.,  0.]), array([ 2.,  1.])]
    >>> print sign_freqs
    [array([ 1.,  0.]), array([ 1. ,  0.5])]
    >>> print sign_idxs
    [array([0, 1]), array([1, 0])]
    """
    # Computing totals and frequencies
    selection_totals = selection_summary(splits_results)
    selection_freqs = selection_totals / len(splits_results)

    # Variables are ordered and filtered by frequency threshold
    sorted_idxs = np.argsort(selection_freqs, axis=1)
    sorted_idxs = (sorted_idxs.T)[::-1].T  # Reverse order

    # ... ordering
    for i, si in enumerate(sorted_idxs):
        selection_freqs[i] = selection_freqs[i][si]
        selection_totals[i] = selection_totals[i][si]

    # ... filtering
    threshold_mask = (selection_freqs >= frequency_threshold)

    # Signatures Ordered and Filtered!
    sign_totals = list()
    sign_freqs = list()
    sign_idxs = list()
    for i, mask in enumerate(threshold_mask):
        sign_totals.append(selection_totals[i][mask])
        sign_freqs.append(selection_freqs[i][mask])
        sign_idxs.append(sorted_idxs[i][mask])

    return sign_totals, sign_freqs, sign_idxs


def selection_summary(splits_results):
    """Count how many times each variables was selected.

    Parameters
    ----------
    splits_results : iterable
        List of results from L1L2Py module, one for each external split.

    Returns
    -------
    summary : :class:`numpy.ndarray`
        Selection summary. ``# mu_values X # variables`` matrix.
    """
    # Sum selection lists by mu values (mu_num x num_var)
    return np.sum(np.asarray(sr['selected_list'], dtype=float)
                  for sr in splits_results)


def confusion_matrix(labels, predictions):
    """Calculate a confusion matrix.

    From given real and predicted labels, the function calculated
    a confusion matrix as a double nested dictionary.
    The external one contains two keys, ``'T'`` and ``'F'``.
    Both internal dictionaries
    contain a key for each class label. Then the ``['T']['C1']`` entry counts
    the number of correctly predicted ``'C1'`` labels,
    while ``['F']['C2']`` the incorrectly predicted ``'C2'`` labels.

    Note that each external dictionary correspond to a confusion
    matrix diagonal and the function works only on two-class labels.

    Parameters
    ----------
    labels : iterable
        Real labels.

    predictions : iterable
        Predicted labels.

    Returns
    -------
    cm : dict
        Dictionary containing the confusion matrix values.
    """
    cm = {'T': dict(), 'F': dict()}

    real_unique_labels, real_C1, real_C2 = _check_unique_labels(labels)
    pred_unique_labels, pred_C1, pred_C2 = _check_unique_labels(predictions)

    if not np.all(real_unique_labels == pred_unique_labels):
        raise PDException('real and predicted labels differ.')

    cm['T'][real_unique_labels[0]] = (real_C1 & pred_C1).sum()  # True C1
    cm['T'][real_unique_labels[1]] = (real_C2 & pred_C2).sum()  # True C2
    cm['F'][real_unique_labels[0]] = (real_C2 & pred_C1).sum()  # False C1
    cm['F'][real_unique_labels[1]] = (real_C1 & pred_C2).sum()  # False C2

    return cm


def classification_measures(confusion_matrix, positive_label=None):
    """Calculate some classification measures.

    Measures are calculated from a given confusion matrix
    (see :func:`confusion_matrix` for a detailed description of the
    required structure).

    The ``positive_label`` arguments allows to specify what label has to be
    considered the positive class. This is needed to calculate some
    measures like F-measure and set some aliases (e.g. precision and recall
    are respectively the 'predictive value' and the 'true rate' for the
    positive class).

    If ``positive_label`` is None, the resulting dictionary will not
    contain all the measures. Assuming to have to classes 'C1' and 'C2',
    and to indicate 'C1' as the positive (P) class, the function returns a
    dictionary with the following structure::

        {
            'C1': {'predictive_value': --,  # TP / (TP + FP)
                   'true_rate':        --}, # TP / (TP + FN)
            'C2': {'predictive_value': --,  # TN / (TN + FN)
                   'true_rate':        --}, # TN / (TN + FP)
            'accuracy':          --,        # (TP + TN) / (TP + FP + FN + TN)
            'balanced_accuracy': --,        # 0.5 * ( (TP / (TP + FN)) +
                                            #         (TN / (TN + FP)) )
            'MCC':               --,        # ( (TP * TN) - (FP * FN) ) /
                                            # sqrt( (TP + FP) * (TP + FN) *
                                            #       (TN + FP) * (TN + FN) )

            # Following, only with positive_labels != None
            'sensitivity':       --,        # P true rate: TP / (TP + FN)
            'specificity':       --,        # N true rate: TN / (TN + FP)
            'precision':         --,        # P predictive value: TP / (TP + FP)
            'recall':            --,        # P true rate: TP / (TP + FN)
            'F_measure':         --         # 2. * ( (Precision * Recall ) /
                                            #        (Precision + Recall) )
        }

    Parameters
    ----------
    confusion_matrix : dict
        Confusion matrix (as the one returned by :func:`confusion_matrix`).

    positive_label : str
        Positive class label.

    Returns
    -------
    summary : dict
        Dictionary containing calculated measures.
    """
    # Confusion Matrix
    #           True P      True N
    # Pred P      TP         FP         P Pred Value
    # Pred N      FN         TN         N Pred Value
    #         Sensitivity Specificity

    labels = confusion_matrix['T'].keys()

    if positive_label is not None:
        P = positive_label
        if P not in labels:
            raise PDException('label %s not found.' % positive_label)

        N = set(labels).difference([positive_label]).pop()
    else:
        P, N = sorted(labels)

    # shortcuts ------------------------------------
    TP = confusion_matrix['T'][P]
    TN = confusion_matrix['T'][N]
    FP = confusion_matrix['F'][P]
    FN = confusion_matrix['F'][N]
    # ----------------------------------------------

    summary = dict({P: dict(), N: dict()})

    summary[P]['predictive_value'] = TP / float(TP + FP)
    summary[P]['true_rate'] = TP / float(TP + FN)           # sensitivity

    summary[N]['predictive_value'] = TN / float(TN + FN)
    summary[N]['true_rate'] = TN / float(TN + FP)           # specificity

    summary['accuracy'] = (TP + TN) / float(TP + FP + FN + TN)
    summary['balanced_accuracy'] = 0.5 * (summary[P]['true_rate'] +
                                          summary[N]['true_rate'])

    den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    summary['MCC'] = (((TP * TN) - (FP * FN)) /
                      (1.0 if den == 0 else np.sqrt(den)))

    if positive_label is not None:
        summary['sensitivity'] = summary[P]['true_rate']
        summary['specificity'] = summary[N]['true_rate']

        summary['precision'] = summary[P]['predictive_value']
        summary['recall'] = summary['sensitivity']

        summary['F_measure'] = (
            2. * ((summary['precision'] * summary['recall']) /
                  (summary['precision'] + summary['recall']))
        )

    return summary


def _check_unique_labels(labels):
    labels = np.array([str(s).strip() for s in labels])
    unique_labels = np.unique(labels)

    if not len(unique_labels) == 2:
        raise PDException('more than 2 classes in labels.')

    unique_labels.sort(kind='mergesort')
    class1 = (labels == unique_labels[0])
    class2 = (labels == unique_labels[1])

    return unique_labels, class1, class2


def set_module_defaults(module, dictionary):
    """Set default variables of a module, given a dictionary.

    Used after the loading of the configuration file to set some defaults.
    """
    for k, v in dictionary.iteritems():
        try:
            getattr(module, k)
        except AttributeError:
            setattr(module, k, v)


def sec_to_timestring(seconds):
    """Transform seconds into a formatted time string.

    Parameters
    -----------
    seconds : int
        Seconds to be transformed.
    Returns
    -----------
    time : string
        A well formatted time string.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)
