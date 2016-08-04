"""Module for loading the dataset from npy/pkl files."""
import os
import cPickle as pkl
import numpy as np

from .Dataset import Dataset


class DatasetNPY(Dataset):
    """"Dataset composed by data matrix and labels vector.

    Matrices are stored in two NPY files, while features/samples names are
    stored in a pkl file.
    """

    def load_dataset(self, base_path):
        """Read data matrix and labels vector from files.

        Requires a .pkl file containing a dictionary storing
        samples and features names (keys ``index`` and
        ``columns`` respectively).

        Parameters
        ----------

        base_path : string
            The base path relative to which files are stored.

        Returns
        -------

        data : ndarray
            The :math:`n \\times p` data matrix .

        labels : ndarray
            The :math:`n`-dimensional vector containing labels.

        feature_names : list
            The list containing the names of the features

        """

        data_path = os.path.join(base_path, self.get_file('data'))
        labels_path = os.path.join(base_path, self.get_file('labels'))
        indcol_path = os.path.join(base_path, self.get_file('indcol'))

        #####################
        ### DATA + LABELS ###
        #####################
        data = np.load(data_path)
        labels = np.load(labels_path)

        if self.get_option('samples_on') == 'col':
            data = data.T

        # TODO FIX IF DATA IS TRANSPOSED???
        with open(indcol_path, 'r') as f:
            res = pkl.load(f)

            feature_names = np.array(res['columns'])
            samples_names = np.array(res['index'])

        # if not self.get_option('data_preprocessing') is None:
        # ### TODO Check!!!
        #     # if rank == 0:
        #         # print("Preprocessing data...")
        #
        #     self.get_option('data_preprocessing').load_data(pd_data)
        #     pd_data = self.get_option('data_preprocessing').process()

        #######################
        ### LABELS TO -1/+1 ###
        #######################

        if not self.get_option('positive_label') is None:
            poslab = self.get_option('positive_label')
        else:
            uv = np.sort(np.unique(labels))

            if len(uv) != 2:
                raise Exception(
                    "More than two unique values in the labels array")

            poslab = uv[0]

        # Auxiliary function required to convert the labels to
        # -1/+1
        def _toPlusMinus(x):
            """Converts the values in the labels"""
            if x == poslab:
                return +1.0
            else:
                return -1.0

        labels_mapped = map(_toPlusMinus, labels)
        labels = np.array(labels_mapped)
        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of samples in data do not correspond "
                             "to the number of samples in labels.")

        return data, labels, feature_names
