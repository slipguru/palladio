"""Dataset loading utilities."""
try:
    import cPickle as pkl
except:
    import pickle as pkl
import numpy as np
import os
import pandas as pd
import shutil
import time
import warnings
from sklearn.datasets.base import Bunch
from sklearn.utils.deprecation import deprecated

__all__ = ('DatasetCSV', 'DatasetNPY', 'DatasetXLS', 'load_csv')


def load_csv(data_path, target_path, return_X_y=False,
             data_loading_options=None, target_loading_options=None,
             samples_on='row'):
    """Tabular data loading utiliy.

    Parameters
    ----------

    data_path : string.
        The path to the csv file containing the `data`.
    target_path : string.
        The path to the csv file containing the `target` (labels).
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    data_loading_options : dictionary.
        The options passed to `pandas.read_csv()` function when loading the
        `data`.
    target_loading_options : dictionary.
        The options passed to `pandas.read_csv()` function when loading the
        `target`.
    samples_on : string.
        If in `col` or `cols`, the samples are assumed on columns; else the
        samples are assumed on rows (default).

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features.
    (data, target) : tuple if ``return_X_y`` is True
    """
    data_df = pd.read_csv(data_path, **(data_loading_options or {}))
    target_df = pd.read_csv(target_path, **(target_loading_options or {}))

    if samples_on.lower() in ['col', 'cols']:
        data_df = data_df.transpose()

    # Unpack pandas DataFrame
    data = data_df.values
    target = target_df.values.ravel()

    # Retrieve feature names from the column names of the DataFrame
    feature_names = data_df.columns
    if feature_names.shape[0] != np.unique(feature_names).shape[0]:
        warnings.warn("Feature names specified are not unique. "
                      "Assigning a unique label.\n")
        feature_names_u = np.array(feature_names, dtype=str)
        for it, _ in enumerate(feature_names_u):
            feature_names_u[it] += '_{}'.format(it)
            np.savetxt("id_correspondence.csv",
                       np.stack((np.array(feature_names),
                                 feature_names_u), axis=-1),
                       delimiter=",", fmt='%s')

    # Select target names
    target_names = np.sort(np.unique(target))

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=feature_names)




def load_npy(data_path, target_path, return_X_y=False,
             samples_on='row'):


    r"""Read data matrix and labels vector from files.

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

    # DATA + LABELS
    data = np.load(data_path)
    target = np.load(target_path)

    if samples_on == 'col':
        data = data.T

    if return_X_y:
        return data, target

    # META INFO
    # with open(meta_path, 'rb') as f:
        # res = pkl.load(f)

        # feature_names = np.array(res['feature_names'])
        # target_names = np.array(res['target_names'])

    n, d = data.shape

    target_names = ['T{}'.format(str(i+1)) for i in range(n)]
    feature_names = ['F{}'.format(str(i+1)) for i in range(d)]

    target_names = np.array(target_names)
    feature_names = np.array(feature_names)

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=feature_names)


def copy_files(data_path, target_path, base_path, session_folder):
    """Create a hard link of all dataset files inside the session folder.

    Create a hard link of all files required by the dataset,
    conveniently renaming them (the destination name is the
    corresponding key in the dataset_files dictionary).

    Parameters
    ----------

    data_path : string.
        The path to the csv file containing the `data`.
    target_path : string.
        The path to the csv file containing the `target` (labels).
    base_path : string
        The base path relative to which files are stored.
    sessio_folder : string
        The folder inside which files links are being created.
    """
    while not os.path.exists(session_folder):
        time.sleep(0.5)
    # print("\n{} created".format(session_folder))

    dataset_files = {'data': data_path, 'labels': target_path}

    for link_name in dataset_files.keys():
        # os.link(
        shutil.copy2(
            os.path.join(base_path, dataset_files[link_name]),  # SRC
            os.path.join(session_folder, link_name)             # DST
        )


class Dataset(object):
    """Main class for containing data and labels."""

    def __init__(self, dataset_files, dataset_options, is_analysis=False):
        """Initialize the class.

        Parameters
        ----------

        is_analysis : bool
            When loading the dataset during the analysis, files
            have been renamed and moved in the session folder;
            therefore only the keys are used to determine files'
            paths.
        """
        if is_analysis:
            aux = {}

            for k in dataset_files.keys():
                aux[k] = k

            self._dataset_files = aux
        else:
            self._dataset_files = dataset_files

        self._dataset_options = dataset_options

        self.positive_label = dataset_options.get('positive_label', None)
        self.negative_label = dataset_options.get('negative_label', None)
        self.multiclass = dataset_options.get('multiclass', False)
        # if self.positive_label is None and not self.multiclass:
        #     warnings.warn(
        #         "Positive label unspecified for binary classification "
        #         "problems. If you want a multiclass learning, please "
        #         "specify multiclass=True in the dataset_options dictionary.")

    def get_file(self, file_key):
        return self._dataset_files[file_key]

    def get_all_files(self):
        return self._dataset_files

    def get_option(self, option_key):
        return self._dataset_options[option_key]

    def get_all_options(self):
        return self._dataset_options

    def load_dataset(self, base_path):
        raise NotImplementedError("Abstract method")

    def copy_files(self, base_path, session_folder):
        """Create a hard link of all dataset files inside the session folder.

        Create a hard link of all files required by the dataset,
        conveniently renaming them (the destination name is the
        corresponding key in the dataset_files dictionary).

        Parameters
        ----------

        base_path : string
            The base path relative to which files are stored.

        sessio_folder : string
            The folder inside which files links are being created.
        """
        while not os.path.exists(session_folder):
            time.sleep(0.5)
        # print("\n{} created".format(session_folder))

        for link_name in self._dataset_files.keys():
            # os.link(
            shutil.copy2(
                os.path.join(base_path, self.get_file(link_name)),  # SRC
                os.path.join(session_folder, link_name)             # DST
            )

@deprecated('Use load_csv()')
class DatasetCSV(Dataset):
    """Dataset composed by data matrix and labels vector.

    Matrices are stored in two CSV files.
    """

    def load_dataset(self, base_path):
        r"""Read data matrix and labels vector from files.

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

        # DATA
        poslab = self._dataset_options.pop('positive_label', None)
        neglab = self._dataset_options.pop('negative_label', None)
        multiclass = self._dataset_options.pop('multiclass', False)
        samples_on = self._dataset_options.pop('samples_on', 'col')
        pd_data = pd.read_csv(data_path, **self._dataset_options)

        if samples_on == 'col':
            pd_data = pd_data.transpose()

        # Retrieve feature names from the column names of the DataFrame
        feature_names = pd_data.columns
        if feature_names.shape[0] != np.unique(feature_names).shape[0]:
            warnings.warn("Feature names specified are not unique. "
                          "Assigning a unique label.\n")
            feature_names_u = np.array(feature_names, dtype=str)
            for it, _ in enumerate(feature_names_u):
                feature_names_u[it] += '_{}'.format(it)
            np.savetxt("id_correspondence.csv",
                       np.stack((np.array(feature_names),
                                 feature_names_u), axis=-1),
                       delimiter=",", fmt='%s')

        ##################
        # LABELS
        ##################
        # Before loading labels, remove parameters that were likely specified
        # for data only.
        self._dataset_options.pop('usecols', None)
        pd_labels = pd.read_csv(labels_path, **self._dataset_options)

        if poslab is None and not multiclass:
            uv = np.sort(np.unique(pd_labels.as_matrix()))
            if len(uv) != 2:
                raise Exception("More than two unique values in the labels "
                                "array.")
            poslab = uv[0]

        if neglab is not None:
            # remove samples that are not positive nor negative.
            # NOTE: unexpected behaviour when poslab is None and neglab is not
            idx = np.logical_or(pd_labels[pd_labels.columns[0]] == neglab,
                                pd_labels[pd_labels.columns[0]] == poslab)
            pd_labels = pd_labels[idx]
            pd_data = pd_data[idx]

        if not multiclass:
            # Convert labels to -1/+1
            pd_labels = pd_labels.applymap(lambda x: 1 if x == poslab else -1)
            if np.unique(pd_labels).shape[0] != 2:
                raise ValueError("labels are not those of a bi-class problem")

        data = pd_data.as_matrix()
        labels = pd_labels.as_matrix().ravel()
        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of samples in data do not correspond "
                             "to the number of samples in labels.")
        return data, labels, feature_names


class DatasetNPY(Dataset):
    """"Dataset composed by data matrix and labels vector.

    Matrices are stored in two NPY files, while features/samples names are
    stored in a pkl file.
    """

    def load_dataset(self, base_path):
        r"""Read data matrix and labels vector from files.

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

        # DATA + LABELS
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

        # LABELS TO -1/+1
        if not self.get_option('positive_label') is None:
            poslab = self.get_option('positive_label')
        else:
            uv = np.sort(np.unique(labels))
            if len(uv) != 2:
                raise Exception(
                    "More than two unique values in the labels array")

            poslab = uv[0]

        def _to_plus_minus(x):
            """Convert labels to -1 / +1."""
            return +1. if x == poslab else -1.

        labels_mapped = map(_to_plus_minus, labels)
        labels = np.array(labels_mapped)
        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of samples in data do not correspond "
                             "to the number of samples in labels.")

        return data, labels, feature_names


class DatasetXLS(Dataset):
    """Dataset composed by data matrix and labels vector.

    Matrices are stored in XLS files.
    """

    def load_dataset(self, base_path):
        r"""Read data matrix and labels vector from files.

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

        # DATA
        poslab = self._dataset_options.pop('positive_label', None)
        samples_on = self._dataset_options.pop('samples_on', 'col')
        pd_data = pd.read_excel(data_path, **self._dataset_options)

        if samples_on == 'col':
            pd_data = pd_data.transpose()

        # Retrieve feature names from the column names of the DataFrame
        feature_names = pd_data.columns
        if feature_names.shape[0] != np.unique(feature_names).shape[0]:
            import sys
            sys.stderr.write("Feature names specified are not unique. "
                             "Assigning a unique label.\n")
            feature_names_u = np.array(feature_names, dtype=str)
            for it, _ in enumerate(feature_names_u):
                feature_names_u[it] += '_{}'.format(it)
            np.savetxt("id_correspondence.csv",
                       np.stack((np.array(feature_names),
                                 feature_names_u), axis=-1),
                       delimiter=",", fmt='%s')

        # if not self.get_option('data_preprocessing') is None:
        # ### TODO Check!!!
        #     # if rank == 0:
        #         # print("Preprocessing data...")
        #
        #     self.get_option('data_preprocessing').load_data(pd_data)
        #     pd_data = self.get_option('data_preprocessing').process()

        ##################
        # LABELS
        ##################
        # Before loading labels, remove parameters that were likely specified
        # for data only.
        self._dataset_options.pop('usecols', None)
        pd_labels = pd.read_xls(labels_path, **self._dataset_options)
        if poslab is None:
            uv = np.sort(np.unique(pd_labels.as_matrix()))
            if len(uv) != 2:
                raise Exception("More than two unique values in the labels "
                                "array.")
            poslab = uv[0]

        def _to_plus_minus(x):
            """Convert labels to -1 / +1."""
            return +1. if x == poslab else -1.

        # Convert labels to -1/+1
        pd_labels_mapped = pd_labels.applymap(_to_plus_minus)

        data = pd_data.as_matrix()
        labels = pd_labels_mapped.as_matrix().ravel()
        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of samples in data do not correspond "
                             "to the number of samples in labels.")
        return data, labels, feature_names
