import os
import time
import shutil

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
        self._poslab = dataset_options['positive_label']

    def get_file(self, file_key):
        return self._dataset_files[file_key]

    def get_all_files(self):
        return self._dataset_files

    def get_option(self, option_key):
        return self._dataset_options[option_key]

    def get_all_options(self):
        return self._dataset_options

    def load_dataset(self, base_path):
        raise Exception("Abstract method")

    def copy_files(self, base_path, session_folder):
        """Create a hard link of all dataset files inside the session folder

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
            #os.link(
            shutil.copy2(
                os.path.join(base_path, self.get_file(link_name)),  # SRC
                os.path.join(session_folder, link_name)             # DST
            )
