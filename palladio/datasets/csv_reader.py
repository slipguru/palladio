import os

import pandas as pd
import numpy as np

from .reader import Reader

class Reader_csv(Reader):
    """Dataset composed by data matrix and labels vector, stored in two csv files.
    
    
    """
    
    def load_dataset(self, base_path):
        """Read data matrix and labels vector from files.
        
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
        
        ##################
        ###### DATA ######
        ##################
        pd_data = pd.read_csv(data_path, header=0, index_col=0, delimiter = self.get_option('delimiter'))
        
        if self.get_option('samples_on') == 'col':
            pd_data = pd_data.transpose()
        
        ### Retrieve feature names from the column names of the DataFrame
        feature_names = pd_data.columns
        
        if not self.get_option('data_preprocessing') is None:
        ### TODO Check!!!
            # if rank == 0:
                # print("Preprocessing data...")
                
            self.get_option('data_preprocessing').load_data(pd_data)
            pd_data = self.get_option('data_preprocessing').process()

        ##################
        ##### LABELS #####
        ##################
        pd_labels = pd.read_csv(labels_path, header=0, index_col=0)
        
        if not self.get_option('positive_label') is None:
            poslab = self.get_option('positive_label')
        else:
            uv = np.sort(np.unique(pd_labels.values))
    
            if len(uv) != 2:
                raise Exception("More than two unique values in the labels array")
    
            poslab = uv[0]
        
        ### Auxiliary function required to convert the labels to
        ### -1/+1
        def _toPlusMinus(x):
            """Converts the values in the labels"""
            if x == poslab:
                return +1.0
            else:
                return -1.0
            
        ### Convert labels to -1/+1
        pd_labels_mapped = pd_labels.applymap(_toPlusMinus)

        data = pd_data.as_matrix()
        labels = pd_labels_mapped.as_matrix().ravel()
    
        return data, labels, feature_names