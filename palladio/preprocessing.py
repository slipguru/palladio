import numpy as np

class Preproc():
    
    def __init__(self, X):
        """
        Saves the data and the labels
        """
        
        self._mean = None
        
        self.set_X(X)
        
    def set_X(self, X):
        
        self._X = X

class Center(Preproc):
    
    def __init__(self, X):
        Preproc.__init__(self, X)
    
    def get_mean(self):
        
        if self._mean is None:
            self._mean = self._X.mean(0)
            
        return self._mean
    
    def center(self, X):
        
        return X - self.get_mean()

class DataPreprocessing:
    
    def process(self):
        
        raise Exception("Stub method")
        pass
    
    def load_data(self, data):
        """
        
        Parameters
        ----------
        
        data : pandas.core.frame.DataFrame
          the X matrix
        """
        self._data = data
    
    pass
    

class FeatureChooser(DataPreprocessing):
    """
    """
    
    def __init__(self, id_list):
        """
        
        Parameters
        ----------
        
        id_list : list
          The list of ids of the feature to keep
        """
        
        self._id_list = id_list
    
    def process(self):
        """
        Choose features based on the id
        """
        
        return self._data.loc[self._id_list]
    
    pass
    
class Scale(Preproc):
    
    pass

