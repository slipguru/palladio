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
    
class Scale(Preproc):
    
    pass