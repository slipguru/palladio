import numpy as np

from sklearn.linear_model import SGDClassifier

from .classification import Classification

class SGDClassification(Classification):
    
    def setup(self, Xtr, Ytr, Xts, Yts):
        
        self._Xtr = Xtr
        self._Ytr = Ytr
        
        self._Xts = Xts
        self._Yts = Yts
        
        