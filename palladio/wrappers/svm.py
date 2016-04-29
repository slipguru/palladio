from sklearn.linear_model import SGDClassifier

class svmClassifier:
    
    def __init__(self, params):
        
        self._clf = SGDClassifier(**params)
        
        
    pass