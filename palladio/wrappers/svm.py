from sklearn.linear_model import SGDClassifier

class svmClassification:

    def __init__(self, params):
        self._clf = SGDClassifier(**params)
