from __future__ import division

import numpy as np

import sys

sys.path.append('.')

import palladio

from numpy.linalg import pinv

from palladio.wrappers.ols import OLSClassification

from sklearn.linear_model import SGDClassifier, LinearRegression, SGDRegressor, ElasticNet

def main():
    
    n,p = 500,10
    
    nts = n/4
    ntr = 3*nts
    
    X = np.random.normal(size = (n,p))
    
    beta = np.random.normal(size = (p,))
    
    Y = X.dot(beta)
    Y = np.sign(Y) # make it a binary classification problem
    
    print("# of positives = {}".format((Y == +1).sum()))
    print("# of negatives = {}".format((Y == -1).sum()))
    
    Xtr = X[:ntr,:]
    Xts = X[ntr:,:]
    
    Ytr = Y[:ntr]
    Yts = Y[ntr:]
    
    clf = OLSClassification()
    clf.setup(Xtr, Ytr, Xts, Yts)
    
    result = clf.run()
    
    result['prediction_ts_list']
    
    print "SGD", (result['prediction_ts_list'] != Yts).sum()/len(Yts)
    
    
    
    ######################################################################
    
    
    clf2 = LinearRegression(fit_intercept = False)
    clf2.fit(Xtr, Ytr)
    
    Ylr = np.sign(clf2.predict(Xts))
    
    print "LinearRegressor", (Ylr != Yts).sum()/len(Yts)
    
    
    
    
    ######################################################################
    
    
    
    
    clf3 = ElasticNet(fit_intercept = False, l1_ratio = 1.0, alpha = 0.0)
    clf3.fit(Xtr, Ytr)
    
    Ylr = np.sign(clf3.predict(Xts))
    
    print "Elasticnet", (Ylr != Yts).sum()/len(Yts)
    
    # ElasticNet
    
    ######################################################################
    
    beta = (pinv(Xtr.T.dot(Xtr)).dot(Xtr.T)).dot(Ytr)
    
    Ylr = np.sign(Xts.dot(beta))
    
    print "Manual", (Ylr != Yts).sum()/len(Yts)
    
    
    
    pass

if __name__ == '__main__':
    
    main()