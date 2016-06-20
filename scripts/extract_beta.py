"""
Retrieve betas from OLS experiments
"""

import numpy as np
import cPickle as pkl

import os, sys

def beta_average():
    
    session_folder = sys.argv[1]
    
    
    beta = None
    intercept = None
    
    i = 0
    
    for f in os.listdir(session_folder):
        
        
        if f.startswith('regular') and os.path.isdir(os.path.join(session_folder, f)):
            
            exp_folder = os.path.join(session_folder, f)
            results_file = os.path.join(exp_folder, 'result.pkl')
            
            with open(results_file, 'r') as f:
                res = pkl.load(f)
                
            if beta is None:
                beta = np.array(res['model'])
                betas = beta[:,np.newaxis]
            else:
                tmp = np.array(res['model'])
                beta += tmp
                betas = np.hstack((betas, tmp[:,np.newaxis]))
                
            if intercept is None:
                intercept = res['intercept']
                intercepts = [intercept]
            else:
                intercept += res['intercept']
                intercepts.append(intercept)
                
            i += 1
            
    
    
    intercepts = np.array(intercepts)
    
    beta = beta/i
    intercept = intercept/i
    
    print betas.mean(1)
    print intercept.mean()
    
def check_splits():
    
    session_folder = sys.argv[1]
    
    i = 0
    
    for f in os.listdir(session_folder):
        
        
        if f.startswith('regular') and os.path.isdir(os.path.join(session_folder, f)):
        # if f.startswith('permutation') and os.path.isdir(os.path.join(session_folder, f)):
            
            exp_folder = os.path.join(session_folder, f)
            results_file = os.path.join(exp_folder, 'in_split.pkl')
            
            with open(results_file, 'r') as f:
                res = pkl.load(f)
                
            print res
                
            i += 1
            
    

if __name__ == '__main__':
    beta_average()
    
    # check_splits()