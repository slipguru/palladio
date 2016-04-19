import os, sys, imp

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from scipy import stats

import cPickle as pkl

def analyze_experiment(exp_folder, config):
    
    with open(os.path.join(exp_folder, 'result.pkl'), 'r') as f:
        result = pkl.load(f)
        
    # result['prediction_ts_list'] # one element for each value of mu
    
    
    Y_ts = result['labels_ts'] # the actual labels
    
    analysis_results = dict()
    analysis_results['summaries'] = list()
    analysis_results['accuracy'] = list()
    analysis_results['balanced_accuracy'] = list()
    
    for aux_y in result['prediction_ts_list']:
        
        ### XXX TODO fix cases in which Y_lr = 0
        Y_lr = np.sign(aux_y.ravel())
        # Y_lr = np.sign(Y_lr-0.1)
        
        
        analysis_results['accuracy'].append((Y_ts == Y_lr).sum()/float(len(Y_ts)))
        
        
        TP = np.sum((Y_lr == 1) * (Y_ts == Y_lr))
        FP = np.sum((Y_lr == 1) * (Y_ts != Y_lr))
        TN = np.sum((Y_lr == -1) * (Y_ts == Y_lr))
        FN = np.sum((Y_lr == -1) * (Y_ts != Y_lr))
        
        # print TP, TN, FP, FN
        
        balanced_accuracy = 0.5 * ( (TP / float(TP + FN)) + (TN / float(TN + FP)) )
        
        analysis_results['balanced_accuracy'].append(balanced_accuracy)
        
        analysis_results['selected_list'] = result['selected_list']
        
        # summary['MCC']
        
    return analysis_results

def analyze_experiments(base_folder):
    """
    Perform a preliminar analysis on all experiments producing a summary
    for each experiment that will be subsequently used for the aggregate analysis
    
    Parameters
    ----------
    
    base_folder : string, the folder containing ALL experiments (regular and permutations)
    """
    
    # Local imports, in order to select backend on startup
    from matplotlib import pyplot as plt
    from l1l2signature import internals as l1l2_core
    from l1l2signature import utils as l1l2_utils
    
    config = imp.load_source('config', os.path.join(base_folder, 'config.py'))

    print("#--------------------------------------------------------------------")
    print('Reading data... ')

    data_path = os.path.join(base_folder, 'data_file')
    labels_path = os.path.join(base_folder, 'labels_file')
    
    br = l1l2_utils.BioDataReader(data_path, labels_path,
                                  config.sample_remover,
                                  config.variable_remover,
                                  config.delimiter,
                                  config.samples_on)
    
    data = br.data
    labels = br.labels
    sample_names = br.samples
    probeset_names = br.variables
    
    # print len(probeset_names)
    
    MCC_regular = list()
    MCC_permutation = list()
    
    acc_regular = list()
    acc_permutation = list()
    
    selected_regular = dict(zip(probeset_names, np.zeros((len(probeset_names),))))
    selected_permutation = dict(zip(probeset_names, np.zeros((len(probeset_names),))))
    
    for exp_folder in [os.path.join(base_folder, x) for x in os.listdir(base_folder)]:
        
        if os.path.isdir(exp_folder):
            
            analysis_result = analyze_experiment(exp_folder, config)
            
            selected_probesets = probeset_names[analysis_result['selected_list'][0]]
            
            if exp_folder.split('/')[1].startswith('regular'):
                
                # MCC_regular.append(analysis_result['summaries'][0]['MCC'])
                # acc_regular.append(analysis_result['summaries'][0]['balanced_accuracy'])
                # acc_regular.append(analysis_result['summaries'][0]['accuracy'])
                # acc_regular.append(analysis_result['accuracy'][0])
                acc_regular.append(analysis_result['balanced_accuracy'][0])
                
                for p in selected_probesets:
                    selected_regular[p] += 1
                
            elif exp_folder.split('/')[1].startswith('permutation'):
                
                # MCC_permutation.append(analysis_result['summaries'][0]['MCC'])
                # acc_permutation.append(analysis_result['summaries'][0]['balanced_accuracy'])
                # acc_permutation.append(analysis_result['summaries'][0]['accuracy'])
                # acc_permutation.append(analysis_result['accuracy'][0])
                acc_permutation.append(analysis_result['balanced_accuracy'][0])
                
                for p in selected_probesets:
                    selected_permutation[p] += 1
                
            else:
                print "error"
            
    out = {
        'v_regular' : np.array(acc_regular),
        'v_permutation' : np.array(acc_permutation),
        'selected_regular' : selected_regular,
        'selected_permutation' : selected_permutation,
    }
    
    return out

def plotting(v_regular, v_permutation, base_folder):
    
    nbins = 15
    
    fig, ax = plt.subplots()
    
    args = {
        'norm_hist' : False,
        'kde' : False,
        # 'bins' : np.arange(0,1.05,0.05)
        'bins' : np.arange(0,105,5)
        
    }
    
    sns.distplot(v_permutation*100, label = 'Permutation tests', color = 'r', ax = ax, **args)
    sns.distplot(v_regular*100, label = 'Regular experiments', color = 'b', ax = ax, **args)
    
    ### Fit a gaussian with permutation data
    (mu, sigma) = stats.norm.fit(v_permutation*100)
    
    # print (mu, sigma)
    
    kstest = stats.kstest(v_regular*100, 'norm', args=(mu, sigma))
    print kstest
    
    rstest = stats.ranksums(v_regular, v_permutation)
    print rstest
    
    
    plt.xlabel("Balanced Accuracy (%)", fontsize="large")
    plt.ylabel("Absolute Frequency", fontsize="large")
    
    ### Determine limits for the x axis
    x_min = v_permutation.min() - v_permutation.mean()/10
    x_max = v_regular.max() + v_regular.mean()/10
    
    plt.xlim([x_min*100,x_max*100])
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize="large")

    plt.savefig(os.path.join(base_folder, 'permutation_acc_distribution.pdf'))

def main():
    
    base_folder = sys.argv[1]
    
    out = analyze_experiments(base_folder)
    
    v_regular, v_permutation = out['v_regular'], out['v_permutation']
    selected_regular, selected_permutation = out['selected_regular'], out['selected_permutation']
    
    ### Manually sorting stuff
    sorted_keys_regular = sorted(selected_regular, key=selected_regular.__getitem__)
    sorted_keys_permutation = sorted(selected_permutation, key=selected_permutation.__getitem__)
    
    with open(os.path.join(base_folder, 'signature_regular.txt'), 'w') as f:
        for k in reversed(sorted_keys_regular):
            # print("{} : {}".format(k,selected_regular[k]))
            f.write("{} : {}\n".format(k,selected_regular[k]))
            
    with open(os.path.join(base_folder, 'signature_permutation.txt'), 'w') as f:
        for k in reversed(sorted_keys_permutation):
            f.write("{} : {}\n".format(k,selected_permutation[k]))
    
    
    
    plotting(v_regular, v_permutation, base_folder)
    
    
    
    
    pass

if __name__ == '__main__':
    main()