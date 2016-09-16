#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Iteratively run pd_analysis on the subfolders of an input folder."""
# Script used to collect the results for pyHPC2016.

from __future__ import division
import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context("notebook")


def listdir(folder):
    """Return the directories stored into the input folder."""
    ls = os.listdir(folder)
    ls = map(lambda x: os.path.join(folder, x), ls)
    return filter(lambda x: os.path.isdir(x), ls)


def get_mean(s):
    """Get the mean from the correct line in stats.txt."""
    return float(s.split(' ')[2].strip(','))


def parse_stats(txt):
    """Parse stats.txt and return accuracy, balanced_accuracy and MCC."""
    with open(txt, 'r') as f:
        lines = f.readlines()

    acc = get_mean(lines[6])
    bacc = get_mean(lines[15])
    mcc = get_mean(lines[24])

    return acc, bacc, mcc


def parse_signature(txt, d):
    """Parse signaure_*.txt and return the list of selected variables."""
    feat_names = list()
    with open(txt, 'r') as f:
        for i in range(d):
            line = f.readline()
            if line[0] == '=': break  # using the same threshold as config.py
            feat_names.append(line.split(' ')[0].split('_')[-1])
    return map(int, feat_names)


def confusion_matrix(selected, d):
    """Return TP, TN, FP, FN from selected and known ground truth."""
    if d >= 1000 and d <= 5000:
        _gt = 25
    elif d >= 10000 and d <= 50000:
        _gt = 50
    else:
        _gt = 100

    # The real list of true variables
    gt = set(np.arange(_gt))  # ground truth
    selected = set(selected)

    all_possible = set(np.arange(d))
    not_selected = all_possible.difference(selected)
    gf = set(np.arange(_gt, d))  # ground "false"

    TP = len(selected.intersection(gt))  # true variables that I got right
    FP = len(selected.difference(gt))  # false variables mistaken for true
    TN = len(not_selected.intersection(gf))  # false variables that I got right
    FN = len(not_selected.intersection(gt))  # real variables that are lost

    return TP, TN, FP, FN


def selection_scores(TP, TN, FP, FN):
    """Compute precision, recall and F1 score."""
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    return precision, recall, F1


def save_heatmap(df, title, tag):
    """Create and save heatmaps."""
    sns.plt.figure()
    sns.plt.clf()
    filename = title+tag+".pdf"
    sns.heatmap(df, cmap="YlGn", annot=True, fmt='1.2f')
    sns.plt.title(title)
    sns.plt.ylabel(r'Number of samples $n$')
    sns.plt.xlabel(r'Number of dimensions $d$')
    sns.plt.savefig(filename)
    print("\t{} saved".format(filename))


def make_heatmaps(collection, tag, idx, cols):
    """Generate heatmaps from dictionaries."""
    # Heatmaps containers
    acc = list()
    bacc = list()
    f1 = list()
    prec = list()
    rcll = list()
    for i, n in enumerate(sorted(collection.keys())):
        # Empty Rows of the heatmap
        n_acc = list()
        n_bacc = list()
        n_f1 = list()
        n_prec = list()
        n_rcll = list()
        for j, d in enumerate(sorted(collection[n])):  # fill columns
            n_acc.append(collection[n][d]['acc'])
            n_bacc.append(collection[n][d]['bacc'])
            n_f1.append(collection[n][d]['f1'])
            n_prec.append(collection[n][d]['prec'])
            n_rcll.append(collection[n][d]['rcll'])
        # Store filled rows
        acc.append(n_acc)
        bacc.append(n_bacc)
        f1.append(n_f1)
        prec.append(n_prec)
        rcll.append(n_rcll)

    # From lists of lists to numpy arrays
    acc = pd.DataFrame(data=np.array(acc), index=idx, columns=cols)
    bacc = pd.DataFrame(data=np.array(bacc), index=idx, columns=cols)
    f1 = pd.DataFrame(data=np.array(f1), index=idx, columns=cols)
    prec = pd.DataFrame(data=np.array(prec), index=idx, columns=cols)
    rcll = pd.DataFrame(data=np.array(rcll), index=idx, columns=cols)

    # Save heatmaps
    save_heatmap(acc, 'Accuracy', tag)
    save_heatmap(bacc, 'Balanced Accuracy', tag)
    save_heatmap(f1, 'F1', tag)
    save_heatmap(prec, 'Precision', tag)
    save_heatmap(rcll, 'Recall', tag)


def main(args):
    """Iteratively run starting from root."""
    # Getting arguments
    root = args.base_folder
    analyze = args.analyze
    collect = args.collect

    # Get first level of folders
    folders = listdir(root)
    cpu_collection = dict()
    gpu_collection = dict()

    # Get second level of folders
    for fcount, f in enumerate(folders):
        print('----------------------------------------------')
        # Getting n and d
        fsplit = os.path.split(f)[-1].split('_')
        n, d = int(fsplit[2]), int(fsplit[4])
        if n not in cpu_collection.keys():
            cpu_collection[n] = dict()
            gpu_collection[n] = dict()

        print("Experiment n: {} x p: {}".format(n, d))

        # Each session can be either CPU or GPU
        sessions = listdir(f)
        for s in sessions:
            gpu_flag = os.path.split(s)[-1].split('_')[-1]
            print("Session: {}".format(gpu_flag))

            if analyze:
                print("Analyzing...")
                subprocess.call(["python", "pd_analysis_lite.py", s])
                print("Done.\n")

            if collect:
                print("Collecting...")
                stats = os.path.join(s, 'stats.txt')
                print(" - Reading stats")
                acc, bacc, mcc = parse_stats(stats)
                print("\tACC: {} BACC: {} MCC: {}".format(acc, bacc, mcc))
                signature = os.path.join(s, 'signature_regular.txt')
                print(" - Reading signature_regular")
                selected = parse_signature(signature, d)
                print(" - {} variables selected".format(len(selected)))
                tp, tn, fp, fn = confusion_matrix(selected, d)
                print("\tTP: {} TN: {} FP: {} FN: {}".format(tp, tn, fp, fn))
                precision, recall, f1 = selection_scores(tp, tn, fp, fn)

                # Dump results into the appropriate dictionary
                if gpu_flag == 'NOGPU':
                    cpu_collection[n][d] = {'acc': acc, 'bacc': bacc,
                                            'mcc': mcc, 'prec': precision,
                                            'rcll': recall, 'f1': f1}
                else:
                    gpu_collection[n][d] = {'acc': acc, 'bacc': bacc,
                                            'mcc': mcc, 'prec': precision,
                                            'rcll': recall, 'f1': f1}
                print('----------------------------------------------\n')
    print("{} folders analyzed.\n".format(fcount))

    # Useful quantities for further data frames
    idx = np.sort(cpu_collection.keys())
    cols = np.sort(cpu_collection[idx[0]].keys())

    if collect:
        print(" - Heatmap generation")
        make_heatmaps(cpu_collection, 'CPU', idx, cols)
        make_heatmaps(gpu_collection, 'CPU-GPU', idx, cols)


if __name__ == '__main__':
    from palladio import __version__
    parser = argparse.ArgumentParser(description='palladio script for '
                                                 'multiple result'
                                                 'analysis.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-c", "--collect", dest="collect",
                        action="store_true",
                        help="collect stats", default=False)
    parser.add_argument("-a", "--analyze", dest="analyze",
                        action="store_true",
                        help="launch analysis", default=False)
    parser.add_argument("base_folder", help="specify base results directory")
    args = parser.parse_args()

    main(args)
