#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Perform precision-recall & ROC analysis for a given pd experiment."""

from __future__ import division
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns

from pd_multiple_analysis import confusion_matrix
from pd_multiple_analysis import selection_scores
sns.set_context("paper")


def parse_signature(txt):
    """Parse signaure_*.txt and return the content as pandas DataFrame."""
    with open(txt, 'r') as f:
        lines = f.readlines()

    feat_names = list()
    selection_frequency = list()
    for l in lines:
        if l[0] != "=":
            splitl = l.split(' ')
            feat_names.append(int(splitl[0].split('_')[-1]))
            selection_frequency.append(float(splitl[-1]))

    df = pd.DataFrame(data=selection_frequency,
                      index=feat_names,
                      columns=['Selection_frequency'])
    return df


def make_curve(x, y, xlabel, ylabel, filename):
    """Draw and save ROC or PR curves."""
    auc = np.trapz(x, y, dx=0.001)  # area under the curve

    sns.plt.figure()
    sns.plt.clf()
    sns.plt.plot(x, y)
    sns.plt.xlim([0, 1])
    sns.plt.ylim([0, 1])
    sns.plt.xlabel(xlabel)
    sns.plt.ylabel(ylabel)
    sns.plt.title("AUC: {}".format(auc))
    sns.plt.savefig(filename)

    return auc


def curves(df):
    """Perform precision-recall & ROC analysis and save curve."""
    # range of possible thresholds
    thresh = np.arange(0, 101)
    d = df.shape[0]

    # Scores containers
    prec_list = list()
    rcll_list = list()
    fpr_list = list()

    # Test every possible threshold and compute scores
    for t in thresh:
        positions = np.where(df['Selection_frequency'] >= t)[0]
        selected = df.index[positions]
        tp, tn, fp, fn = confusion_matrix(selected, d)
        prec, rcll, _ = selection_scores(tp, tn, fp, fn)
        fpr = fp / (fp + tn)
        prec_list.append(prec)
        rcll_list.append(rcll)
        fpr_list.append(fpr)

    # Precision - Recall arrays
    prec = np.array(prec_list)
    rcll = np.array(rcll_list)
    fpr = np.array(fpr_list)

    # Make and save plot
    make_curve(rcll, prec, xlabel='Recall', ylabel='Precision',
               filename="pr-curve.pdf")
    make_curve(fpr, rcll, xlabel='False positive rate', ylabel='Recall',
               filename="roc-curve.pdf")


def main(args):
    """Perform roc analysis on a given pd experiment."""
    folder = args.folder

    # Read signature regular
    signature = os.path.join(folder, 'signature_regular.txt')
    df = parse_signature(signature)

    # Make PR curve
    curves(df)

if __name__ == '__main__':
    from palladio import __version__
    parser = argparse.ArgumentParser(description='palladio script for '
                                                 'precision-recall analysis'
                                                 ' of a given experiment.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("folder", help="experiments results directory")
    args = parser.parse_args()

    main(args)
