#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Palladio script for merging two sessions."""

import argparse
import gzip

import os
import shutil

from six.moves import cPickle as pkl

def main():
    """Summary and plot generation."""
    parser = parse_args()

    merge_folder = parser.merge_folder[0]

    # Create the main merged session folder
    os.mkdir(merge_folder)

    # Create the folder for the experiments
    os.mkdir(os.path.join(merge_folder, 'experiments'))
    os.mkdir(os.path.join(merge_folder, 'logs'))

    # ######################################
    # STEP 1: COPY EXPERIMENTS PKL FILES
    # ######################################

    # Also copy logs and reports

    # Keep track of how many regular and permutation experiments have been
    # copied in the merged session so far
    i_permutation = 0
    i_regular = 0

    i_session = 0

    for sf in parser.session_folders:

        # Copy report
        shutil.copy(
        os.path.join(sf,'report.txt'),
        os.path.join(
            merge_folder,
            "report_{}.txt".format(i_session)
            ),
        )

        # Copy logs
        shutil.copytree(
        os.path.join(sf,'logs'),
        os.path.join(
            merge_folder,
            'logs',
            "logs_{}".format(i_session)
            ),
        )

        for exp_pkl in os.listdir(os.path.join(sf,'experiments')):
            if exp_pkl.endswith(".pkl"):

                # Copy regular experiment
                if exp_pkl.startswith('regular'):

                    shutil.copy(
                    os.path.join(sf,'experiments',exp_pkl),
                    os.path.join(
                        merge_folder,
                        'experiments',
                        "regular_{}.pkl".format(i_regular)
                        ),
                    )

                    i_regular += 1

                # Copy permutation experiment
                if exp_pkl.startswith('permutation'):

                    shutil.copy(
                    os.path.join(sf,'experiments',exp_pkl),
                    os.path.join(
                        merge_folder,
                        'experiments',
                        "permutation_{}.pkl".format(i_permutation)
                        ),
                    )

                    i_permutation += 1

        i_session += 1

    # ######################################
    # STEP 2: COPY AND ADAPT SESSION PKL
    # ######################################

    # Load original session object
    with gzip.open(os.path.join(parser.session_folders[0], 'pd_session.pkl.gz'), 'r') as f:
        pd_session_object = pkl.load(f)

    # Update values for number of regular and permutation experiments
    pd_session_object._n_split_regular = i_regular
    pd_session_object._n_split_permutation = i_permutation

    # Dump the modified object in the merge folder
    with gzip.open(os.path.join(merge_folder, 'pd_session.pkl.gz'), 'w') as f:
        pkl.dump(pd_session_object, f)

    # ######################################
    # STEP 3: COPY AND ADAPT CONFIG FILE
    # ######################################

    # Read content from the original config.py file
    with open(os.path.join(parser.session_folders[0], 'config.py'), 'r') as in_cfg:
        content = in_cfg.readlines()

    # Dump the content in the merged folder
    with open(os.path.join(merge_folder, 'config.py'), 'w') as out_cfg:

        for line in content:

            if line.startswith('n_splits_regular'):
                out_cfg.write("n_splits_regular = {}\n".format(i_regular))
            elif line.startswith('n_splits_permutation'):
                out_cfg.write("n_splits_permutation = {}\n".format(i_permutation))
            else:
                out_cfg.write(line)

def parse_args():
    """Parse arguments."""
    from palladio import __version__
    parser = argparse.ArgumentParser(
        description='palladio script for merging two sessions.')

    parser.add_argument('session_folders', metavar='SESSION_FOLDER', type=str, nargs='+', help='The session folders')

    parser.add_argument('merge_folder', metavar='MERGE_FOLDER', type=str, nargs=1, help='The destination folder where the sessions will be merged')

    return parser.parse_args()


if __name__ == '__main__':
    main()
