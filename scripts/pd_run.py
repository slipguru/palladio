#!/usr/bin/env python
"""Main palladio script."""

import os
import argparse

import palladio as pd

DATA_EXT = ('csv', 'npy')  # list of allowed data extensions
MODELS = ('l1l2', 'elasticnet', 'logit')  # implemented methods


def init_main():
    """Start palladio run."""
    from palladio import __version__
    parser = argparse.ArgumentParser(description='palladio script for '
                                                 'running the framework.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("-f", "--format", dest="format", action="store",
                        help="select data input format (csv or npy)",
                        default='csv')
    parser.add_argument("-m", "--model", dest="model", action="store",
                        help="select model for the standard config file in "
                             "(l1l2, elasticnet, logit)",
                        default='l1l2')
    parser.add_argument("configuration_file", help="specify config file",
                        default='pd_config.py')
    args = parser.parse_args()

    if args.create:
        # Argument check
        if args.format.lower() not in DATA_EXT:
            raise ValueError("Format '{}' not understood. Please specify one"
                             "of {}.\n".format(args.format.lower(),
                                               DATA_EXT))

        if args.model.lower() not in MODELS:
            raise ValueError("Model '{}' not understood. Please specify one"
                             "of {}.\n".format(args.model.lower(),
                                               MODELS))

        # Define which config_file needs to be loaded
        config_file = 'config_' + args.model.lower() + \
                      '_' + args.format.upper() + '.py'

        std_config_path = os.path.join(pd.__path__[0], 'config_templates',
                                       config_file)
        # Check for .pyc
        if std_config_path.endswith('.pyc'):
            std_config_path = std_config_path[:-1]
        # Check if the file already exists
        if os.path.exists(args.configuration_file):
            parser.error("palladio configuration file already exists")

        # Copy the config file
        from shutil import copy
        copy(std_config_path, args.configuration_file)
    else:
        pd.main(os.path.abspath(args.configuration_file))


if __name__ == '__main__':
    init_main()
