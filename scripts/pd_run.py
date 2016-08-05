#!/usr/bin/env python
"""Main palladio script."""

import os
import sys
import argparse

from palladio import main

__extensions__ = ('csv', 'npy')  # list of allowed data extesions
__models__ = ('l1l2', 'elasticnet', 'logit')  # list of implmenented methods

# Script entry ----------------------------------------------------------------
if __name__ == '__main__':
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
        import shutil
        import palladio as pd

        # Argument check
        if args.format.lower() not in __extensions__:
            sys.stderr.write("Format {} not understood. Please specify one"
                             "of {}.\n".format(args.format.lower(),
                                               __extensions__))
            sys.exit(-1)

        if args.model.lower() not in __models__:
            sys.stderr.write("Model {} not understood. Please specify one"
                             "of {}.\n".format(args.model.lower(),
                                               __models__))
            sys.exit(-1)

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
        shutil.copy(std_config_path, args.configuration_file)
    else:
        main(os.path.abspath(args.configuration_file))
