#!/usr/bin/env python
import os
import argparse

from palladio import main

# Script entry ----------------------------------------------------------------
if __name__ == '__main__':
    from palladio import __version__
    parser = argparse.ArgumentParser(description='palladio script for '
                                                 'running the framework.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file",
                        default='pd_config.py')
    args = parser.parse_args()

    if args.create:
        import shutil
        import palladio as pd
        std_config_path = os.path.join(pd.__path__[0], 'default_config_files',
                                       'config_l1l2.py')
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
