#!/usr/bin/env python
"""Main palladio script."""

import argparse
import os
import palladio as pd


def init_main():
    """Start palladio run."""
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
        # Argument check
        config_file = 'default_config.py'

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
        pd.main(config_path=os.path.abspath(args.configuration_file))


if __name__ == '__main__':
    init_main()
