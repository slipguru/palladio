#!/usr/bin/env python

import os
import shutil
import argparse

# from palladio import __version__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Palladio script for '
                                                 'deploying.')
    # If no argument is given, assume installation in home folder
    parser.add_argument('deployment_folder', nargs='?',
                        default=os.path.expanduser("~"),
                        help="Specify the deployment folder")
    # parser.add_argument('--version', action='version',
    #                    version='%(prog)s v'+__version__)

    parser.add_argument("-s", "--sample-data",
                        action="store_true", dest="sample_data", default=False,
                        help="Also copy sample data to the deployment folder")
    args = parser.parse_args()

    # The folder where scripts are located
    scripts_folder = os.path.dirname(os.path.realpath(__file__))

    # COPY PALLADIO LIBRARY FOLDER
    shutil.copytree(os.path.join(scripts_folder, "..", 'palladio'),
                    os.path.join(args.deployment_folder, 'palladio'))

    # COPY L1L2PY LIBRARY FOLDER
    shutil.copytree(os.path.join(scripts_folder, "..",
                                 'ext_libraries', 'l1l2py'),
                    os.path.join(args.deployment_folder, 'l1l2py'))

    # COPY SCRIPTS
    shutil.copy(os.path.join(scripts_folder, "pd_run.py"),
                os.path.join(args.deployment_folder, "pd_run.py"))
    shutil.copy(os.path.join(scripts_folder, "pd_analysis.py"),
                os.path.join(args.deployment_folder, "pd_analysis.py"))

    # IF REQUESTED, COPY SAMPLE DATA
    if args.sample_data:
        shutil.copytree(os.path.join(scripts_folder, "..", 'example'),
                        os.path.join(args.deployment_folder,
                                     'palladio_example'))
