#!/usr/bin/env python

import os, sys, shutil

from optparse import OptionParser



def main():

    parser = OptionParser()

    parser.add_option("-s", "--sample-data",
                      action="store_true", dest="sample_data", default=False,
                      help="Also copy sample data to the deployment folder")

    (options, args) = parser.parse_args()

    ### If no argument is given, assume installation in home folder
    if len(args) < 1:
        deployment_folder = os.path.expanduser("~")
    else:
        deployment_folder = args[0]

    ### The folder where scripts are located
    scripts_folder = os.path.dirname(os.path.realpath(__file__))

    ### COPY PALLADIO LIBRARY FOLDER
    shutil.copytree(os.path.join(scripts_folder, "..", 'palladio'), os.path.join(deployment_folder, 'palladio'))

    ### COPY L1L2PY LIBRARY FOLDER
    shutil.copytree(os.path.join(scripts_folder, "..", 'ext_libraries', 'l1l2py'), os.path.join(deployment_folder, 'l1l2py'))

    ### COPY SCRIPTS
    shutil.copy(os.path.join(scripts_folder, "pd_run.py"), os.path.join(deployment_folder, "pd_run.py"))
    shutil.copy(os.path.join(scripts_folder, "pd_analysis.py"), os.path.join(deployment_folder, "pd_analysis.py"))

    ### IF REQUESTED, COPY SAMPLE DATA
    if options.sample_data:
        shutil.copytree(os.path.join(scripts_folder, "..", 'example'), os.path.join(deployment_folder, 'palladio_example'))


if __name__ == '__main__':
    main()
