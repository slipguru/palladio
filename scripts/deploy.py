#!/usr/bin/env python

import os, sys, shutil

def main():
    
    ### If no argument is given, assume installation in home folder
    if len(sys.argv) < 2:
        deployment_folder = os.path.expanduser("~")
    else:
        deployment_folder = sys.argv[1]
    
    ### The folder where scripts are located
    scripts_folder = os.path.dirname(os.path.realpath(__file__))
    
    ### COPY PALLADIO LIBRARY FOLDER
    shutil.copytree(os.path.join(scripts_folder, "..", 'palladio'), os.path.join(deployment_folder, 'palladio'))
    
    ### COPY L1L2PY LIBRARY FOLDER
    shutil.copytree(os.path.join(scripts_folder, "..", 'ext_libraries', 'l1l2py'), os.path.join(deployment_folder, 'l1l2py'))
    
    ### COPY SCRIPTS
    shutil.copy(os.path.join(scripts_folder, "pd_run.py"), os.path.join(deployment_folder, "pd_run.py"))
    shutil.copy(os.path.join(scripts_folder, "pd_analysis.py"), os.path.join(deployment_folder, "pd_analysis.py"))
    
if __name__ == '__main__':
    main()