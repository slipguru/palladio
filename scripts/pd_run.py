#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import os, sys

from palladio import main

from palladio.utils import sec_to_timestring

# Script entry ----------------------------------------------------------------
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('incorrect number of arguments')
    config_file_path = sys.argv[1]
    
    main(os.path.abspath(config_file_path))
