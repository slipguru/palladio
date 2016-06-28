#!/usr/bin/python -u
# -*- coding: utf-8 -*-
import os, sys
import imp
import shutil
import cPickle as pkl
import random

from hashlib import sha512

import pandas as pd

import time

import numpy as np

### Iniziatlize GLOBAL MPI variables (or dummy ones for the single process case)
try:
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    
    IS_MPI_JOB = True
    
except:
    
    comm = None
    size = 1
    rank = 1
    name = 'localhost'
    
    IS_MPI_JOB = False

from palladio import main

from palladio.utils import sec_to_timestring

# Script entry ----------------------------------------------------------------
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('incorrect number of arguments')
    config_file_path = sys.argv[1]
    
    main(os.path.abspath(config_file_path), size, rank, name, comm)
