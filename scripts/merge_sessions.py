#!/usr/bin/env python

import argparse

import shutil, os

def main():
    
    parser = argparse.ArgumentParser(description='Merge the partial results of two or more sessions into a new one.')
    
    parser.add_argument('session_folders', metavar='session_folders', type=str, nargs='+',
                    help='The folders of all sessions that must be merged')
    
    parser.add_argument('-d', '--destination', help='Destination folder', type= str, required = True)
    
    args = parser.parse_args()
    
    ### Create destination folder
    if not os.path.exists(args.destination):
        os.mkdir(args.destination)
        
    ### Copy files (config, data and labels file)
    for f in ['config.py', 'data_file', 'labels_file']:
        shutil.copy2(os.path.join(args.session_folders[0], f), os.path.join(args.destination, f))
    
    
    
    
    print args
    
    
    pass

if __name__ == '__main__':
    main()