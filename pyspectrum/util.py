'''

General utility functions 

'''
import os

def dat_dir():  
    ''' local directory for dumping files. This is mainly used for  
    the test runs.
    '''
    if os.environ.get('PYSPEC_DIR') is None: 
        raise ValueError("set $PYSPEC_DIR environment varaible!") 
    return os.environ.get('PYSPEC_DIR') 


def code_dir(): 
    ''' location of the repo directory. set $PYSPEC_CODEDIR 
    environment varaible in your bashrc file. 
    '''
    if os.environ.get('PYSPEC_CODEDIR') is None: 
        raise ValueError("set $PYSPEC_CODEDIR environment varaible!") 
    return os.environ.get('PYSPEC_CODEDIR') 
    

def fig_dir(): 
    ''' directory to dump all the figure files 
    '''
    if os.path.isdir(code_dir()+'figs/'):
        return code_dir()+'figs/'
    else: 
        raise ValueError("create figs/ folder in $FOMOSPEC_CODEDIR directory for figures")


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    if os.path.isdir(code_dir()+'doc/'):
        return code_dir()+'doc/'
    else: 
        raise ValueError("create doc/ folder in $FOMOSPEC_CODEDIR directory for documntation")

