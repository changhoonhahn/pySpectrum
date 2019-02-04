'''

General utility functions 

'''
import os
import numpy as np 


def ijl_order(i_k, j_k, l_k, typ='GM'): 
    ''' triangle configuration ordering, returns indices
    '''
    i_bq = np.arange(len(i_k))
    if typ == 'GM': # same order as Hector's
        i_bq_new = [] 

        l_usort = np.sort(np.unique(l_k))
        for l in l_usort: 
            j_usort = np.sort(np.unique(j_k[l_k == l]))
            for j in j_usort: 
                i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                for i in i_usort: 
                    i_bq_new.append(i_bq[(i_k == i) & (j_k == j) & (l_k == l)])
    else: 
        raise NotImplementedError
    return np.array(i_bq_new)


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

