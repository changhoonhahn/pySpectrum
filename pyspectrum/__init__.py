import os 

# Path to data files required for cosmos
_PYSPEC_DAT_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dat')
_PYSPEC_FIG_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'fig')
_PYSPEC_DOC_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'doc')

def dat_dir():
    return _PYSPEC_DAT_DIR

def fig_dir():
    return _PYSPEC_FIG_DIR

def doc_dir():
    return _PYSPEC_DOC_DIR
