import os 

# Path to data files required for cosmos
_PYSPEC_DAT_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat')

def dat_dir():
    return _PYSPEC_DAT_DIR
