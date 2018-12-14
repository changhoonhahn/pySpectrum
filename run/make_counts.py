import time 
import numpy as np 
from scipy.io import FortranFile 
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec


if __name__=="__main__":
    counts = pySpec._counts_Bk123(Ngrid=360, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=False) 
    counts_f77 = pySpec._counts_Bk123_f77(Ngrid=360, Nmax=40, Ncut=3, step=3, silent=False) 

    # check counts 
    f = FortranFile(''.join([UT.dat_dir(), 'counts2_n360_nmax40_ncut3_s3']), 'r') 
    _counts = f.read_reals(dtype=np.float64) 
    _counts = np.reshape(_counts, (40, 40, 40), order='F') 
    _counts_swap = np.swapaxes(_counts, 0, 2) # axes are swaped for whatever reason 
    assert np.allclose(_counts_swap, counts_f77) 
    assert np.allclose(_counts_swap, counts) 

    # both the python and fortran wrapped implementations work well 
