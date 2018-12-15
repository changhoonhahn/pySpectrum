'''

time profiling for the bispectrum 

'''
import time 
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec


if __name__=="__main__": 
    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z
    
    delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=2600, Ngrid=360, silent=False) 
    delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 
    print('-------------------------') 

    for nthread in [1,2,3,4,5,6]: 
        t0 = time.time()
        # calculate bispectrum 
        i_k, j_k, l_k, b123, q123 = pySpec.Bk123_periodic(
                delta_fft, Nmax=40, Ncut=3, step=3, 
                fft_method='pyfftw', nthreads=nthread, 
                silent=True) 
        print('Bispectrum calculation with %i cpus takes %f mins' % (nthread, ((time.time() - t0)/60.)))
