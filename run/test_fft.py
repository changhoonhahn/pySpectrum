''' test pyspectrum.fft and make sure it's the same as the fortran 
code output 
'''
import pyfftw
import numpy as np 
from scipy.io import FortranFile 
from pyspectrum import fft as FFT
from pyspectrum import util as UT 


if __name__=="__main__": 
    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    delta = FFT.FFTperiodic(xyz, Lbox=2600, Ngrid=360, silent=False) 
    
    #f = FortranFile(''.join([UT.dat_dir(), 'zmap.dcl']), 'r')
    #delta_assign = f.read_reals(np.complex64) 
    #delta_assign = np.reshape(delta_assign, (360, 360, 360), order='F') 
    #print delta[0,:10,0]
    #print delta_assign[0,:10,0]
    #print np.array_equal(delta, delta_assign)
    
    #f = FortranFile(''.join([UT.dat_dir(), 'zmap.fft.dcl']), 'r')
    #delt = f.read_reals(dtype=np.complex64) 
    #delt = np.reshape(delt, (360, 360, 360), order='F') 
    #print delta[:5,0,0]
    #print delt[:5,0,0]

    # read in fortran code FFT output 
    #f = FortranFile(''.join([UT.dat_dir(), 'FFT.BoxN1.mock1000000.Ngrid360']), 'r')
    f = FortranFile(''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']), 'r')
    _ = f.read_ints(dtype=np.int32)
    delt = f.read_reals(dtype=np.complex64) 
    delt = np.reshape(delt, (181, 360, 360), order='F')
    print delta[:10,0,0]
    print delt[:10,0,0]
