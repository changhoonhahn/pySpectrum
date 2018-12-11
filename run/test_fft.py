''' test pyspectrum.fft and make sure it's the same as the fortran 
code output 
'''
import numpy as np 
from scipy.io import FortranFile 
from pyspectrum import fft as FFT
from pyspectrum import util as UT 


if __name__=="__main__": 

    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock1000000']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    delta = FFT.FFTperiodic(xyz, Lbox=2600, Ngrid=360, silent=False) 
    
    f = FortranFile(''.join([UT.dat_dir(), 'zmap.dcl']), 'r')
    #delta_assign = f.read_reals(np.float32) 
    #delta_assign = np.reshape(delta_assign, (3,10000000), order='F') 
    #delta_assign = delta_assign[:,:1000000]
    delta_assign = f.read_reals(np.complex64) 
    delta_assign = np.reshape(delta_assign, (360, 360, 360), order='F') 
    print np.array_equal(delta, delta_assign)

    '''
    # read in fortran code FFT output 
    f = FortranFile(''.join([UT.dat_dir(), 'FFT.BoxN1.mock1000000.Ngrid360']), 'r')
    _ = f.read_ints(dtype=np.int32)
    delt = f.read_reals(dtype=np.complex64) 
    delt = np.reshape(delt, (181, 360, 360), order='F')

    print delta[0,:,:]
    print delt[0,:,:]
    '''
