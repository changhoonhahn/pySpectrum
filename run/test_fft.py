'''
'''
import numpy as np 
from pyspectrum import fft as FFT
from pyspectrum import util as UT 


if __name__=="__main__": 
    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock100']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    delta= FFT.FFTperiodic(xyz, Lbox=2600, Ngrid=10) 
    print delta[0,:,:]
