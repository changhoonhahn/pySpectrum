'''

calculate powerspectrum 

'''
import time 
import numpy as np 
import matplotlib.pyplot as plt 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec


if __name__=="__main__": 
    # read in FFTed density grid
    f_fft = ''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']) 
    delta = pySpec.read_fortFFT(file=f_fft) 

    # calculate powerspectrum monopole  
    k, p0k = pySpec.Pk_periodic(delta) 
    kf = 2.*np.pi/2600.
    
    _k, _p0k = np.loadtxt(''.join([UT.dat_dir(), 'PK.BoxN1.mock.Ngrid360']), unpack=True, usecols=[0,1]) 

    plt.plot(kf*k, p0k/(2.*np.pi)**3/kf**3) 
    plt.plot(_k, _p0k, ls='--') 
    plt.xscale("log")
    plt.yscale("log") 
    plt.show() 
