'''
'''
import pyfftw
import numpy as np 
import estimator as fEstimate


def FFTperiodic(gals, Lbox=2600., Ngrid=360): 
    '''
    '''
    kf_ks = np.float32(float(Ngrid) / Lbox)
    #Ngrid = np.int32(Ngrid)
    Ng = np.int32(gals.shape[1]) # number of galaxies

    # position of galaxies 
    xyz_gals = np.zeros([3, Ng], dtype=np.float32, order='F') 
    xyz_gals[0,:] = gals[0,:] 
    xyz_gals[1,:] = gals[1,:]
    xyz_gals[2,:] = gals[2,:] 

    #delta = pyfftw.n_byte_align_empty((2*Ngrid, Ngrid, Ngrid), 16, dtype='complex64', order='F')
    delta = np.zeros([Ngrid, Ngrid, Ngrid], dtype=np.float32, order='F') 
    fEstimate.assign2(xyz_gals, delta, kf_ks, Ng, Ngrid) 
    
    # FFT delta 
    fft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F')
    fftw_ob = pyfftw.builders.ifftn(delta, planner_effort='FFTW_ESTIMATE')
    pyfftw.interfaces.cache.enable()
    _fft_delta = fftw_ob(delta)
    fft_delta[:,:,:] = _fft_delta.copy()

    # fcombine 
    fEstimate.fcomb(fft_delta,Ng,Ngrid) 
    return fft_delta
