'''
'''
import pyfftw
import numpy as np 
import estimator as fEstimate


def FFTperiodic(gals, Lbox=2600., Ngrid=360, silent=True): 
    '''
    '''
    kf_ks = np.float32(float(Ngrid) / Lbox)
    Ng = np.int32(gals.shape[1]) # number of galaxies

    # position of galaxies (checked with fortran) 
    xyz_gals = np.zeros([3, Ng], dtype=np.float32, order='F') 
    xyz_gals[0,:] = np.clip(gals[0,:], 0., Lbox*(1.-1e-6))
    xyz_gals[1,:] = np.clip(gals[1,:], 0., Lbox*(1.-1e-6))
    xyz_gals[2,:] = np.clip(gals[2,:], 0., Lbox*(1.-1e-6))
    if not silent: print('%i galaxy positions saved' % Ng) 

    #delta = pyfftw.n_byte_align_empty((2*Ngrid, Ngrid, Ngrid), 16, dtype='complex64', order='F')
    _delta = np.zeros([2*Ngrid, Ngrid, Ngrid], dtype=np.float32, order='F') 
    fEstimate.assign2(xyz_gals, _delta, kf_ks, Ng, Ngrid) 
    if not silent: print('galaxy positions assigned to grid') 
    delta = np.zeros([Ngrid, Ngrid, Ngrid], dtype=np.complex64, order='F') 
    delta = _delta[::2,:,:]  # even indices are reals
    delta = j*_delta[np.arange(1, Ngrid+1)[::2],:,:] # odds are complex 
    return delta 
    ''' 
    # FFT delta 
    fft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F')
    fftw_ob = pyfftw.builders.ifftn(delta, planner_effort='FFTW_ESTIMATE')
    pyfftw.interfaces.cache.enable()
    _fft_delta = fftw_ob(delta)
    fft_delta[:,:,:] = _fft_delta.copy()
    if not silent: print('galaxy grid FFTed') 

    # fcombine 
    fEstimate.fcomb(fft_delta,Ng,Ngrid) 
    if not silent: print('fcomb complete') 
    return fft_delta[:Ngrid/2+1,:,:]
    '''
