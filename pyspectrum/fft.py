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
    
    # assign galaxies to grid (checked with fortran) 
    _delta = np.zeros([2*Ngrid, Ngrid, Ngrid], dtype=np.float32, order='F') # even indices (real) odd (complex)
    fEstimate.assign2(xyz_gals, _delta, kf_ks, Ng, Ngrid) 

    delta = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')
    delta.real = _delta[::2,:,:] 
    delta.imag = _delta[1::2,:,:] 
    if not silent: print('galaxy positions assigned to grid') 

    # FFT delta (checked with fortran code, more or less matches)
    #_empty = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex128')
    #_ifft_empty = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex128')
    #fftw_obj = pyfftw.FFTW(_empty, _ifft_empty, axes=(0,1,2,), 
    #        direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE', ))
    #ifft_delta = fftw_obj(delta, normalise_idft=False)
    fftw_ob = pyfftw.builders.ifftn(delta, axes=(0,1,2,))#, planner_effort='FFTW_ESTIMATE')
    pyfftw.interfaces.cache.enable()
    ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F') 
    ifft_delta[:,:,:] = fftw_ob(normalise_idft=False)

    if not silent: print('galaxy grid FFTed') 
    _ifft_delta = ifft_delta.copy() 
    # fcombine 
    fEstimate.fcomb(ifft_delta,Ng,Ngrid) 
    if not silent: print('fcomb complete') 
    return ifft_delta[:Ngrid/2+1,:,:]
