#!/bin/python 
import time 
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec
# -- plotting -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


if __name__=="__main__": 
    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z
    
    # pyspectrum P(k) calculation 
    t0 = time.time() # ~0.120479 sec
    delta0 = pySpec.FFTperiodic(xyz, fft='pyfftw', Lbox=2600, Ngrid=360, silent=False) 
    delta_fft0 = pySpec.reflect_delta(delta0, Ngrid=360) 
    print('--pyfftw: %f sec' % ((time.time() - t0)/60.)) 
    #t0 = time.time() # ~0.286884 sec 
    #delta1 = pySpec.FFTperiodic(xyz, fft='fftw3', Lbox=2600, Ngrid=360, silent=False) 
    #delta_fft1 = pySpec.reflect_delta(delta1, Ngrid=360) 
    #print('--fftw3: %f sec' % ((time.time() - t0)/60.)) 
    t0 = time.time() # ~0.090432 sec
    delta2 = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=2600, Ngrid=360, silent=False) 
    delta_fft2 = pySpec.reflect_delta(delta2, Ngrid=360) 
    print('--fortran: %f sec' % ((time.time() - t0)/60.)) 

    '''
    t0 = time.time() # 
    k0, p0k0 = pySpec.Pk_periodic(delta_fft0, Lbox=2600) 
    print('--python periodic Pk: %f sec' % ((time.time() - t0)/60.)) 
    #k1, p0k1 = pySpec.Pk_periodic(delta_fft1, Lbox=2600, lang='fortran') 
    t0 = time.time() # 
    k2, p0k2 = pySpec.Pk_periodic_f77(delta2, Lbox=2600) 
    print('--fortran periodic Pk: %f sec' % ((time.time() - t0)/60.)) 

    k_ref, p0k_ref = np.loadtxt(''.join([UT.dat_dir(), 'PK.BoxN1.mock.Ngrid360']), unpack=True, usecols=[0,1]) 
    kf = 2.*np.pi/2600.

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot(k0, p0k0, c='C0') 
    #sub.plot(k1, p0k1, c='C1') 
    sub.plot(k2, p0k2, c='C2') 
    sub.plot(k_ref, (2.*np.pi)**3*p0k_ref, c='k', ls=':') 
    sub.set_ylabel('$P(k)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([1e-3, 1.]) 
    sub.set_xscale('log') 
    fig.savefig(''.join([UT.fig_dir(), 'pk_test.png']), bbox_inches='tight') 
    '''

    t0 = time.time() # ~0.090432 sec
    _,_,_, bk, qk = pySpec.Bk123_periodic(delta_fft0, Nmax=40, Ncut=3, step=3, fft_method='pyfftw') 
    print('--python bk: %f sec' % ((time.time() - t0)/60.)) 
    _,_,_, bk_ref, qk_ref = np.loadtxt(''.join([UT.dat_dir(), 'BISP.BoxN1.mock.Ngrid360']), 
            unpack=True, usecols=[0,1,2,6,7]) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.arange(len(bk)), qk, c='C0', s=5) 
    sub.scatter(np.arange(len(bk_ref)), qk_ref, c='k', s=5) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('Triangle Index', fontsize=25) 
    sub.set_xlim([0, len(bk)]) 
    fig.savefig(''.join([UT.fig_dir(), 'bk_test.png']), bbox_inches='tight') 
