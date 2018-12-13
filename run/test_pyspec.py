#!/bin/python 
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
    delta_fft = pySpec.FFTperiodic(xyz, Lbox=2600, Ngrid=360, silent=False) 
    _delta_fft = pySpec.read_fortFFT(file=''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']))

    k, p0k = pySpec.Pk_periodic(delta_fft, Lbox=2600) 
    _k, _p0k = pySpec.Pk_periodic(_delta_fft, Lbox=2600) 

    k_ref, p0k_ref = np.loadtxt(''.join([UT.dat_dir(), 'PK.BoxN1.mock.Ngrid360']), unpack=True, usecols=[0,1]) 
    kf = 2.*np.pi/2600.

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot(k, p0k, c='C0') 
    sub.plot(_k, _p0k, c='C1') 
    sub.plot(k_ref, (2.*np.pi)**3*p0k_ref, c='k', ls='--') 
    sub.set_ylabel('$P(k)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([1e-3, 1.]) 
    sub.set_xscale('log') 
    fig.savefig(''.join([UT.fig_dir(), 'pk_test.png']), bbox_inches='tight') 
    raise ValueError

    _,_,_, bk = pySpec.Bk123_periodic(delta_fft, Nmax=10, Ncut=3, step=3, fft_method='pyfftw') 
    _,_,_, _bk = pySpec.Bk123_periodic(_delta_fft, Nmax=10, Ncut=3, step=3, fft_method='pyfftw') 
    _,_,_, bk_ref = np.loadtxt(''.join([UT.dat_dir(), 'BISP.BoxN1.mock.Ngrid360']), 
            unpack=True, usecols=[0,1,2,3]) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.arange(len(bk)), bk, c='C0', s=5) 
    sub.scatter(np.arange(len(_bk)), _bk, c='C1', s=5) 
    sub.scatter(np.arange(len(bk_ref)), bk_ref, c='k', s=5) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('Triangle Index', fontsize=25) 
    sub.set_xlim([0, np.max(len(bk), len(_bk))]) 
    fig.savefig(''.join([UT.fig_dir(), 'bk_test.png']), bbox_inches='tight') 
