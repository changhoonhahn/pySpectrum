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
    delta = pySpec.FFTperiodic(xyz, fft='pyfftw', Lbox=2600, Ngrid=360, silent=False) 
    delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 
    print('--pyfftw: %f sec' % ((time.time() - t0)/60.)) 

    #t0 = time.time() 
    #k0, p0k0, cnts = pySpec.Pk_periodic(delta_fft0, Lbox=2600) 
    #print('--python periodic Pk: %f min' % ((time.time() - t0)/60.)) 
    
    t0 = time.time() 
    output = pySpec.Bk123_periodic(delta_fft, Nmax=40, Ncut=3, step=3, fft_method='pyfftw') 
    print('Bk123_periodic takes %f' % (time.time() - t0))
    bk, qk = output['b123'], output['q123'] 
    print('--python bk: %f min' % ((time.time() - t0)/60.)) 
    np.savetxt(''.join([UT.dat_dir(), '_test_pySpec.BISP.BoxN1.mock.Ngrid360']), np.vstack([bk, qk]).T)

    _,_,_, bk_ref, qk_ref = np.loadtxt(''.join([UT.dat_dir(), 'BISP.BoxN1.mock.Ngrid360']), 
            unpack=True, usecols=[0,1,2,6,7]) 
    print(np.abs(bk - bk_ref).max())
    print((np.abs(bk - bk_ref)/bk).max())
    print(np.abs(qk - qk_ref).max())
    print((np.abs(qk - qk_ref)/qk).max())

    '''
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.arange(len(bk)), bk, c='C0', s=5) 
    sub.scatter(np.arange(len(bk_ref)), bk_ref, c='k', s=3) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('Triangle Index', fontsize=25) 
    sub.set_xlim([0, len(bk)]) 
    sub.set_yscale("log") 
    fig.savefig(''.join([UT.fig_dir(), 'bk_test.png']), bbox_inches='tight') 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.arange(len(bk)), qk, c='C0', s=5) 
    sub.scatter(np.arange(len(bk_ref)), qk_ref, c='k', s=3) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('Triangle Index', fontsize=25) 
    sub.set_xlim([0, len(bk)]) 
    sub.set_yscale("log") 
    fig.savefig(''.join([UT.fig_dir(), 'qk_test.png']), bbox_inches='tight') 
    '''
