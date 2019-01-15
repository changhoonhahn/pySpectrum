'''

test whether reducing the precision to complex64 impacts the bispectrum 
calculations significantly

'''
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
    print('--pyfftw: %f min' % ((time.time() - t0)/60.)) 

    t0 = time.time() # ~0.090432 sec
    _,_,_, bk32, qk32, _ = pySpec.Bk123_periodic(delta_fft, Nmax=40, Ncut=3, step=3, bit=32, fft_method='pyfftw') 
    print('--python bk: %f min' % ((time.time() - t0)/60.)) 
    
    _,_,_, bk, qk, _ = pySpec.Bk123_periodic(delta_fft, Nmax=40, Ncut=3, step=3, bit=64, fft_method='pyfftw') 
    print('--python bk: %f min' % ((time.time() - t0)/60.)) 

    dbk = bk - bk32
    dqk = qk - qk32
    print('%f < dbk < %f' % (dbk.min(), dbk.max())) 
    print('%f < dqk < %f' % (dqk.min(), dqk.max())) 
    dbk /= bk
    dqk /= qk
    print('%f < dbk/bk < %f' % (np.abs(dbk).min(), np.abs(dbk).max())) 
    print('%f < dqk/qk < %f' % (np.abs(dqk).min(), np.abs(dqk).max())) 
    _,_,_, bk_ref, qk_ref = np.loadtxt(''.join([UT.dat_dir(), 'BISP.BoxN1.mock.Ngrid360']), 
            unpack=True, usecols=[0,1,2,6,7]) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.arange(len(bk)), bk, c='C0', s=5) 
    sub.scatter(np.arange(len(bk)), bk32, c='r', s=5) 
    sub.scatter(np.arange(len(bk_ref)), bk_ref, c='k', s=5) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('Triangle Index', fontsize=25) 
    sub.set_xlim([0, len(bk)]) 
    fig.savefig(''.join([UT.fig_dir(), 'bk_32bit_test.png']), bbox_inches='tight') 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.arange(len(bk)), qk, c='C0', s=5) 
    sub.scatter(np.arange(len(bk)), qk32, c='r', s=5) 
    sub.scatter(np.arange(len(bk_ref)), qk_ref, c='k', s=5) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('Triangle Index', fontsize=25) 
    sub.set_xlim([0, len(bk)]) 
    fig.savefig(''.join([UT.fig_dir(), 'qk_32bit_test.png']), bbox_inches='tight') 
