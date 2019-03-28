#!/bin/python 
import os 
import h5py 
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


Lbox = 2600. # box size
Ngrid = 360  # fft grid size
kf = 2*np.pi/Lbox # fundament mode 

# read in Nseries box data
fnbox = h5py.File(os.path.join(UT.dat_dir(), 'BoxN1.hdf5'), 'r') 
xyz = fnbox['xyz'].value 
vxyz = fnbox['vxyz'].value 
nhalo = xyz.shape[0]
nbar = float(nhalo)/Lbox**3

# real-space power/bispectrum with pySpectrum
t0 = time.time() 
pspec = pySpec.Pk_periodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
print('--pySpec.Pk_periodic: %f sec' % ((time.time() - t0)/60.)) 
t0 = time.time() 
bispec = pySpec.Bk_periodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
print('--pySpec.Bk_periodic: %f sec' % ((time.time() - t0)/60.)) 

# compare real-space bispectrum with Roman's output 
f_bk_rs = os.path.join(UT.dat_dir(), 'bk.BoxN1.mock') 
_i, _j, _l, _pi, _pj, _pl, _b123, _q123 = np.loadtxt(f_bk_rs, unpack=True, usecols=[0,1,2,3,4,5,6,7]) 

# compare output powerspectrum 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
isort = np.argsort(bispec['i_k1'])
sub.plot(pspec['k'], pspec['p0k'], c='C1', label='pySpec Pk code') 
sub.plot(kf*bispec['i_k1'][isort], bispec['p0k1'][isort], c='C0', label='pySpec Bk code') 
isort = np.argsort(_i)
sub.plot(kf*_i[isort], (2*np.pi/kf)**3 * _pi[isort] - 1./nbar, c='k', ls='--', label="Roman's code") 
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_ylim([3e3, 2e5]) 
sub.set_yscale('log') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
fig.savefig(os.path.join(UT.dat_dir(), 'test_p0k.BoxN1.png'), bbox_inches='tight')

# compare output bispectrum 
fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.scatter(range(len(bispec['i_k1'])), bispec['b123'], c='C0', s=5, label='pySpec')  # shot noise corrected
b_sn = (2*np.pi/kf)**3 * (_pi + _pj + _pl)/nbar - 2/nbar**2 
sub.scatter(range(len(_i)), (2*np.pi/kf)**6 * _b123 - b_sn, c='C1', s=1, label="Roman's code") 
sub.legend(loc='upper right') 
sub.set_xlabel('triangles', fontsize=25)
sub.set_xlim([0, len(_i)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25)
sub.set_yscale('log') 
sub.set_ylim([1e7, 8e9])
fig.savefig(''.join([UT.dat_dir(), 'test_b0k.BoxN1.png']), bbox_inches='tight')

########################################
# calculate redshift-space bispectrum 
########################################
xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=2600.) 

# z-space power/bispectrum with pySpectrum
t0 = time.time() 
pspec = pySpec.Pk_periodic(xyz_s, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
print('--pySpec.Pk_periodic: %f sec' % ((time.time() - t0)/60.)) 
t0 = time.time() 
bispec = pySpec.Bk_periodic(xyz_s, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
print('--pySpec.Bk_periodic: %f sec' % ((time.time() - t0)/60.)) 

# compare real-space bispectrum with Roman's output 
f_bk_rs = os.path.join(UT.dat_dir(), 'bk.BoxN1.rsd_z.mock') 
_i, _j, _l, _pi, _pj, _pl, _b123, _q123 = np.loadtxt(f_bk_rs, unpack=True, usecols=[0,1,2,3,4,5,6,7]) 

# compare output powerspectrum 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
isort = np.argsort(bispec['i_k1'])
sub.plot(pspec['k'], pspec['p0k'], c='C1', label='pySpec Pk code') 
sub.plot(kf*bispec['i_k1'][isort], bispec['p0k1'][isort], c='C0', label='pySpec Bk code') 
isort = np.argsort(_i)
sub.plot(kf*_i[isort], (2*np.pi/kf)**3 * _pi[isort] - 1./nbar, c='k', ls='--', label="Roman's code") 
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_ylim([3e3, 2e5]) 
sub.set_yscale('log') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
fig.savefig(os.path.join(UT.dat_dir(), 'test_p0k.BoxN1.zspace.png'), bbox_inches='tight')

# compare output bispectrum 
fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.scatter(range(len(bispec['i_k1'])), bispec['b123'], c='C0', s=5, label='pySpec')  # shot noise corrected
b_sn = (2*np.pi/kf)**3 * (_pi + _pj + _pl)/nbar - 2/nbar**2 
sub.scatter(range(len(_i)), (2*np.pi/kf)**6 * _b123 - b_sn, c='C1', s=1, label="Roman's code") 
sub.legend(loc='upper right') 
sub.set_xlabel('triangles', fontsize=25)
sub.set_xlim([0, len(_i)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25)
sub.set_yscale('log') 
sub.set_ylim([1e7, 8e9])
fig.savefig(''.join([UT.dat_dir(), 'test_b0k.BoxN1.zspace.png']), bbox_inches='tight')

