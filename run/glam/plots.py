#!/bin/python 
'''

plot GLAM P(k) and B(k1,k2,k3)


'''
import os, sys 
import h5py 
import numpy as np 
from scipy.io import FortranFile
# -- nbodykit -- 
import nbodykit.lab as NBlab
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import plots as Plots 
from pyspectrum import pyspectrum as pySpec
# -- eMaNu -- 
from emanu import forwardmodel as FM
# -- plotting -- 
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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


dir_dat = '/home/chhahn/scratch/glam/'

Lbox = 1000. 
kf = 2.*np.pi/Lbox


def glam_pk(ireal, logmlim=13., rsd=False): 
    str_rsd = ['', '.rsd'][rsd]
    f_pell = os.path.join(dir_dat, 
            'pySpec.Plk.glam.CatshortV.0136.%s.halo.mlim1e%i.Lbox%.f.Ngrid360%s.dat' % (str(ireal).zfill(4), int(logmlim), Lbox, str_rsd)) 

    k, p0k, cnts = np.loadtxt(f_pell, skiprows=1, unpack=True, usecols=[0,1,2]) 
    return k, p0k 


def glam_bk(ireal, logmlim=13., rsd=False): 
    str_rsd = ['', '.rsd'][rsd]
    f_b123 = os.path.join(dir_dat, 
            'pySpec.B123.glam.CatshortV.0136.%s.halo.mlim1e%i.Lbox%.f.Ngrid360.Nmax40.Ncut3.step3.pyfftw%s.dat' % (str(ireal).zfill(4), int(logmlim), Lbox, str_rsd)) 
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123, 
            skiprows=1, unpack=True, usecols=range(10)) 
    k1 = i_k *kf 
    k2 = j_k *kf 
    k3 = l_k *kf 
    return k1, k2, k3, p0k1, p0k2, p0k3, b123


# plot real-space P0k 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)

for ireal in range(1, 16): 
    k, _p0k = glam_pk(ireal, logmlim=13., rsd=False) 
    sub.plot(k, _p0k, lw=1)
    if ireal == 1: p0k = _p0k 
    else: p0k += _p0k
p0k /= 15.

sub.plot(k, p0k, c='k', lw=2, label='avg') 

sub.legend(loc='lower left', fontsize=20) 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_yscale('log') 
fig.savefig(os.path.join(dir_dat, 'glam_p0k.real.png'), bbox_inches='tight')

# plot redshift-space P0k 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)

for ireal in range(1, 16): 
    k, _p0k = glam_pk(ireal, logmlim=13., rsd=True) 
    sub.plot(k, _p0k, lw=1) 
    if ireal == 1: p0k = _p0k 
    else: p0k += _p0k
p0k /= 15.

sub.plot(k, p0k, c='k', lw=2, label='avg') 

sub.legend(loc='lower left', fontsize=20) 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_yscale('log') 
fig.savefig(os.path.join(dir_dat, 'glam_p0k.rsd.png'), bbox_inches='tight')


# plot real-space B0(k1,k2,k3)
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)

for ireal in range(1, 16): 
    k1, k2, k3, _, _, _, _b123 = glam_bk(ireal, logmlim=13., rsd=False) 
    if ireal == 1: b123 = _b123
    else: b123 += _b123
b123 /= 15.
klim = (k1 < 0.5) & (k2 < 0.5) & (k3 < 0.5) 

sub.scatter(range(np.sum(klim)), b123[klim], c='k', s=1)
sub.set_xlabel(r'$k_3 < k_2 < k_1 < 0.5~h/{\rm Mpc}$ triangle index', fontsize=25) 
sub.set_xlim([0, np.sum(klim)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
sub.set_yscale('log') 
fig.savefig(os.path.join(dir_dat, 'glam_b123.real.png'), bbox_inches='tight')


# plot redshift-space B0(k1,k2,k3)
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)

for ireal in range(1, 16): 
    k1, k2, k3, _, _, _, _b123 = glam_bk(ireal, logmlim=13., rsd=True) 
    if ireal == 1: b123 = _b123
    else: b123 += _b123
b123 /= 15.

sub.scatter(range(np.sum(klim)), b123[klim], c='k', s=1)
sub.set_xlabel(r'$k_3 < k_2 < k_1 < 0.5~h/{\rm Mpc}$ triangle index', fontsize=25) 
sub.set_xlim([0, np.sum(klim)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
sub.set_yscale('log') 
fig.savefig(os.path.join(dir_dat, 'glam_b123.rsd.png'), bbox_inches='tight')
