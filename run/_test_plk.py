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


Lbox = 2600. # box size
Ngrid = 360  # fft grid size
kf = 2*np.pi/Lbox # fundament mode 

# read in Nseries box data
fnbox = h5py.File(os.path.join(UT.dat_dir(), 'BoxN1.hdf5'), 'r') 
xyz     = fnbox['xyz'][...]
vxyz    = fnbox['vxyz'][...]
nhalo   = xyz.shape[0]
nbar = float(nhalo)/Lbox**3

N = xyz.shape[1] # number of positions 
kf = 2 * np.pi / Lbox 

# apply RSD
xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=Lbox) 

t0 = time.time() 
# Plk from redshift-space powerspectrum function  
pkout = pySpec.Pk_periodic_rsd(xyz_s, Lbox=Lbox, Ngrid=Ngrid, rsd=2, fft='pyfftw', code='fortran', silent=False) 
print('--pySpec.Pk_periodic_rsd: %f sec' % ((time.time() - t0)/60.)) 

# Plk from original fortran code 
f_og = os.path.join(UT.dat_dir(), 'plk.BoxN1.rsd_z.mock') 
k_og, p0k_og, p2k_og, p4k_og = np.loadtxt(f_og, unpack=True, usecols=[0, 1, 2, 3]) 
p0k_og *= (2.*np.pi)**3
p2k_og *= (2.*np.pi)**3
p4k_og *= (2.*np.pi)**3
p0k_og -= 1./nbar # shot noise correction

fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
sub.plot(pkout['k'], pkout['p0k'], c='C0', label='pySpec ($\ell=0$)') 
sub.plot(pkout['k'], pkout['p2k'], c='C1', label='pySpec ($\ell=2$)') 
sub.plot(pkout['k'], pkout['p4k'], c='C2', label='pySpec ($\ell=4$)') 
sub.plot(k_og, p0k_og, ls='--', c='k', label='original') 
sub.plot(k_og, p2k_og, ls='-.', c='k') 
sub.plot(k_og, p4k_og, ls=':', c='k') 
sub.legend(loc='upper right', handletextpad=0.2, fontsize=15)
sub.set_ylabel('real-space $P(k)$', fontsize=25) 
sub.set_ylim([1e3, 3e5]) 
sub.set_yscale('log') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
fig.savefig(os.path.join(UT.dat_dir(), 'test_pk.zspace.png'), bbox_inches='tight')

# P(k,mu) from original fortran code 
f_og = os.path.join(UT.dat_dir(), 'pkmu.BoxN1.rsd_z.mock') 
_k_og, _mu_og, _pkmu_og, _nmodes_og = np.loadtxt(f_og, unpack=True, usecols=[0, 1, 4, 3]) 
_pkmu_og *= (2.*np.pi)**3

k_og    = _k_og.reshape((180, 20))[:,10:]
mu_og   = _mu_og.reshape((180, 20))[:,10:]

_pkmu_og = _pkmu_og.reshape((180,20))
_nmodes_og = _nmodes_og.reshape((180,20))
nmodes_og = _nmodes_og[:,:10][:,::-1] + _nmodes_og[:,10:]
pkmu_og = ((_pkmu_og[:,:10] * _nmodes_og[:,:10])[:,::-1] + (_pkmu_og[:,10:] * _nmodes_og[:,10:]))/nmodes_og

# P(k,mu) from nbodykit 
import nbodykit.lab as NBlab
objs = {} 
objs['Position'] = xyz.T
objs['Velocity'] = vxyz.T
objs['RSDPosition'] = xyz_s

cat = NBlab.ArrayCatalog(objs, BoxSize=Lbox)
mesh = cat.to_mesh(window='tsc', Nmesh=360, BoxSize=Lbox, compensated=True, position='RSDPosition')
r = NBlab.FFTPower(mesh, mode='2d', dk=kf, kmin=0.5*kf, Nmu=10, los=[0,0,1])
print(r.power)

fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(121)
cm = sub.pcolormesh(pkmu_og, norm=LogNorm(vmin=1e3, vmax=1e5))
sub.set_yscale('log') 
sub.set_ylim(1, 180) 
sub.set_title('original') 
sub = fig.add_subplot(122)
cm = sub.pcolormesh(pkout['p_kmu'], norm=LogNorm(vmin=1e3, vmax=1e5))
sub.set_yscale('log') 
sub.set_ylim(1, 180) 
sub.set_title('pySpectrum') 
fig.savefig(os.path.join(UT.dat_dir(), 'test_pk.2d.zspace.png'), bbox_inches='tight')
