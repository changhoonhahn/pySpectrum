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
xyz     = fnbox['xyz'][...]
vxyz    = fnbox['vxyz'][...]
nhalo   = xyz.shape[0]
nbar = float(nhalo)/Lbox**3

N = xyz.shape[1] # number of positions 
kf = 2 * np.pi / Lbox 

# Pk from real-space powerspectrum function 
t0 = time.time() 
pkout = pySpec.Pk_periodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, fft='pyfftw', silent=False) 
print('--pySpec.Pk_periodic: %f sec' % ((time.time() - t0)/60.)) 

t0 = time.time() 
# Pk from redshift-space powerspectrum function  
_pkout = pySpec.Pk_periodic_rsd(xyz.T, Lbox=Lbox, Ngrid=Ngrid, rsd=2, fft='pyfftw', code='fortran', silent=False) 
print('--pySpec.Pk_periodic_rsd: %f sec' % ((time.time() - t0)/60.)) 

# Pk from original fortran code 
f_og = os.path.join(UT.dat_dir(), 'plk.BoxN1.mock') 
k_og, p0k_og, p2k_og, p4k_og = np.loadtxt(f_og, unpack=True, usecols=[0, 1, 2, 3]) 
p0k_og *= (2.*np.pi)**3
p2k_og *= (2.*np.pi)**3
p4k_og *= (2.*np.pi)**3
p0k_og -= 1./nbar # shot noise correction

fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
sub.plot(pkout['k'], pkout['p0k'], c='C0', label='Pk periodic') 
sub.plot(_pkout['k'], _pkout['p0k'], ls='--', c='C1', label='Pk periodic rsd ($\ell=0$)') 
sub.plot(_pkout['k'], _pkout['p2k'], ls='-.', c='C1', label='Pk periodic rsd ($\ell=2$)') 
sub.plot(_pkout['k'], _pkout['p4k'], ls=':', c='C1', label='Pk periodic rsd ($\ell=4$)') 
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
fig.savefig(os.path.join(UT.dat_dir(), 'test_pk.rspace.png'), bbox_inches='tight')

'''
fpk = os.path.join(UT.dat_dir(), 'test_pk.BoxN1.dat')
hdr = ('halo powerspectrum for mneut=%f, realization %i, redshift bin %i; k_f = 2pi/%.1f, Nhalo=%i' %
       (0.0, 1, 4, Lbox, xyz.shape[0]))
np.savetxt(fpk, np.array([k, p0k, p_sn, cnts]).T, fmt='%.5e %.5e %.5e %i', delimiter='\t', header=hdr)
'''

