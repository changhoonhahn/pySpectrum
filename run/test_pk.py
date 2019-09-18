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

# real-space power/bispectrum with pySpectrum
t0 = time.time() 
pkout = pySpec.Pk_periodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, fft='pyfftw', silent=False) 
print('--pySpec.Pk_periodic: %f sec' % ((time.time() - t0)/60.)) 

k = pkout['k']
p0k = pkout['p0k']
p_sn = np.repeat(pkout['p0k_sn'], len(k)) 
cnts = pkout['counts']

'''
fpk = os.path.join(UT.dat_dir(), 'test_pk.BoxN1.dat')
hdr = ('halo powerspectrum for mneut=%f, realization %i, redshift bin %i; k_f = 2pi/%.1f, Nhalo=%i' %
       (0.0, 1, 4, Lbox, xyz.shape[0]))
np.savetxt(fpk, np.array([k, p0k, p_sn, cnts]).T, fmt='%.5e %.5e %.5e %i', delimiter='\t', header=hdr)
'''

t0 = time.time() 
# redshift-space powerspectrum with pySpectrum
_pkout = pySpec.Pk_periodic_rsd(xyz.T, Lbox=Lbox, Ngrid=Ngrid, rsd=2, fft='pyfftw', code='fortran', silent=False) 
print('--pySpec.Pk_periodic_rsd: %f sec' % ((time.time() - t0)/60.)) 

_p0k = _pkout['p0k']
_cnts = _pkout['counts']
print(p0k[:10])
print(_p0k[:10])
print(p0k[-10:])
print(_p0k[-10:])
print(((p0k - _p0k)/p0k))
