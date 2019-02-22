'''
Test Nseries box bispectrum by comparing to outputs of Roman's code
'''
import os 
import h5py 
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

########################################
# calculate real-space bispectrum 
########################################
_delta = pySpec.FFTperiodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
delta = pySpec.reflect_delta(_delta, Ngrid=Ngrid, silent=False)
bisp = pySpec.Bk123_periodic(delta, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 
# save to file for posterity
f_bk = os.path.join(UT.dat_dir(), 'pyspec.test_bk.dat') 
hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/%.f' % Lbox
np.savetxt(f_bk, np.array([bisp['i_k1'], bisp['i_k2'], bisp['i_k3'], bisp['b123'], bisp['q123'], bisp['counts']]).T, 
        fmt='%i %i %i %.5e %.5e %.5e', delimiter='\t', header=hdr) 

# compare real-space bispectrum with Roman's output 
f_bk_rs = os.path.join(UT.dat_dir(), 'bk.BoxN1.mock') 
_i, _j, _l, _pi, _pj, _pl, _b123, _q123 = np.loadtxt(f_bk_rs, unpack=True, usecols=[0,1,2,3,4,5,6,7]) 

# compare output powerspectrum 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
isort = np.argsort(bisp['i_k1'])
sub.plot(kf*bisp['i_k1'][isort], (2*np.pi/kf)**3 * bisp['p0k1'][isort], c='C0', ls=':') 
sub.plot(kf*bisp['i_k1'][isort], (2*np.pi/kf)**3 * bisp['p0k1'][isort] - 1./nbar, c='C0', label='pySpec') 
isort = np.argsort(_i)
sub.plot(kf*_i[isort], (2*np.pi/kf)**3 * _pi[isort] - 1./nbar, c='k', ls='--', label="Roman's code") 
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_ylim([3e3, 2e5]) 
sub.set_yscale('log') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
fig.savefig(os.path.join(UT.dat_dir(), 'test_bk.p0k.png'), bbox_inches='tight')

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
b_sn = (bisp['p0k1'] + bisp['p0k2'] + bisp['p0k3'])/nbar + 1/nbar**2 
sub.plot(range(len(bisp['i_k1'])), (2*np.pi/kf)**6 * bisp['b123'], c='C0', lw=0.5, ls=':')  # shot noise uncorrected
sub.scatter(range(len(bisp['i_k1'])), (2*np.pi/kf)**6 * bisp['b123'] - b_sn, c='C0', s=5, label='pySpec')  # shot noise corrected
b_sn = (_pi + _pj + _pl)/nbar + 1/nbar**2 
sub.scatter(range(len(_i)), (2*np.pi/kf)**6 * _b123 - b_sn, c='k', s=1, label="Roman's code") 
sub.set_xlabel('triangles', fontsize=25)
sub.set_xlim([0, len(_i)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$ (not SN corrected)', fontsize=25)
sub.set_yscale('log') 
sub.set_ylim([1e7, 8e9])
fig.savefig(''.join([UT.dat_dir(), 'test_bk.bk.png']), bbox_inches='tight')

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.scatter(range(len(bisp['i_k1'])), bisp['b123']/_b123, c='C0', s=2, label="pySpec/roman's code") 
sub.set_xlabel('triangles', fontsize=25)
sub.set_xlim([0, len(_i)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$ ratio (not SN corrected)', fontsize=25)
sub.set_ylim([0.95, 1.05])
fig.savefig(''.join([UT.dat_dir(), 'test_bk.bk.ratio.png']), bbox_inches='tight')

########################################
# calculate redshift-space bispectrum 
########################################
xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=2600.) 
_delta = pySpec.FFTperiodic(xyz_s, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
delta = pySpec.reflect_delta(_delta, Ngrid=Ngrid, silent=False)
bisp_s = pySpec.Bk123_periodic(delta, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 
# save to file for posterity
f_bk = os.path.join(UT.dat_dir(), 'pyspec.test_bk.zspace.dat') 
hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/%.f' % Lbox
np.savetxt(f_bk, np.array([bisp['i_k1'], bisp['i_k2'], bisp['i_k3'], bisp['b123'], bisp['q123'], bisp['counts']]).T, 
        fmt='%i %i %i %.5e %.5e %.5e', delimiter='\t', header=hdr) 

# compare real-space bispectrum with Roman's output 
f_bk_rs = os.path.join(UT.dat_dir(), 'bk.BoxN1.rsd_z.mock') 
_i, _j, _l, _pi, _pj, _pl, _b123, _q123 = np.loadtxt(f_bk_rs, unpack=True, usecols=[0,1,2,3,4,5,6,7]) 

# compare output powerspectrum 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
isort = np.argsort(bisp['i_k1'])
sub.plot(kf*bisp['i_k1'][isort], (2*np.pi/kf)**3 * bisp['p0k1'][isort] - 1/nbar, c='C1', ls='--', label='real-space') 
isort = np.argsort(bisp_s['i_k1'])
sub.plot(kf*bisp_s['i_k1'][isort], (2*np.pi/kf)**3 * bisp_s['p0k1'][isort] - 1/nbar, c='C0', label='pySpec') 
isort = np.argsort(_i)
sub.plot(kf*_i[isort], (2*np.pi/kf)**3 * _pi[isort] - 1/nbar, c='k', ls='--', label="Roman's code") 
sub.legend(loc='lower right', fontsize=25)
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_ylim([3e3, 2e5]) 
sub.set_yscale('log') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
fig.savefig(os.path.join(UT.dat_dir(), 'test_bk.p0k.zspace.png'), bbox_inches='tight')

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
b_sn = (bisp['p0k1'] + bisp['p0k2'] + bisp['p0k3'])/nbar + 1/nbar**2 
sub.scatter(range(len(bisp['i_k1'])), (2*np.pi/kf)**6 * bisp['b123'] - b_sn, c='C1', s=1, label='real-space') 
b_sn = (bisp_s['p0k1'] + bisp_s['p0k2'] + bisp_s['p0k3'])/nbar + 1/nbar**2 
sub.plot(range(len(bisp_s['i_k1'])), (2*np.pi/kf)**6 * bisp_s['b123'], c='C0', lw=0.5, ls=':') # shot noise uncorrected
sub.scatter(range(len(bisp_s['i_k1'])), (2*np.pi/kf)**6 * bisp_s['b123'] - b_sn, c='C0', s=5, label='pySpec') #shot noise corrected
b_sn = (_pi + _pj + _pl)/nbar + 1/nbar**2 
sub.scatter(range(len(_i)), (2*np.pi/kf)**6 * _b123 - b_sn, c='k', s=1, label="Roman's code") 
sub.set_xlabel('triangles', fontsize=25)
sub.set_xlim([0, len(_i)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$ (not SN corrected)', fontsize=25)
sub.set_yscale('log') 
sub.set_ylim([1e7, 8e9])
fig.savefig(''.join([UT.dat_dir(), 'test_bk.bk.zspace.png']), bbox_inches='tight')

fig = plt.figure(figsize=(15,5))
sub = fig.add_subplot(111)
sub.scatter(range(len(bisp_s['i_k1'])), bisp_s['b123']/_b123, c='C0', s=2, label="pySpec/roman's code") 
sub.set_xlabel('triangles', fontsize=25)
sub.set_xlim([0, len(_i)])
sub.set_ylabel(r'$B(k_1, k_2, k_3)$ ratio (not SN corrected)', fontsize=25)
sub.set_ylim([0.95, 1.05])
fig.savefig(''.join([UT.dat_dir(), 'test_bk.bk.ratio.zspace.png']), bbox_inches='tight')
