#!/bin/python 
'''

calculate the powerspectrum and bipsectrum for QPM halo box 

'''
import os
import numpy as np 
from scipy.io import FortranFile
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import plots as Plots 
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


f_halo = ''.join([UT.dat_dir(), 'qpm/halo_ascii.dat'])
f_fftw = ''.join([UT.dat_dir(), 'qpm/pySpec.fft.halo', 
    '.Ngrid360', 
    '.dat']) 
f_pell = ''.join([UT.dat_dir(), 'qpm/pySpec.Plk.halo', 
    '.Ngrid360', 
    '.dat']) 
f_b123 = ''.join([UT.dat_dir(), 'qpm/pySpec.B123.halo', 
    '.Ngrid360', 
    '.Nmax40', 
    '.Ncut3', 
    '.step3', 
    '.pyfftw', 
    '.dat']) 

kf = 2.*np.pi/1024.

# calculate the FFT 
if not os.path.isfile(f_fftw):  
    # read in QPM halos 
    x, y, z = np.loadtxt(f_halo, unpack=True, usecols=[1,2,3]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    print('%f < x < %f' % (x.min(), x.max()))
    print('%f < y < %f' % (y.min(), y.max()))
    print('%f < z < %f' % (z.min(), z.max()))
    # calculate FFTs
    delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=1024, Ngrid=360, silent=False) 
    delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 

    f = FortranFile(f_fftw, 'w') 
    f.write_record(delta_fft) # double prec 
    f.close() 

# calculate powerspectrum 
if not os.path.isfile(f_pell): 
    try:
        delta_fft
    except NameError: 
        f = FortranFile(f_fftw, 'r')
        delta_fft = f.read_reals(dtype=np.complex64) 
        delta_fft = np.reshape(delta_fft, (Ngrid, Ngrid, Ngrid), order='F')

    # calculate powerspectrum monopole  
    k, p0k = pySpec.Pk_periodic(delta_fft) 
    
    # save to file 
    hdr = 'pyspectrum P_l=0(k) calculation. k_f = 2pi/1024'
    np.savetxt(f_pell, np.array([k*kf, p0k]).T, fmt='%.5e %.5e', delimiter='\t', header=hdr) 

# calculate bispectrum 
if not os.path.isfile(f_b123): 
    try:
        delta_fft
    except NameError: 
        f = FortranFile(f_fftw, 'r')
        delta_fft = f.read_reals(dtype=np.complex64) 
        delta_fft = np.reshape(delta_fft, (Ngrid, Ngrid, Ngrid), order='F')

    # calculate bispectrum 
    i_k, j_k, l_k, b123, q123, counts = pySpec.Bk123_periodic(
            delta_fft, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', nthreads=2, silent=False) 
    # save to file 
    hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/3600'
    np.savetxt(f_b123, np.array([i_k, j_k, l_k, b123, q123, counts]).T, fmt='%i %i %i %.5e %.5e %.5e', 
            delimiter='\t', header=hdr) 

# plot powerspecrtrum shape triangle plot 
k, p0k = np.loadtxt(f_pell, unpack=True, skiprows=1, usecols=[0,1]) 
fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
sub.plot(k, p0k/kf**3, c='k', lw=2) 
sub.set_ylabel('$P_0(k)$', fontsize=25) 
sub.set_yscale('log') 
sub.set_xlabel('$k$', fontsize=25) 
sub.set_xlim([3e-3, 1.]) 
sub.set_xscale('log') 
fig.savefig(''.join([UT.dat_dir(), 'qpm/p0k.png']), bbox_inches='tight')

# plot bispectrum shape triangle plot 
i_k, j_k, l_k, b123, q123, counts = np.loadtxt(f_b123, unpack=True, skiprows=1, usecols=[0,1,2,3,4,5]) 

nbin = 50
x_bins = np.linspace(0., 1., nbin+1)
y_bins = np.linspace(0.5, 1., (nbin//2) + 1) 

fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
Qgrid = Plots._BorQgrid(l_k/i_k, j_k/i_k, q123, counts, x_bins, y_bins)
bplot = plt.pcolormesh(x_bins, y_bins, Qgrid.T, vmin=0.0, vmax=1.0, cmap='RdBu')
cbar = plt.colorbar(bplot, orientation='vertical')

sub.set_title(r'$Q(k_1, k_2, k_3)$ QPM halo catalog', fontsize=25)
sub.set_xlabel('$k_3/k_1$', fontsize=25)
sub.set_ylabel('$k_2/k_1$', fontsize=25)
fig.savefig(''.join([UT.dat_dir(), 'qpm/Q123_shape.png']), bbox_inches='tight')

# plot bispectrum amplitude 
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
sub.scatter(range(len(q123)), q123, c='k', s=1)
sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
sub.set_xlim([0, len(q123)]) 
sub.set_ylabel(r'$Q(k_1, k_2, k_3)$', fontsize=25) 
sub.set_ylim([0., 1.]) 
fig.savefig(''.join([UT.dat_dir(), 'qpm/Q123.png']), bbox_inches='tight')
