#!/bin/python 
'''

calculate the powerspectrum and bipsectrum for QPM halo box 

'''
import os
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
f_hdf5 = ''.join([UT.dat_dir(), 'qpm/halo_ascii.hdf5'])
f_fftw = ''.join([UT.dat_dir(), 'qpm/pySpec.fft.halo', 
    '.Ngrid360', 
    '.dat']) 
f_pell = ''.join([UT.dat_dir(), 'qpm/pySpec.Plk.halo', 
    '.Ngrid360', 
    '.dat']) 
f_pnkt = ''.join([UT.dat_dir(), 'qpm/pySpec.Plk.halo', 
    '.Ngrid360', 
    '.nbodykit', 
    '.dat']) 
f_b123 = ''.join([UT.dat_dir(), 'qpm/pySpec.B123.halo', 
    '.Ngrid360', 
    '.Nmax40', 
    '.Ncut3', 
    '.step3', 
    '.pyfftw', 
    '.dat']) 

Lbox = 1024. 
kf = 2.*np.pi/Lbox

if not os.path.isfile(f_hdf5):  
    mh, x, y, z, vx, vy, vz = np.loadtxt(f_halo, unpack=True, skiprows=1, usecols=[0,1,2,3,4,5,6]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    vxyz = np.zeros((len(x),3))
    vxyz[:,0] = vx
    vxyz[:,1] = vy 
    vxyz[:,2] = vz

    f = h5py.File(f_hdf5, 'w') 
    f.create_dataset('xyz', data=xyz) 
    f.create_dataset('vxyz', data=vxyz) 
    f.create_dataset('mhalo', data=mh) 
    f.close() 

# calculate the FFT 
if not os.path.isfile(f_fftw):  
    # read in QPM halos 
    f = h5py.File(f_hdf5, 'r') 
    xyz = f['xyz'].value 
    # calculate FFTs
    delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=Lbox, Ngrid=360, silent=False) 
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
        delta_fft = np.reshape(delta_fft, (360, 360, 360), order='F')

    # calculate powerspectrum monopole  
    k, p0k, cnts = pySpec.Pk_periodic(delta_fft) 
    
    # save to file 
    hdr = 'pyspectrum P_l=0(k) calculation. k_f = 2pi/1024'
    np.savetxt(f_pell, np.array([k*kf, p0k/(kf**3), cnts]).T, fmt='%.5e %.5e %.5e', delimiter='\t', header=hdr) 

# calculate P(k) using nbodykit for santiy check 
if not os.path.isfile(f_pnkt): 
    # read in QPM halos 
    f = h5py.File(f_hdf5, 'r') 
    xyz = f['xyz'].value
    vxyz = f['vxyz'].value
    mh = f['mhalo'].value 

    # get cosmology from header 
    Omega_m = 0.3175
    Omega_b = 0.049 # fixed baryon 
    h = 0.6711
    cosmo = NBlab.cosmology.Planck15.clone(Omega_cdm=Omega_m-Omega_b, h=h, Omega_b=Omega_b)

    halo_data = {}  
    halo_data['Position']  = xyz.T 
    halo_data['Velocity']  = vxyz
    halo_data['Mass']      = mh
    print("putting it into array catalog") 
    halos = NBlab.ArrayCatalog(halo_data, BoxSize=np.array([Lbox, Lbox, Lbox])) 
    print("putting it into halo catalog") 
    halos = NBlab.HaloCatalog(halos, cosmo=cosmo, redshift=0., mdef='vir') 
    print("putting it into mesh") 
    mesh = halos.to_mesh(window='tsc', Nmesh=360, compensated=True, position='Position')
    print("calculating powerspectrum" ) 
    r = NBlab.FFTPower(mesh, mode='2d', dk=kf, kmin=kf, Nmu=5, los=[0,0,1], poles=[0,2,4])
    poles = r.poles
    plk = {'k': poles['k']} 
    for ell in [0, 2, 4]:
        P = (poles['power_%d' % ell].real)
        if ell == 0: 
            P = P - poles.attrs['shotnoise'] # subtract shotnoise from monopole 
        plk['p%dk' % ell] = P 
    plk['shotnoise'] = poles.attrs['shotnoise'] # save shot noise term

    # header 
    hdr = 'pyspectrum P_l(k) calculation. k_f = 2pi/1024; P_shotnoise '+str(plk['shotnoise']) 
    # write to file 
    np.savetxt(f_pnkt, np.array([plk['k'], plk['p0k'], plk['p2k'], plk['p4k']]).T, header=hdr) 

# calculate bispectrum 
if not os.path.isfile(f_b123): 
    try:
        delta_fft
    except NameError: 
        f = FortranFile(f_fftw, 'r')
        delta_fft = f.read_reals(dtype=np.complex64) 
        delta_fft = np.reshape(delta_fft, (360, 360, 360), order='F')

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
sub.plot(k, p0k, c='k', lw=1) 
#sub.plot(plk['k'], plk['p0k'], c='C1', lw=1) 
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
