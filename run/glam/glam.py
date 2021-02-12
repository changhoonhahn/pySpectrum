#!/bin/python 
'''

calculate the powerspectrum and bipsectrum for GLAM halo catalogs  

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


ireal   = int(sys.argv[1]) 
logmlim = float(sys.argv[2]) # log10(Mhalo) limit 
rsd     = sys.argv[3] == 'True' 

Lbox = 1000. 
kf = 2.*np.pi/Lbox

str_rsd = ['', '.rsd'][rsd]
f_halo = os.path.join(dir_dat, 'CatshortV.0136.%s.DAT' % str(ireal).zfill(4))
f_hdf5 = f_halo.replace(".DAT", '.mlim1e%i.hdf5' % int(logmlim)) 
f_pell = os.path.join(dir_dat, 'pySpec.Plk.glam.CatshortV.0136.%s.halo.mlim1e%i.Lbox%.f.Ngrid360%s.dat' % (str(ireal).zfill(4), int(logmlim), Lbox, str_rsd)) 
f_b123 = os.path.join(dir_dat, 'pySpec.B123.glam.CatshortV.0136.%s.halo.mlim1e%i.Lbox%.f.Ngrid360.Nmax40.Ncut3.step3.pyfftw%s.dat' % (str(ireal).zfill(4), int(logmlim), Lbox, str_rsd)) 

# read halo catalog 
if not os.path.isfile(f_hdf5):  
    x, y, z, vx, vy, vz, mh = np.loadtxt(f_halo, unpack=True, skiprows=8, usecols=[0, 1, 2, 3, 4, 5, 7]) 

    xyz = np.zeros((len(x),3)) 
    xyz[:,0] = x
    xyz[:,1] = y 
    xyz[:,2] = z
        
    vxyz = np.zeros((len(x),3))
    vxyz[:,0] = vx
    vxyz[:,1] = vy 
    vxyz[:,2] = vz

    # RSD along the z axis 
    _xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.0, h=0.7, omega0_m=0.340563, LOS='z', Lbox=Lbox) 
    xyz_s = _xyz_s.T

    # impose halo mass limit 
    cut = (np.log10(mh) > logmlim)

    # save to hdf5 for easy access
    f = h5py.File(f_hdf5, 'w') 
    f.create_dataset('xyz', data=xyz[cut,:]) 
    f.create_dataset('vxyz', data=vxyz[cut,:]) 
    f.create_dataset('xyz_s', data=xyz_s[cut,:]) 
    f.create_dataset('mhalo', data=mh[cut]) 
    f.close() 

else: 
    f = h5py.File(f_hdf5, 'r') 
    xyz     = f['xyz'].value
    xyz_s   = f['xyz_s'].value 
    vxyz    = f['vxyz'].value
    mh      = f['mhalo'].value

Nhalo = xyz.shape[0]
print('# halos = %i in %.1f box' % (Nhalo, Lbox)) 
nhalo = float(Nhalo) / Lbox**3
print('number density = %f' % nhalo) 
print('1/nbar = %f' % (1./nhalo))

# calculate powerspectrum 
if not os.path.isfile(f_pell): 
    # calculate powerspectrum monopole  
    if not rsd: 
        spec = pySpec.Pk_periodic(xyz.T, Lbox=Lbox, Ngrid=360, silent=False) 
    else: 
        spec = pySpec.Pk_periodic(xyz_s.T, Lbox=Lbox, Ngrid=360, silent=False) 
    k       = spec['k'] 
    p0k     = spec['p0k']
    cnts    = spec['counts']
    # save to file 
    hdr = ('pyspectrum P_l=0(k) calculation. Lbox=%.1f, k_f=%.5e, SN=%.5e' % (Lbox, kf, 1./nhalo))
    np.savetxt(f_pell, np.array([k, p0k, cnts]).T, fmt='%.5e %.5e %.5e', delimiter='\t', header=hdr) 
else: 
    k, p0k, cnts = np.loadtxt(f_pell, skiprows=1, unpack=True, usecols=[0,1,2]) 

# calculate bispectrum 
if not os.path.isfile(f_b123): 
    # calculate bispectrum 
    if rsd: 
        bispec = pySpec.Bk_periodic(xyz_s.T, Lbox=Lbox, Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 
    else: 
        bispec = pySpec.Bk_periodic(xyz.T, Lbox=Lbox, Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 

    i_k = bispec['i_k1']
    j_k = bispec['i_k2']
    l_k = bispec['i_k3']
    p0k1 = bispec['p0k1'] 
    p0k2 = bispec['p0k2'] 
    p0k3 = bispec['p0k3'] 
    b123 = bispec['b123'] 
    b123_sn = bispec['b123_sn'] 
    q123 = bispec['q123'] 
    counts = bispec['counts']
    # save to file 
    hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/%.1f' % Lbox
    np.savetxt(f_b123, 
            np.array([i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn]).T, 
            fmt='%i %i %i %.5e %.5e %.5e %.5e %.5e %.5e %.5e', 
            delimiter='\t', header=hdr) 
else: 
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123, 
            skiprows=1, unpack=True, usecols=range(10)) 
