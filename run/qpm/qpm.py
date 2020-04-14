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


dir_dat = '/home/chhahn/data/pyspectrum/qpm/'


def QPMspectra(rsd=False):     
    ''' calculate the powerspectrum and bispectrum of the QPM 
    catalog.

    :param rsd: (default: False)
        if True calculate in redshift space. Otherwise, real-space 
    '''
    str_rsd = ''
    if rsd: str_rsd = '.rsd'
    f_halo = ''.join([dir_dat, 'halo_ascii.dat'])
    f_hdf5 = ''.join([dir_dat, 'halo.mlim1e13.Lbox1050.hdf5'])
    f_pell = ''.join([dir_dat, 'pySpec.Plk.halo.mlim1e13.Lbox1050', 
        '.Ngrid360', str_rsd, '.dat']) 
    f_pnkt = ''.join([dir_dat, 'pySpec.Plk.halo.mlim1e13.Lbox1050', 
        '.Ngrid360', '.nbodykit', str_rsd, '.dat']) 
    f_b123 = ''.join([dir_dat, 'pySpec.B123.halo.mlim1e13.Lbox1050', 
        '.Ngrid360', '.Nmax40', '.Ncut3', '.step3', '.pyfftw', str_rsd, '.dat']) 

    Lbox = 1050. 
    kf = 2.*np.pi/Lbox
    
    # 1. read in ascii file
    # 2. impose 10^13 halo mass limit 
    # 3. calculate RSD positions 
    # 4. write to hdf5 file
    if not os.path.isfile(f_hdf5):  
        mh, x, y, z, vx, vy, vz = np.loadtxt(f_halo, unpack=True, skiprows=1, usecols=[0,1,2,3,4,5,6]) 
        xyz = np.zeros((len(x),3)) 
        xyz[:,0] = x
        xyz[:,1] = y 
        xyz[:,2] = z

        vxyz = np.zeros((len(x),3))
        vxyz[:,0] = vx
        vxyz[:,1] = vy 
        vxyz[:,2] = vz
        
        # RSD along the z axis 
        xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.55, h=0.7, omega0_m=0.340563, LOS='z', Lbox=Lbox) 

        mlim = (mh > 1e13) 

        mh = mh[mlim] 
        xyz = xyz[mlim,:] 
        vxyz = vxyz[mlim,:]
        xyz_s = xyz_s.T[mlim,:]

        f = h5py.File(f_hdf5, 'w') 
        f.create_dataset('xyz', data=xyz) 
        f.create_dataset('vxyz', data=vxyz) 
        f.create_dataset('xyz_s', data=xyz_s) 
        f.create_dataset('mhalo', data=mh) 
        f.close() 
    else: 
        f = h5py.File(f_hdf5, 'r') 
        xyz = f['xyz'].value
        xyz_s = f['xyz_s'].value 
        vxyz = f['vxyz'].value
        mh = f['mhalo'].value

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

    # calculate P(k) using nbodykit for santiy check 
    if not os.path.isfile(f_pnkt): 
        # get cosmology from header 
        Omega_m = 0.3175
        Omega_b = 0.049 # fixed baryon 
        h = 0.6711
        cosmo = NBlab.cosmology.Planck15.clone(Omega_cdm=Omega_m-Omega_b, h=h, Omega_b=Omega_b)

        halo_data = {}  
        if not rsd: halo_data['Position']  = xyz 
        else: halo_data['Position'] = xyz_s
        halo_data['Velocity']  = vxyz
        halo_data['Mass']      = mh
        print("putting it into array catalog") 
        halos = NBlab.ArrayCatalog(halo_data, BoxSize=np.array([Lbox, Lbox, Lbox])) 
        print("putting it into halo catalog") 
        halos = NBlab.HaloCatalog(halos, cosmo=cosmo, redshift=0., mdef='vir') 
        print("putting it into mesh") 
        mesh = halos.to_mesh(window='tsc', Nmesh=360, compensated=True, position='Position')
        print("calculating powerspectrum" ) 
        r = NBlab.FFTPower(mesh, mode='1d', dk=kf, kmin=kf, poles=[0,2,4])
        poles = r.poles
        plk = {'k': poles['k']} 
        for ell in [0, 2, 4]:
            P = (poles['power_%d' % ell].real)
            if ell == 0: 
                P = P - poles.attrs['shotnoise'] # subtract shotnoise from monopole 
            plk['p%dk' % ell] = P 
        plk['shotnoise'] = poles.attrs['shotnoise'] # save shot noise term

        # header 
        hdr = 'pyspectrum P_l(k) calculation. k_f = 2pi/%.1f; P_shotnoise %f' % (Lbox, plk['shotnoise']) 
        # write to file 
        np.savetxt(f_pnkt, np.array([plk['k'], plk['p0k'], plk['p2k'], plk['p4k']]).T, header=hdr) 
    else: 
        _k, _p0k, _p2k, _p4k = np.loadtxt(f_pnkt, skiprows=1, unpack=True, usecols=[0,1,2,3]) 
        plk = {} 
        plk['k'] = _k
        plk['p0k'] = _p0k
        plk['p2k'] = _p2k
        plk['p4k'] = _p4k

    # calculate bispectrum 
    if not os.path.isfile(f_b123): 
        # calculate bispectrum 
        if not rsd: 
            bispec = pySpec.Bk_periodic(xyz.T, Lbox=Lbox, Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 
        else: 
            bispec = pySpec.Bk_periodic(xyz_s.T, Lbox=Lbox, Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 

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

    # plot powerspecrtrum shape triangle plot 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.plot(k, p0k, c='k', lw=1, label='pySpectrum') 
    sub.plot(plk['k'], plk['p0k'], c='C1', lw=1, label='nbodykit') 
    sub.plot(i_k * kf, p0k1, c='k', lw=1, ls='--', label='bispectrum code') 
    sub.legend(loc='lower left', fontsize=20) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    #sub.set_ylim([1e2, 3e4]) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([3e-3, 1.]) 
    sub.set_xscale('log') 
    fig.savefig(''.join([dir_dat, 'qpm_p0k', str_rsd, '.png']), bbox_inches='tight')

    # plot bispectrum shape triangle plot 
    nbin = 31 
    x_bins = np.linspace(0., 1., nbin+1)
    y_bins = np.linspace(0.5, 1., (nbin//2) + 1) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), q123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, vmin=0, vmax=1, cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$Q(k_1, k_2, k_3)$ QPM halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(''.join([dir_dat, 'qpm_Q123_shape', str_rsd, '.png']), bbox_inches='tight')
    
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), b123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, norm=LogNorm(vmin=1e6, vmax=1e8), cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$B(k_1, k_2, k_3)$ QPM halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(''.join([dir_dat, 'qpm_B123_shape', str_rsd, '.png']), bbox_inches='tight')

    # plot bispectrum amplitude 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), q123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_ylim([0., 1.]) 
    fig.savefig(''.join([dir_dat, 'qpm_Q123', str_rsd, '.png']), bbox_inches='tight')

    # plot bispectrum amplitude 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), b123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    fig.savefig(''.join([dir_dat,'qpm_B123', str_rsd, '.png']), bbox_inches='tight')
    return None


def AEMspectra(rsd=False): 
    ''' calculate the powerspectrum and bispectrum of the Aemulus simulation box 
    '''
    str_rsd = ''
    if rsd: str_rsd = '.rsd'
    f_halo = ''.join([UT.dat_dir(), 'aemulus/aemulus_test002_halos.mlim1e13.hdf5'])
    f_hdf5 = ''.join([UT.dat_dir(), 'aemulus/aemulus_test002_halos.mlim1e13.hdf5'])
    f_pell = ''.join([UT.dat_dir(), 'aemulus/pySpec.Plk.halo.mlim1e13.Ngrid360', str_rsd, '.dat']) 
    f_pnkt = ''.join([UT.dat_dir(), 'aemulus/pySpec.Plk.halo.mlim1e13.Ngrid360.nbodykit', str_rsd, '.dat']) 
    f_b123 = ''.join([UT.dat_dir(), 'aemulus/pySpec.B123.halo.mlim1e13.Ngrid360.Nmax40.Ncut3.step3.pyfftw', str_rsd, '.dat']) 

    Lbox=1050.
    kf = 2.*np.pi/Lbox

    if not os.path.isfile(f_hdf5):  
        f = h5py.File(f_halo, 'r') 
        xyz     = f['xyz'].value
        vxyz    = f['vxyz'].value
        mh      = f['mhalo'].value
        xyz_s   = pySpec.applyRSD(xyz.T, vxyz.T, 0.55, h=0.7, omega0_m=0.340563, LOS='z', Lbox=Lbox) 
        xyz_s   = xyz_s.T

        f = h5py.File(f_hdf5, 'w') 
        f.create_dataset('xyz', data=xyz) 
        f.create_dataset('vxyz', data=vxyz) 
        f.create_dataset('xyz_s', data=xyz_s) 
        f.create_dataset('mhalo', data=mh) 
        f.close() 
    else: 
        f = h5py.File(f_hdf5, 'r') 
        xyz     = f['xyz'].value
        vxyz    = f['vxyz'].value
        xyz_s   = f['xyz_s'].value
        mh      = f['mhalo'].value 
        f.close() 
    
    Nhalo = xyz.shape[0]
    print('# halos = %i' % Nhalo) 
    nhalo = float(Nhalo) / Lbox**3
    print('number density = %f' % nhalo) 
    print('1/nbar = %f' % (1./nhalo))

    # calculate powerspectrum 
    if not os.path.isfile(f_pell): 
        # calculate FFTs
        if not rsd: 
            delta = pySpec.FFTperiodic(xyz.T, fft='fortran', Lbox=Lbox, Ngrid=360, silent=False) 
        else: 
            delta = pySpec.FFTperiodic(xyz_s.T, fft='fortran', Lbox=Lbox, Ngrid=360, silent=False) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 

        # calculate powerspectrum monopole  
        k, p0k, cnts = pySpec.Pk_periodic(delta_fft) 
        k = k * kf 
        p0k = p0k/kf**3 - 1./nhalo
        
        # save to file 
        hdr = 'pyspectrum P_l=0(k) calculation. k_f = 2pi/1050.'
        np.savetxt(f_pell, np.array([k, p0k, cnts]).T, fmt='%.5e %.5e %.5e', delimiter='\t', header=hdr) 
    else: 
        k, p0k, cnts = np.loadtxt(f_pell, skiprows=1, unpack=True, usecols=[0,1,2]) 
    
    # calculate P(k) using nbodykit for santiy check 
    if not os.path.isfile(f_pnkt): 
        # get cosmology from header 
        Omega_m = 0.3175
        Omega_b = 0.049 # fixed baryon 
        h = 0.6711
        cosmo = NBlab.cosmology.Planck15.clone(Omega_cdm=Omega_m-Omega_b, h=h, Omega_b=Omega_b)

        halo_data = {}  
        if not rsd: halo_data['Position']  = xyz 
        else: halo_data['Position'] = xyz_s
        halo_data['Velocity']  = vxyz
        halo_data['Mass']      = mh
        print("putting it into array catalog") 
        halos = NBlab.ArrayCatalog(halo_data, BoxSize=np.array([Lbox, Lbox, Lbox])) 
        print("putting it into halo catalog") 
        halos = NBlab.HaloCatalog(halos, cosmo=cosmo, redshift=0., mdef='vir') 
        print("putting it into mesh") 
        mesh = halos.to_mesh(window='tsc', Nmesh=360, compensated=True, position='Position')
        print("calculating powerspectrum" ) 
        r = NBlab.FFTPower(mesh, mode='1d', dk=kf, kmin=kf, poles=[0,2,4])
        poles = r.poles
        plk = {'k': poles['k']} 
        for ell in [0, 2, 4]:
            P = (poles['power_%d' % ell].real)
            if ell == 0: 
                P = P - poles.attrs['shotnoise'] # subtract shotnoise from monopole 
            plk['p%dk' % ell] = P 
        plk['shotnoise'] = poles.attrs['shotnoise'] # save shot noise term

        # header 
        hdr = 'pyspectrum P_l(k) calculation. k_f = 2pi/1050; P_shotnoise '+str(plk['shotnoise']) 
        # write to file 
        np.savetxt(f_pnkt, np.array([plk['k'], plk['p0k'], plk['p2k'], plk['p4k']]).T, header=hdr) 
    else: 
        _k, _p0k, _p2k, _p4k = np.loadtxt(f_pnkt, skiprows=1, unpack=True, usecols=[0,1,2,3]) 
        plk = {} 
        plk['k'] = _k
        plk['p0k'] = _p0k
        plk['p2k'] = _p2k
        plk['p4k'] = _p4k
    
    # calculate bispectrum 
    if not os.path.isfile(f_b123): 
        if not rsd: 
            bispec = pySpec.Bk_periodic(xyz.T, Lbox=Lbox, Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 
        else: 
            bispec = pySpec.Bk_periodic(xyz_s.T, Lbox=Lbox, Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 

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

    # plot powerspecrtrum shape triangle plot 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.plot(k, p0k, c='k', lw=1, label='pySpectrum') 
    sub.plot(plk['k'], plk['p0k'], c='C1', lw=1, label='nbodykit') 
    iksort = np.argsort(i_k) 
    sub.plot(i_k[iksort] * kf, p0k1[iksort], c='k', lw=1, ls='--', label='bispectrum code') 
    sub.legend(loc='lower left', fontsize=20) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    #sub.set_ylim([1e2, 3e4]) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([3e-3, 1.]) 
    sub.set_xscale('log') 
    fig.savefig(''.join([UT.dat_dir(), 'aemulus/aemulus_p0k', str_rsd, '.png']), bbox_inches='tight')

    # plot bispectrum shape triangle plot 
    nbin = 31 
    x_bins = np.linspace(0., 1., nbin+1)
    y_bins = np.linspace(0.5, 1., (nbin//2) + 1) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), q123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, vmin=0, vmax=1, cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$Q(k_1, k_2, k_3)$ QPM halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(''.join([UT.dat_dir(), 'aemulus/aemulus_Q123_shape', str_rsd, '.png']), bbox_inches='tight')
    
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), b123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, norm=LogNorm(vmin=1e6, vmax=1e8), cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$B(k_1, k_2, k_3)$ QPM halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(''.join([UT.dat_dir(), 'aemulus/aemulus_B123_shape', str_rsd, '.png']), bbox_inches='tight')

    # plot bispectrum amplitude 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), q123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_ylim([0., 1.]) 
    fig.savefig(''.join([UT.dat_dir(), 'aemulus/aemulus_Q123', str_rsd, '.png']), bbox_inches='tight')

    # plot bispectrum amplitude 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), b123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    fig.savefig(''.join([UT.dat_dir(), 'aemulus/aemulus_B123', str_rsd, '.png']), bbox_inches='tight')
    return None


def QPM_AEM(rsd=False): 
    str_rsd = ''
    if rsd: str_rsd = '.rsd'
    f_pell = lambda sim: ''.join([UT.dat_dir(), sim, '/pySpec.Plk.halo.mlim1e13.Ngrid360', str_rsd, '.dat']) 
    f_pnkt = lambda sim: ''.join([UT.dat_dir(), sim, '/pySpec.Plk.halo.mlim1e13.Ngrid360.nbodykit', str_rsd, '.dat']) 
    f_b123 = lambda sim: ''.join([UT.dat_dir(), sim, '/pySpec.B123.halo.mlim1e13.Ngrid360.Nmax40.Ncut3.step3.pyfftw', str_rsd, '.dat']) 
    
    Lbox_qpm = 1050. 
    Lbox_aem = 1050. 
    kf_qpm = 2.*np.pi/Lbox_qpm
    kf_aem = 2.*np.pi/Lbox_aem
    
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    k, p0k = np.loadtxt(f_pell('aemulus'), unpack=True, skiprows=1, usecols=[0,1]) 
    klim = (k < 0.3) 
    sub.plot(k[klim], p0k[klim], c='k', lw=1, label='Aemulus') 
    k, p0k = np.loadtxt(f_pnkt('aemulus'), unpack=True, skiprows=1, usecols=[0,1]) 
    klim = (k < 0.3) 
    sub.plot(k[klim], p0k[klim], c='k', ls='--', lw=1) 
    k, p0k = np.loadtxt(f_pell('qpm'), unpack=True, skiprows=1, usecols=[0,1]) 
    klim = (k < 0.3) 
    sub.plot(k[klim], p0k[klim], c='C1', lw=1, label='QPM') 
    k, p0k = np.loadtxt(f_pnkt('qpm'), unpack=True, skiprows=1, usecols=[0,1]) 
    klim = (k < 0.3) 
    sub.plot(k[klim], p0k[klim], c='C1', ls='--', lw=1) 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([8e-3, 0.5]) 
    sub.set_xscale('log') 
    fig.savefig(''.join([UT.dat_dir(), 'qpm/p0k_qpm_aemulus', str_rsd, '.png']), bbox_inches='tight')

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123('aemulus'),
            skiprows=1, unpack=True, usecols=range(10)) 
    klim = ((i_k*kf_aem <= 0.22) & (i_k*kf_aem >= 0.03) &
            (j_k*kf_aem <= 0.22) & (j_k*kf_aem >= 0.03) & 
            (l_k*kf_aem <= 0.22) & (l_k*kf_aem >= 0.03)) 
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]
        
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
   
    sub.scatter(range(np.sum(klim)), b123[klim][ijl], c='k', s=5, label='Aemulus') 
    sub.plot(range(np.sum(klim)), b123[klim][ijl], c='k') 

    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123('qpm'),
            skiprows=1, unpack=True, usecols=range(10)) 
    klim = ((i_k*kf_qpm <= 0.22) & (i_k*kf_qpm >= 0.03) &
            (j_k*kf_qpm <= 0.22) & (j_k*kf_qpm >= 0.03) & 
            (l_k*kf_qpm <= 0.22) & (l_k*kf_qpm >= 0.03)) 
    i_k, j_k, l_k, = i_k[klim], j_k[klim], l_k[klim]
    
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    sub.scatter(range(np.sum(klim)), b123[klim][ijl], c='C1', s=5, label='QPM') 
    sub.plot(range(np.sum(klim)), b123[klim][ijl], c='C1') 
    sub.legend(loc='upper right', markerscale=4, handletextpad=0., fontsize=20) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim([1e8, 6e9]) 
    sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    fig.savefig(''.join([UT.dat_dir(), 'qpm/B123_qpm_aemulus', str_rsd, '.png']), bbox_inches='tight')

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123('aemulus'),
            skiprows=1, unpack=True, usecols=range(10)) 
    sub.scatter(range(len(q123)), q123, c='k', s=2, label='Aemulus') 
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123('qpm'),
            skiprows=1, unpack=True, usecols=range(10)) 
    sub.scatter(range(len(q123)), q123, c='C1', s=2, label='QPM') 
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('$k_1 > k_2 > k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0, len(q123)]) 
    sub.set_ylim([0., 1.]) 
    fig.savefig(''.join([UT.dat_dir(), 'qpm/Q123_qpm_aemulus', str_rsd, '.png']), bbox_inches='tight')
    return None


def AEM_rzspace(): 
    ''' comparison between real and redshift space 
    '''
    f_b123 = lambda sim, rsd: os.path.join(UT.dat_dir(), sim, 'pySpec.B123.halo.mlim1e13.Ngrid360.Nmax40.Ncut3.step3.pyfftw%s.dat' % ['', '.rsd'][rsd]) 
    Lbox_aem = 1050. 
    kf_aem = 2.*np.pi/Lbox_aem

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, _ = np.loadtxt(f_b123('aemulus', False),
            skiprows=1, unpack=True, usecols=range(10)) 
    i_k, j_k, l_k, p0k1, p0k2, p0k3, b123_s, q123_s, counts, _ = np.loadtxt(f_b123('aemulus', True),
            skiprows=1, unpack=True, usecols=range(10)) 
    klim = ((i_k*kf_aem <= 0.3) & (j_k*kf_aem <= 0.3) & (l_k*kf_aem <= 0.3)) 
    ijl = UT.ijl_order(i_k[klim], j_k[klim], l_k[klim], typ='GM') # order of triangles 
   
    sub.scatter(range(np.sum(klim)), b123[klim][ijl], c='k', s=5, label='real-space') 
    sub.plot(range(np.sum(klim)), b123[klim][ijl], c='k') 
    sub.scatter(range(np.sum(klim)), b123_s[klim][ijl], c='C1', s=5, label='z-space') 
    sub.plot(range(np.sum(klim)), b123_s[klim][ijl], c='C1') 
    print (b123_s[klim][ijl] / b123[klim][ijl]).flatten()
    sub.legend(loc='upper right', markerscale=4, handletextpad=0., fontsize=20) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    fig.savefig(''.join([UT.dat_dir(), 'qpm/B123_aemulus_rzspace.png']), bbox_inches='tight')

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(np.sum(klim)), q123[klim][ijl], c='k', s=2, label='real-space') 
    sub.plot(range(np.sum(klim)), q123[klim][ijl], c='k', lw=1) 
    sub.scatter(range(np.sum(klim)), q123_s[klim][ijl], c='C1', s=2, label='redshift-space') 
    sub.plot(range(np.sum(klim)), q123_s[klim][ijl], c='C1', lw=1) 
    sub.legend(loc='lower right', handletextpad=0.2, markerscale=10, fontsize=20) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_xlabel('$k_1 > k_2 > k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)]) 
    sub.set_ylim([0., 1.]) 
    fig.savefig(''.join([UT.dat_dir(), 'qpm/Q123_aemulus_rzspace.png']), bbox_inches='tight')
    return None



if __name__=="__main__": 
    #QPMspectra(rsd=False)
    QPMspectra(rsd=True)
    #QPMspectra(rsd=False)
    #AEMspectra(rsd=True)
    #AEMspectra(rsd=False)
    #QPM_AEM(rsd=False)
    #QPM_AEM(rsd=True)
    #AEM_rzspace()
