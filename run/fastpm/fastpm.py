#!/bin/python 
'''

calculate the powerspectrum and bipsectrum for QPM halo box 

'''
import os
import h5py 
import numpy as np 
# -- nbodykit -- 
import nbodykit.lab as NBlab
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import plots as Plots 
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


def fastPM(z, str_flag='', mh_lim=15., Lbox=205., Nmax=40, Ncut=3, step=3):     
    ''' calculate the powerspectrum and bispectrum of the fastPM catalog.
    '''
    dir_fpm = os.path.join(UT.dat_dir(), 'fastpm') 
    f_halo = ('halocat_FastPM_40step_N250_IC500_B2_z%.2f%s.txt' % (z, str_flag))
    f_mlim = ('halocat_FastPM_40step_N250_IC500_B2_z%.2f%s.mlim%.fe10' % (z, str_flag, mh_lim))
    f_hdf5 = ('%s/%s.hdf5' % (dir_fpm, f_mlim)) 
    f_pell = ('%s/pySpec.Plk.%s.Lbox%.f.Ngrid360.dat' % (dir_fpm, f_mlim, Lbox))
    f_pnkt = ('%s/pySpec.Plk.%s.Lbox%.f.Ngrid360.nbodykit.dat' % (dir_fpm, f_mlim, Lbox))
    f_b123 = ('%s/pySpec.Bk.%s.Lbox%.f.Ngrid360.step%i.Ncut%i.Nmax%i.dat' % (dir_fpm, f_mlim, Lbox, step, Ncut, Nmax))

    kf = 2.*np.pi/Lbox
    
    if not os.path.isfile(f_hdf5): 
        # read in halo catalog
        dat_halo = np.loadtxt(os.path.join(dir_fpm, f_halo), unpack=True, usecols=[0,1,2,3,7,8,9]) 
        mh = dat_halo[0]
        Nhalo = len(mh) 
        print('%i halos in %.f Mpc/h box' % (len(mh), Lbox))
        print('%f < M_h/10^10Msun < %f' % (mh.min(), mh.max()))
        xyz = np.zeros((Nhalo,3)) 
        xyz[:,0] = dat_halo[1]
        xyz[:,1] = dat_halo[2]
        xyz[:,2] = dat_halo[3]
        print('%f < x < %f' % (xyz[:,0].min(), xyz[:,0].max()))
        print('%f < y < %f' % (xyz[:,1].min(), xyz[:,1].max()))
        print('%f < z < %f' % (xyz[:,2].min(), xyz[:,2].max()))

        vxyz = np.zeros((Nhalo,3))
        vxyz[:,0] = dat_halo[4]
        vxyz[:,1] = dat_halo[5] 
        vxyz[:,2] = dat_halo[6]

        mlim = (mh > 15.) 
        Nhalo = np.sum(mlim) 
        print('%i halos in %.f Mpc/h box with Mh > %f' % (Nhalo, Lbox, mh_lim))
        
        mh = mh[mlim] 
        xyz = xyz[mlim,:] 
        vxyz = vxyz[mlim,:]

        f = h5py.File(f_hdf5, 'w') 
        f.create_dataset('xyz', data=xyz) 
        f.create_dataset('vxyz', data=vxyz) 
        f.create_dataset('mhalo', data=mh) 
        f.close() 
    else: 
        f = h5py.File(f_hdf5, 'r') 
        xyz = f['xyz'].value
        vxyz = f['vxyz'].value
        mh = f['mhalo'].value
        Nhalo = xyz.shape[0]
        print('%i halos in %.f Mpc/h box with Mh > %f' % (len(mh), Lbox, mh_lim))

    nhalo = float(Nhalo) / Lbox**3
    print('number density = %f' % nhalo) 
    print('1/nbar = %f' % (1./nhalo))

    # calculate powerspectrum 
    if not os.path.isfile(f_pell): 
        delta = pySpec.FFTperiodic(xyz.T, fft='fortran', Lbox=Lbox, Ngrid=360, silent=False) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 

        # calculate powerspectrum monopole  
        k, p0k, cnts = pySpec.Pk_periodic(delta_fft) 
        k *= kf  
        p0k = p0k/(kf**3) - 1./nhalo 
        # save to file 
        hdr = ('pySpectrum P_l=0(k). Nhalo=%i, Lbox=%.f, k_f=%.5e, SN=%.5e' % (Nhalo, Lbox, kf, 1./nhalo))
        hdr += '\n k, p0k, counts'
        np.savetxt(f_pell, np.array([k, p0k, cnts]).T, fmt='%.5e %.5e %.5e', delimiter='\t', header=hdr) 
    else: 
        k, p0k, cnts = np.loadtxt(f_pell, skiprows=1, unpack=True, usecols=[0,1,2]) 

    # calculate P(k) using nbodykit for santiy check 
    if not os.path.isfile(f_pnkt): 
        cosmo = NBlab.cosmology.Planck15

        halo_data = {}  
        halo_data['Position']  = xyz 
        halo_data['Velocity']  = vxyz
        halo_data['Mass']      = mh
        print("putting it into array catalog") 
        halos = NBlab.ArrayCatalog(halo_data, BoxSize=np.array([Lbox, Lbox, Lbox])) 
        print("putting it into halo catalog") 
        halos = NBlab.HaloCatalog(halos, cosmo=cosmo, redshift=z, mdef='vir') 
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
        hdr = ('pySpectrum P_l(k). Nhalo=%i, Lbox=%.f, k_f=%.5e, SN=%.5e' % (Nhalo, Lbox, kf, plk['shotnoise']))
        hdr += '\n k, p0k, p2k, p4k'
        # save to file 
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
        hdr += '\n i_k1, i_k2, i_k3, p0k1, p0k2, p0k3, bk, qk, counts, bk_shotnoise'
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
    sub.set_ylim([1e0, 1e4]) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([1e-2, 10.]) 
    sub.set_xscale('log') 
    fig.savefig(f_pell.replace('.dat', '.png'), bbox_inches='tight')

    # plot bispectrum shape triangle plot 
    nbin = 31 
    x_bins = np.linspace(0., 1., nbin+1)
    y_bins = np.linspace(0.5, 1., (nbin//2) + 1) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), q123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, vmin=0, vmax=1, cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$Q(k_1, k_2, k_3)$ FastPM halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(f_b123.replace('.dat', '.Qk_shape.png'), bbox_inches='tight') 
    
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), b123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, norm=LogNorm(vmin=1e6, vmax=1e8), cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$B(k_1, k_2, k_3)$ FastPM halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(f_b123.replace('.dat', '.Bk_shape.png'), bbox_inches='tight') 

    # plot bispectrum amplitude 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), q123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_ylim([0., 1.]) 
    fig.savefig(f_b123.replace('.dat', '.Qk.png'), bbox_inches='tight') 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), b123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    fig.savefig(f_b123.replace('.dat', '.Bk.png'), bbox_inches='tight') 
    return None


def TNG(z, mh_lim=15., Lbox=205., Nmax=40, Ncut=3, step=3):     
    ''' calculate the powerspectrum and bispectrum of the fastPM catalog.
    '''
    dir_fpm = os.path.join(UT.dat_dir(), 'fastpm') 
    f_halo = ('halocat_TNG300Dark_z%.2f.txt' % z) 
    f_mlim = ('halocat_TNG300Dark_z%.2f.mlim%.fe10' % (z, mh_lim)) 
    f_hdf5 = ('%s/%s.hdf5' % (dir_fpm, f_mlim)) 
    f_pell = ('%s/pySpec.Plk.%s.Lbox%.f.Ngrid360.dat' % (dir_fpm, f_mlim, Lbox))
    f_pnkt = ('%s/pySpec.Plk.%s.Lbox%.f.Ngrid360.nbodykit.dat' % (dir_fpm, f_mlim, Lbox))
    f_b123 = ('%s/pySpec.Bk.%s.Lbox%.f.Ngrid360.step%i.Ncut%i.Nmax%i.dat' % (dir_fpm, f_mlim, Lbox, step, Ncut, Nmax))

    kf = 2.*np.pi/Lbox
    
    if not os.path.isfile(f_hdf5): 
        # read in halo catalog
        dat_halo = np.loadtxt(os.path.join(dir_fpm, f_halo), unpack=True, usecols=[0,1,2,3,7,8,9]) 
        mh = dat_halo[0]
        Nhalo = len(mh) 
        print('%i halos in %.f Mpc/h box' % (Nhalo, Lbox))
        print('%f < M_h/10^10Msun < %f' % (mh.min(), mh.max()))
        xyz = np.zeros((Nhalo,3)) 
        xyz[:,0] = dat_halo[1]
        xyz[:,1] = dat_halo[2]
        xyz[:,2] = dat_halo[3]

        vxyz = np.zeros((Nhalo,3))
        vxyz[:,0] = dat_halo[4]
        vxyz[:,1] = dat_halo[5] 
        vxyz[:,2] = dat_halo[6]

        mlim = (mh > 15.) 
        Nhalo = np.sum(mlim) 
        print('%i halos in %.f Mpc/h box with Mh > %f' % (Nhalo, Lbox, mh_lim))
        
        mh = mh[mlim] 
        xyz = xyz[mlim,:] 
        vxyz = vxyz[mlim,:]

        f = h5py.File(f_hdf5, 'w') 
        f.create_dataset('xyz', data=xyz) 
        f.create_dataset('vxyz', data=vxyz) 
        f.create_dataset('mhalo', data=mh) 
        f.close() 
    else: 
        f = h5py.File(f_hdf5, 'r') 
        xyz = f['xyz'].value
        vxyz = f['vxyz'].value
        mh = f['mhalo'].value
        Nhalo = xyz.shape[0]
        print('%i halos in %.f Mpc/h box with Mh > %f' % (len(mh), Lbox, mh_lim))

    nhalo = float(Nhalo) / Lbox**3
    print('number density = %f' % nhalo) 
    print('1/nbar = %f' % (1./nhalo))

    # calculate powerspectrum 
    if not os.path.isfile(f_pell): 
        delta = pySpec.FFTperiodic(xyz.T, fft='fortran', Lbox=Lbox, Ngrid=360, silent=False) 
        delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 

        # calculate powerspectrum monopole  
        k, p0k, cnts = pySpec.Pk_periodic(delta_fft) 
        k *= kf  
        p0k = p0k/(kf**3) - 1./nhalo 
        # save to file 
        hdr = ('pySpectrum P_l=0(k). Nhalo=%i, Lbox=%.f, k_f=%.5e, SN=%.5e' % (Nhalo, Lbox, kf, 1./nhalo))
        hdr += '\n k, p0k, counts'
        np.savetxt(f_pell, np.array([k, p0k, cnts]).T, fmt='%.5e %.5e %.5e', delimiter='\t', header=hdr) 
    else: 
        k, p0k, cnts = np.loadtxt(f_pell, skiprows=1, unpack=True, usecols=[0,1,2]) 

    # calculate P(k) using nbodykit for santiy check 
    if not os.path.isfile(f_pnkt): 
        cosmo = NBlab.cosmology.Planck15

        halo_data = {}  
        halo_data['Position']  = xyz 
        halo_data['Velocity']  = vxyz
        halo_data['Mass']      = mh
        print("putting it into array catalog") 
        halos = NBlab.ArrayCatalog(halo_data, BoxSize=np.array([Lbox, Lbox, Lbox])) 
        print("putting it into halo catalog") 
        halos = NBlab.HaloCatalog(halos, cosmo=cosmo, redshift=z, mdef='vir') 
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
        hdr = ('pySpectrum P_l(k). Nhalo=%i, Lbox=%.f, k_f=%.5e, SN=%.5e' % (Nhalo, Lbox, kf, plk['shotnoise']))
        hdr += '\n k, p0k, p2k, p4k'
        # save to file 
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
        hdr += '\n i_k1, i_k2, i_k3, p0k1, p0k2, p0k3, bk, qk, counts, bk_shotnoise'
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
    sub.set_xlim([1e-2, 10.]) 
    sub.set_xscale('log') 
    fig.savefig(f_pell.replace('.dat', '.png'), bbox_inches='tight')


    # plot bispectrum shape triangle plot 
    nbin = 31 
    x_bins = np.linspace(0., 1., nbin+1)
    y_bins = np.linspace(0.5, 1., (nbin//2) + 1) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), q123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, vmin=0, vmax=1, cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$Q(k_1, k_2, k_3)$ Illustris TNG halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(f_b123.replace('.dat', '.Qk_shape.png'), bbox_inches='tight') 
    
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    Bgrid = Plots._BorQgrid(l_k.astype(float)/i_k.astype(float), j_k.astype(float)/i_k.astype(float), b123, counts, x_bins, y_bins)
    bplot = plt.pcolormesh(x_bins, y_bins, Bgrid.T, norm=LogNorm(vmin=1e6, vmax=1e8), cmap='RdBu')
    cbar = plt.colorbar(bplot, orientation='vertical')
    sub.set_title(r'$B(k_1, k_2, k_3)$ Illustris TNG halo catalog', fontsize=25)
    sub.set_xlabel('$k_3/k_1$', fontsize=25)
    sub.set_ylabel('$k_2/k_1$', fontsize=25)
    fig.savefig(f_b123.replace('.dat', '.Bk_shape.png'), bbox_inches='tight') 

    # plot bispectrum amplitude 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), q123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_ylim([0., 1.]) 
    fig.savefig(f_b123.replace('.dat', '.Qk.png'), bbox_inches='tight') 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(b123)), b123, c='k', s=1)
    sub.set_xlabel(r'$k_1 > k_2 > k_3$ triangle index', fontsize=25) 
    sub.set_xlim([0, len(b123)]) 
    sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    fig.savefig(f_b123.replace('.dat', '.Bk.png'), bbox_inches='tight') 
    return None


def fastpm_v_tng(z, str_flag='', mh_lim=15., Lbox=205., Nmax=40, Ncut=3, step=3): 
    ''' make plots that compare the powerspectrum and bispectrum of 
    fastpm and illustris tng 
    '''
    dir_fpm = os.path.join(UT.dat_dir(), 'fastpm') 
    f_fpm = ('halocat_FastPM_40step_N250_IC500_B2_z%.2f%s.mlim%.fe10' % (z, str_flag, mh_lim))
    f_tng = ('halocat_TNG300Dark_z%.2f.mlim%.fe10' % (z, mh_lim)) 
    print('FastPM file: %s' % f_fpm) 
    print('Illustris TNG file: %s' % f_tng) 

    f_pell = lambda f_halo: ('%s/pySpec.Plk.%s.Lbox%.f.Ngrid360.dat' % (dir_fpm, f_halo, Lbox))
    f_pnkt = lambda f_halo: ('%s/pySpec.Plk.%s.Lbox%.f.Ngrid360.nbodykit.dat' % (dir_fpm, f_halo, Lbox))
    f_b123 = lambda f_halo: ('%s/pySpec.Bk.%s.Lbox%.f.Ngrid360.step%i.Ncut%i.Nmax%i.dat' % (dir_fpm, f_halo, Lbox, step, Ncut, Nmax))

    kf = 2.*np.pi/Lbox
    
    # P(k) comparison 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    k, p0k = np.loadtxt(f_pell(f_fpm), unpack=True, skiprows=1, usecols=[0,1]) 
    sub.plot(k, p0k, c='k', lw=1, label='FastPM') 
    k, p0k_fpm = np.loadtxt(f_pnkt(f_fpm), unpack=True, skiprows=1, usecols=[0,1]) 
    sub.plot(k, p0k_fpm, c='k', ls='--', lw=1) 
    k, p0k = np.loadtxt(f_pell(f_tng), unpack=True, skiprows=1, usecols=[0,1]) 
    sub.plot(k, p0k, c='C1', lw=1, label='Illustris TNG') 
    k, p0k_tng = np.loadtxt(f_pnkt(f_tng), unpack=True, skiprows=1, usecols=[0,1]) 
    sub.plot(k, p0k_tng, c='C1', ls='--', lw=1) 
    print (p0k_fpm / p0k_tng)[k < 0.2] - 1.
    sub.legend(loc='lower left', fontsize=20) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim([1e0, 1e4]) 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xscale('log') 
    sub.set_xlim([1e-2, 10.]) 
    fig.savefig(os.path.join(dir_fpm, 'p0k_fpm_tng_z%.2f%s.png' % (z, str_flag)), bbox_inches='tight')

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    for fh, lbl, c in zip([f_fpm, f_tng], ['FastPM', 'Illustris TNG'], ['k', 'C1']): 
        i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123(fh), 
                skiprows=1, unpack=True, usecols=range(10)) 
        klim = ((i_k * kf <= 1.) & (i_k * kf >= 0.01) &
                (j_k * kf <= 1.) & (j_k * kf >= 0.01) & 
                (l_k * kf <= 1.) & (l_k * kf >= 0.01)) 

        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]
        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
   
        sub.scatter(range(np.sum(klim)), b123[klim][ijl], c=c, s=5, label=lbl) 
        sub.plot(range(np.sum(klim)), b123[klim][ijl], c=c) 

    sub.legend(loc='upper right', markerscale=4, handletextpad=0., fontsize=20) 
    sub.set_ylabel('$B(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim([1e3, 5e6]) 
    sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    fig.savefig(os.path.join(dir_fpm, 'bk_fpm_tng_z%.2f%s.png' % (z, str_flag)), bbox_inches='tight')


    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    for fh, lbl, c in zip([f_fpm, f_tng], ['FastPM', 'Illustris TNG'], ['k', 'C1']): 
        i_k, j_k, l_k, p0k1, p0k2, p0k3, b123, q123, counts, b123_sn = np.loadtxt(f_b123(fh), 
                skiprows=1, unpack=True, usecols=range(10)) 
        klim = ((i_k *kf >= 0.01) & (j_k *kf >= 0.01) & (l_k *kf >= 0.01)) 
        i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]
        ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
   
        sub.scatter(range(np.sum(klim)), q123[klim][ijl], c=c, s=5, label=lbl) 
        sub.plot(range(np.sum(klim)), q123[klim][ijl], c=c) 

    sub.legend(loc='upper right', markerscale=4, handletextpad=0., fontsize=20) 
    sub.set_ylabel('$Q(k_1, k_2, k_3)$', fontsize=25) 
    sub.set_ylim([0., 1.]) 
    sub.set_xlabel(r'$k_1 \le k_2 \le k_3$ triangle indices', fontsize=25) 
    sub.set_xlim([0, np.sum(klim)])
    fig.savefig(os.path.join(dir_fpm, 'qk_fpm_tng_z%.2f%s.png' % (z, str_flag)), bbox_inches='tight')
    return None


if __name__=="__main__": 
    for z in [0., 0.52, 1.04, 2.]: 
        #fastPM(z)
        #fastPM(z, str_flag='_calibratebias')
        #TNG(z) 
        #fastpm_v_tng(z, Lbox=205., Nmax=40, Ncut=3, step=3)
        fastpm_v_tng(z, str_flag='_calibratebias', Lbox=205., Nmax=40, Ncut=3, step=3)
