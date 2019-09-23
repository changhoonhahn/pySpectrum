#!/bin/python 
'''

scripts for the gqp mock challenge

'''
import os 
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


dir_mc = os.path.join(UT.dat_dir(), 'gqc_mock_challenge') 


def stage1(name): 
    ''' stage 1 of mock challenge. run power spectrum for periodic box 
    '''
    # read in mock 
    if name == 'unit': 
        _fname = 'UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt'
        Lbox = 1000.
    elif name == 'pmill': 
        _fname = 'DESI_ELG_z0.76_catalogue.dat'
        Lbox = 542.16

    x, y, z, z_rsd = np.loadtxt(os.path.join(dir_mc, _fname), unpack=True, usecols=[0, 1, 2, 3]) 

    print('--- %s mock ---' % name) 
    print('%.1f < x < %.1f' % (x.min(), x.max()))
    print('%.1f < y < %.1f' % (y.min(), y.max()))
    print('%.1f < z < %.1f' % (z.min(), z.max()))
    print('%.1f < z_rsd < %.1f' % (z_rsd.min(), z_rsd.max()))
    
    # real power spectrum
    xyz = np.array([x, y, z]) 
    p0k_real = pySpec.Pk_periodic(xyz, Lbox=Lbox, Ngrid=512, fft='pyfftw', silent=False) 

    # redshift-space power spectrum
    xyz_s = np.array([x, y, z_rsd]) 
    pk_rsd = pySpec.Pk_periodic_rsd(xyz_s, Lbox=Lbox, Ngrid=512, rsd=2, Nmubin=120, fft='pyfftw', code='fortran', silent=False) 

    # write out Pl(k)
    fout = os.path.join(dir_mc, 'Pkl_lin_HAHN_%s_1.txt' % name.upper()) 
    np.savetxt(fout, 
            np.array([pk_rsd['k'], pk_rsd['p0k'], pk_rsd['p2k'], pk_rsd['p4k'], pk_rsd['counts']]).T, 
            header='k, p0k, p2k, p4k, n_modes', 
            fmt='%.5e %.5e %.5e %.5e %i') 
    
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.plot(p0k_real['k'], p0k_real['p0k'], c='k', ls=':', label='real-space')
    sub.plot(pk_rsd['k'], pk_rsd['p0k'], c='k', label='$\ell=0$') 
    sub.plot(pk_rsd['k'], pk_rsd['p2k'], c='C0', label='$\ell=2$') 
    sub.plot(pk_rsd['k'], pk_rsd['p4k'], c='C1', label='$\ell=4$') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('$k$ (Mpc/$h$)', fontsize=25) 
    sub.set_xlim(1e-2, 2)
    sub.set_xscale("log")
    sub.set_ylabel('$P_\ell(k)$', fontsize=25) 
    sub.set_yscale("log") 
    sub.set_ylim(3., 5e4) 
    fig.savefig(os.path.join(dir_mc, 'plk.%s.png' % _fname.replace('.txt', '')), bbox_inches='tight') 

    # write out 2D P(k,mu) 
    fout = os.path.join(dir_mc, 'pk2D_lin_HAHN_%s_1.txt' % name.upper()) 
    np.savetxt(fout, 
            np.array([
                pk_rsd['k_kmu'].flatten(), 
                pk_rsd['mu_kmu'].flatten(), 
                pk_rsd['p_kmu'].flatten(), 
                pk_rsd['counts_kmu'].flatten()]).T, 
            header='k, mu, p(k,mu), n_modes', 
            fmt='%.5e %.5e %.5e %i')  

    fig = plt.figure(figsize=(6,5))
    sub = fig.add_subplot(111)
    cm = sub.pcolormesh(pk_rsd['mu_kmu'], pk_rsd['k_kmu'], pk_rsd['p_kmu'], norm=LogNorm(vmin=1e3, vmax=1e5))
    sub.set_yscale('log') 
    sub.set_ylim(1e-2, 2) 
    fig.savefig(os.path.join(dir_mc, 'pk2d.%s.png' % _fname.replace('.txt', '')), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #stage1('unit')
    stage1('pmill')
