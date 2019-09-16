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

    x, y, z, z_rsd = np.loadtxt(os.path.join(dir_mc, _fname), unpack=True, usecols=[0, 1, 2, 3]) 

    print('--- %s mock ---' % name) 
    print('%.1f < x < %.1f' % (x.min(), x.max()))
    print('%.1f < y < %.1f' % (y.min(), y.max()))
    print('%.1f < z < %.1f' % (z.min(), z.max()))
    print('%.1f < z_rsd < %.1f' % (z_rsd.min(), z_rsd.max()))
    
    # real power spectrum
    xyz = np.array([x, y, z]) 
    p0k_real = pySpec.Pk_periodic(xyz, Lbox=1000, Ngrid=512, fft='pyfftw', silent=False) 
    
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.plot(p0k_real['k'], p0k_real['p0k'], c='k')
    sub.set_xlabel('$k$ (Mpc/$h$)', fontsize=25) 
    sub.set_xlim(1e-2, 5)
    sub.set_xscale("log")
    sub.set_ylabel('real-space $P(k)$', fontsize=25) 
    sub.set_yscale("log") 
    fig.savefig(os.path.join(dir_mc, 'pk_real.%s.png' % _fname.replace('.txt', '')), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    stage1('unit')
