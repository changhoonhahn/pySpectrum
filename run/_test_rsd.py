''' 

test that the rsd portion of the pyspectrum.pyspectrum works as expected

'''
import os 
import numpy as np 
from scipy.io import FortranFile 
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


if __name__=="__main__": 
    kf = 2.*np.pi/2600.

    x, y, z, vx, vy, vz = np.loadtxt(os.path.join(UT.dat_dir(), 'BoxN1.mock'), unpack=True, usecols=[0,1,2,3,4,5]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    vxyz = np.zeros((3, len(x))) 
    vxyz[0,:] = vx
    vxyz[1,:] = vy 
    vxyz[2,:] = vz

    s_xyz = pySpec.applyRSD(xyz, vxyz, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=2600.) 

    f = FortranFile(os.path.join(UT.dat_dir(), 'r_rsd'), 'r')
    xyz_f = f.read_reals(dtype=np.float32) 
    xyz_f = np.reshape(xyz_f, (3, xyz_f.shape[0]/3), order='F')
    
    for i, ax in zip(range(3), ['x', 'y', 'z']):  
        print(ax)
        print(s_xyz[i,:10])
        print(xyz_f[i,:10])

    _delta = pySpec.FFTperiodic(s_xyz, Lbox=2600, Ngrid=360, silent=False) 
    delta = pySpec.reflect_delta(_delta, Ngrid=360, silent=False)
    
    delt = pySpec.read_fortFFT(file=os.path.join(UT.dat_dir(), 'FFT.BoxN1.mock.rsd_z.Ngrid360'))

    print (delta-delt)[:10,0,0]
    print delta.ravel()[np.argmax(np.abs(delta-delt))]
    print delt.ravel()[np.argmax(np.abs(delta-delt))]

    # calculate powerspectrum monopole  
    k, p0k, cnts = pySpec.Pk_periodic(delta) 
    
    f_pk = os.path.join(UT.dat_dir(), 'p0k.rsd_test.dat') 
    f_b123 = os.path.join(UT.dat_dir(), 'B123.rsd_test.dat') 
    # save to file 
    hdr = 'pyspectrum P_l=0(k) calculation'
    np.savetxt(f_pk, np.array([k*kf, p0k/(kf**3), cnts]).T, fmt='%.5e %.5e %.5e', delimiter='\t', header=hdr) 
    # calculate bispectrum 
    bisp = pySpec.Bk123_periodic(
            delta, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=False) 
    # save to file 
    hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/3600'
    np.savetxt(f_b123, 
            np.array([
                bisp['i_k1'], 
                bisp['i_k2'], 
                bisp['i_k3'], 
                bisp['b123'],
                bisp['q123'], 
                bisp['counts']]).T, fmt='%i %i %i %.5e %.5e %.5e', 
            delimiter='\t', header=hdr) 
    i_k, j_k, l_k, b123, q123 = np.loadtxt(f_b123, unpack=True, skiprows=1, usecols=[0,1,2,3,4]) 

    f_b_fort = os.path.join(UT.dat_dir(), 'BISP.BoxN1.mock.rsd_z.Ngrid360') 
    k_i, k_j, k_l, pk_i, pk_j, pk_l, _b123, _q123 = np.loadtxt(f_b_fort, unpack=True, usecols=[0,1,2,3,4,5,6,7]) 
    k, p0k = np.loadtxt(f_pk, unpack=True, usecols=[0,1]) 

    # compare powerspectrum 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.plot(k, p0k, c='k', lw=1) 
    sub.plot(kf*k_i, (2*np.pi)**3*pk_i/(kf**3), c='C1', lw=1) 
    j_sort = np.argsort(k_j)
    sub.plot(kf*k_j[j_sort], (2*np.pi)**3*pk_j[j_sort]/(kf**3), c='C2', lw=1) 
    l_sort = np.argsort(k_l)
    sub.plot(kf*k_l[l_sort], (2*np.pi)**3*pk_l[l_sort]/(kf**3), c='C3', lw=1) 
    sub.set_ylabel('$P_0(k)$', fontsize=25) 
    sub.set_ylim([3e3, 2e5]) 
    sub.set_yscale('log') 
    sub.set_xlabel('$k$', fontsize=25) 
    sub.set_xlim([3e-3, 1.]) 
    sub.set_xscale('log') 
    fig.savefig(''.join([UT.dat_dir(), 'p0k.rsd_test.png']), bbox_inches='tight')

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    sub.scatter(range(len(i_k)), (2*np.pi)**6 * b123 / kf**6, c='k', s=1) 
    sub.scatter(range(len(k_i)), (2*np.pi)**6 * _b123 / kf**6, c='C1', s=1) 
    sub.set_xlabel('triangles', fontsize=25)
    sub.set_xlim([0, len(i_k)])
    sub.set_ylabel(r'$B(k_1, k_2, k_3)$ (not SN corrected)', fontsize=25)
    sub.set_yscale('log') 
    sub.set_ylim([1e7, 8e9])
    fig.savefig(''.join([UT.dat_dir(), 'b123.rsd_test.png']), bbox_inches='tight')
