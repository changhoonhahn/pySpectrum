''' 

test that the rsd portion of the pyspectrum.pyspectrum works as expected

'''
import numpy as np 
from scipy.io import FortranFile 
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec


if __name__=="__main__": 
    x, y, z, vx, vy, vz = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock']), unpack=True, usecols=[0,1,2,3,4,5]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    vxyz = np.zeros((3, len(x))) 
    vxyz[0,:] = vx
    vxyz[1,:] = vy 
    vxyz[2,:] = vz

    s_xyz = pySpec.applyRSD(xyz, vxyz, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=2600.) 

    f = FortranFile(''.join([UT.dat_dir(), 'r_rsd']), 'r')
    xyz_f = f.read_reals(dtype=np.float32) 
    xyz_f = np.reshape(xyz_f, (3, xyz_f.shape[0]/3), order='F')
    
    for i, ax in zip(range(3), ['x', 'y', 'z']):  
        print(ax)
        print(s_xyz[i,:10])
        print(xyz_f[i,:10])

    _delta = pySpec.FFTperiodic(s_xyz, Lbox=2600, Ngrid=360, silent=False) 
    delta = pySpec.reflect_delta(_delta, Ngrid=360, silent=False)
    
    delt = pySpec.read_fortFFT(file=''.join([UT.dat_dir(), 'FFT.BoxN1.mock.rsd_z.Ngrid360']))

    print (delta-delt)[:10,0,0]
    print delta.ravel()[np.argmax(np.abs(delta-delt))]
    print delt.ravel()[np.argmax(np.abs(delta-delt))]

    # calculate bispectrum 
    i_k, j_k, l_k, b123, q123, counts = pySpec.Bk123_periodic(
            delta, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', nthreads=1, silent=False) 
    # save to file 
    f_b123 = ''.join([UT.dat_dir(), 'B123.rsd_test.dat']) 
    hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/3600'
    np.savetxt(f_b123, np.array([i_k, j_k, l_k, b123, q123, counts]).T, fmt='%i %i %i %.5e %.5e %.5e', 
            delimiter='\t', header=hdr) 
