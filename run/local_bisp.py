'''

calculate bispectrum 

'''
import time 
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec


if __name__=="__main__": 
    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z
    
    delta = pySpec.FFTperiodic(xyz, fft='fortran', Lbox=2600, Ngrid=360, silent=False) 
    delta_fft = pySpec.reflect_delta(delta, Ngrid=360) 
    
    t0 = time.time()
    # calculate bispectrum 
    i_k, j_k, l_k, b123, q123 = pySpec.Bk123_periodic(
            delta_fft, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', nthreads=2, silent=False) 
    print('Bispectrum calculation takes %f mins' % ((time.time() - t0)/60.))
    # save to file 
    hdr = 'pyspectrum bispectrum calculation test. k_f = 2pi/3600'
    f_b123 = ''.join([UT.dat_dir(), 'pySpec.B123.BoxN1.mock.Ngrid360', 
        '.Ngrid360', 
        '.Nmax40', 
        '.Ncut3', 
        '.step3', 
        '.pyfftw', 
        '.dat']) 
    np.savetxt(f_b123, np.array([i_k, j_k, l_k, b123, q123]).T, fmt='%i %i %i %.5e %.5e', 
            delimiter='\t', header=hdr) 

    _i, _j, _l, _b123, _q123 = np.loadtxt(''.join([UT.dat_dir(), 'bisp.mbp.BoxN1.mock.Ngrid360']), 
            unpack=True, usecols=[0,1,2,6,7]) 
    print b123[:10]
    print _b123[:10]
    print round(q123[:10],5) 
    print _q123[:10]
    assert np.allclose(b123, _b123) 
    assert np.allclose(q123, _q123) 

    #f = open(f_b123, 'w') 
    #f.write(hdr) 
    #for i in range(len(i_k)): 
    #    f.write('%i\t%i\t%i\t%f\n' % (i_k[i], j_k[i], l_k[i], bisp[i])) 
    #f.close() 
