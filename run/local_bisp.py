'''

calculate bispectrum 

'''
import time 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec

if __name__=="__main__": 
    # read in FFTed density grid
    f_fft = ''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']) 
    delta = pySpec.read_fortFFT(file=f_fft) 
    t0 = time.time() 
    # calculate bispectrum 
    i_k, j_k, l_k, bisp = pySpec.Bk123_periodic(
            delta, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=False) 
    print('Bispectrum calculation takes %f mins' % ((time.time() - t0)/60.))
    # save to file 
    hdr = '# pyspectrum bispectrum calculation test. k_f = 2pi/3600 \n'
    f_b123 = ''.join([UT.dat_dir(), 'pySpec.B123.BoxN1.mock.Ngrid360', 
        '.Ngrid360', 
        '.Nmax40', 
        '.Ncut3', 
        '.step3', 
        '.pyfftw', 
        '.dat']) 
    f = open(f_b123, 'w') 
    f.write(hdr) 
    for i in range(len(i_k)): 
        f.write('%i\t%i\t%i\t%f\n' % (i_k[i], j_k[i], l_k[i], bisp[i])) 
    f.close() 
