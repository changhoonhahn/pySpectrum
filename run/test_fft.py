''' test pyspectrum.fft and make sure it's the same as the fortran 
code output 
'''
import pyfftw
import numpy as np 
from scipy.io import FortranFile 
# -- pyspectrum -- 
import estimator as fEstimate
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec


if __name__=="__main__": 
    x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock']), unpack=True, usecols=[0,1,2]) 
    #x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock1000000']), unpack=True, usecols=[0,1,2]) 
    #x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'BoxN1.mock100']), unpack=True, usecols=[0,1,2]) 
    xyz = np.zeros((3, len(x))) 
    xyz[0,:] = x
    xyz[1,:] = y 
    xyz[2,:] = z

    #delta = pySpec.FFTing(xyz, Lbox=2600, Ngrid=360, silent=False) 
    #f = FortranFile(''.join([UT.dat_dir(), 'zmap.fft.dcl']), 'r')
    #_delt = f.read_reals(dtype=np.complex64) 
    #_delt = np.reshape(_delt, (360, 360, 360), order='F') 
    #fEstimate.fcomb(_delt,len(x),360) 
    #__delt = _delt[:181,:,:] 
    ''' 
    delta = pySpec.FFTperiodic(xyz, Lbox=2600, Ngrid=360, silent=False, test='assign') 
    f = FortranFile(''.join([UT.dat_dir(), 'zmap.dcl']), 'r')
    delta_assign = f.read_reals(np.complex64) 
    delta_assign = np.reshape(delta_assign, (360, 360, 360), order='F') 
    assert np.array_equal(delta, delta_assign)
    print('delta array agrees after assign!') 
    print('------------') 
    print('delta array after pyFFTW') 
    delta = pySpec.FFTperiodic(xyz, Lbox=2600, Ngrid=360, fft='pyfftw', silent=False, test='fft') 
    f = FortranFile(''.join([UT.dat_dir(), 'zmap.fft.dcl.fftw2']), 'r')
    delt = f.read_reals(dtype=np.complex64) 
    delt = np.reshape(delt, (360, 360, 360), order='F') 
    _delt = delt.copy() 
    print (delta-delt)[:10,0,0]
    print np.allclose(delta, delt) 
    #assert np.array_equal(delta, delt) 
    print('------------') 
    print('delta array after FFTW3') 

    delta = pySpec.FFTperiodic(xyz, Lbox=2600, Ngrid=360, fft='fftw3', silent=False, test='fft') 
    f = FortranFile(''.join([UT.dat_dir(), 'zmap.fft.dcl']), 'r')
    delt = f.read_reals(dtype=np.complex64) 
    delt = np.reshape(delt, (360, 360, 360), order='F') 
    print (delta-delt)[:10,0,0]
    print (delt-_delt)[:10,0,0]
    #assert np.array_equal(delta, delt) 
    #print('delta array agrees after fft!') 

    print('------------') 
    print('delta array FFT output') 
    # read in fortran code FFT output 
    delta = pySpec.FFTperiodic(xyz, Lbox=2600, Ngrid=360, silent=False, test='fcomb') 
    #f = FortranFile(''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']), 'r')
    f = FortranFile(''.join([UT.dat_dir(), 'FFT.BoxN1.mock1000000.Ngrid360']), 'r')
    _ = f.read_ints(dtype=np.int32)
    delt = f.read_reals(dtype=np.complex64) 
    delt = np.reshape(delt, (181, 360, 360), order='F')
    print (delta-delt)[:10,0,0]
    print delta.ravel()[np.argmax(np.abs(delta-delt))]
    print delt.ravel()[np.argmax(np.abs(delta-delt))]
    print np.allclose(delta, delt)
    '''
    
    print('------------') 
    print('delta array after reflect') 
    #_delta_fft = pySpec.read_fortFFT(file=''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']))
    _delta = pySpec.FFTperiodic(xyz, Lbox=2600, Ngrid=360, silent=False) 
    delta = pySpec.reflect_delta(_delta, Ngrid=360, silent=False)
    #delt = pySpec.read_fortFFT(file=''.join([UT.dat_dir(), 'FFT.BoxN1.mock1000000.Ngrid360']))
    delt = pySpec.read_fortFFT(file=''.join([UT.dat_dir(), 'FFT.BoxN1.mock.Ngrid360']))
    print (delta-delt)[:10,0,0]
    print delta.ravel()[np.argmax(np.abs(delta-delt))]
    print delt.ravel()[np.argmax(np.abs(delta-delt))]
    print np.allclose(delta, delt)
