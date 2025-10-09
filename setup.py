import os
import numpy as np 
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

__version__ = '0.1'

ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'], 
        language='f77', 
        library_dirs = ["/usr/local/lib"],
        libraries = ['fftw3f'], 
        include_dirs=[np.get_include(), '/usr/local/include'],
        extra_f77_compile_args=['-fcheck=all', '-fallow-argument-mismatch'])

if __name__=="__main__": 
    setup(name = 'pySpectrum',
          version = __version__,
          description = 'TBD',
          author='ChangHoon Hahn',
          author_email='changhoon.hahn@utexas.edu',
          url='',
          package_data={'pyspectrum': ['dat/fftw3.f', 'dat/*.pyfftw', 'dat/test_box.hdf5']},
          platforms=['*nix'],
          license='GPL',
          requires = ['numpy', 'scipy', 'h5py', 'pyfftw', 'pytest'],
          provides = ['pyspectrum'],
          packages = ['pyspectrum'], 
          ext_modules = [ext]
          )
