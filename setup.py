import numpy as np 
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

__version__ = '0.0'

ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'], 
        language='f77', 
        library_dirs = ["/usr/local/lib"],
        libraries = ['fftw3f'],
        include_dirs=[np.get_include(), '/usr/local/include'])

if __name__=="__main__": 
    setup(name = 'pySpectrum',
          version = __version__,
          description = 'TBD',
          author='ChangHoon Hahn',
          author_email='hahn.changhoon@gmail.com',
          url='',
          platforms=['*nix'],
          license='GPL',
          requires = ['numpy', 'scipy', 'matplotlib', 'pyfftw'],
          provides = ['pyspectrum'],
          packages = ['pyspectrum'], 
          ext_modules = [ext]
          )
