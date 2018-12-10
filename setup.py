#from setuptools import setup, find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension


__version__ = '0.0'

ext = Extension(name='estimator', sources=['pyspectrum/estimator.f'])

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
