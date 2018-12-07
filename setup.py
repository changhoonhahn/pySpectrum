#from setuptools import setup, find_packages
from distutils.core import setup

__version__ = '0.0'

setup(name = 'pySpectrum',
      version = __version__,
      description = 'TBD',
      author='ChangHoon Hahn',
      author_email='hahn.changhoon@gmail.com',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'matplotlib', 'scipy', 'pyfftw'],
      provides = ['pyspectrum'],
      packages = ['pyspectrum']
      )
