from numpy.distutils.core import setup
from numpy.distutils.core import Extension

__version__ = '0.0'

ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'],
        extra_link_args=['-L/usr/local/lib', '-lfftw3', '-I/usr/local/include'])
#        extra_link_args=['-L/usr/local/lib', '-lrfftw', '-lfftw', '-I/usr/local/include'])

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
