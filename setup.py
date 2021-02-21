import os
import numpy as np 
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

__version__ = '0.0'

try: 
    if 'NERSC_HOST' in os.environ.keys(): 
        if os.environ['NERSC_HOST'] == 'edison': 
            ext = Extension(name='estimator', 
                    sources=['pyspectrum/estimator.f'], 
                    language='f77', 
                    library_dirs = ["/opt/cray/pe/fftw/3.3.8.1/x86_64/lib"],
                    libraries = ['fftw3f'], 
                    include_dirs=[np.get_include(), "/opt/cray/pe/fftw/3.3.8.1/x86_64/include"])
        elif os.environ['NERSC_HOST'] == 'cori': 
            ext = Extension(name='estimator', 
                    sources=['pyspectrum/estimator.f'], 
                    language='f77', 
                    library_dirs = ["/opt/cray/pe/fftw/default/x86_64/lib"],
                    libraries = ['fftw3f'], 
                    include_dirs=[np.get_include(), "/opt/cray/pe/fftw/default/x86_64/include"])
        else: 
            raise KeyError
    elif 'machine' in os.environ.keys(): 
        if os.environ['machine'] == 'tiger': 
            print('install on princeton Tiger') 
            ext = Extension(name='estimator', 
                    sources=['pyspectrum/estimator.f'], 
                    language='f77', 
                    library_dirs = ["/usr/local/fftw/intel-16.0/3.3.4/lib64"], 
                    libraries = ['fftw3f'], 
                    include_dirs=[np.get_include(), "/usr/local/fftw/intel-16.0/3.3.4/include"])
        elif os.environ['machine'] == 'mbp': 
            ext = Extension(name='estimator', 
                    sources=['pyspectrum/estimator.f'], 
                    language='f77', 
                    library_dirs = ["/usr/local/lib"],
                    libraries = ['fftw3f'], 
                    include_dirs=[np.get_include(), '/usr/local/include'],
                    extra_f77_compile_args=['-fcheck=all', '-fallow-argument-mismatch'])
    else: 
        raise KeyError
except KeyError: 
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
          author_email='hahn.changhoon@gmail.com',
          url='',
          package_data={'pyspectrum': ['dat/fftw3.f', 'dat/*.pyfftw', 'dat/test_box.hdf5']},
          platforms=['*nix'],
          license='GPL',
          requires = ['numpy', 'scipy', 'h5py', 'pyfftw', 'pytest'],
          provides = ['pyspectrum'],
          packages = ['pyspectrum'], 
          ext_modules = [ext]
          )
