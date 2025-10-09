# pySpectrum
Python package for calculating the galaxy/halo/dark matter power spectrum and bispectrum. 

- [Installation](#installation)
- [Coming Soon](#coming-soon)
- [Contact](#contact) 

## Installation
required packages: 
* numpy 
* scipy
* astropy
* matplotlib
* f2py

Also requires [`FFTW3`](http://www.fftw.org/install/mac.html). You can also
install `FFTW3` using [homebrew](https://formulae.brew.sh/formula/fftw).

To install the package, clone the github repo and run setpy.py
```bash
git clone https://github.com/changhoonhahn/pySpectrum.git
cd pySpectrum
```

At the moment installation is quite janky. You'll need to modify the following
lines in `setup.py` to point to your fftw3: 
```python 
ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'], 
        language='f77', 
        library_dirs = ["/usr/local/lib"], # modify this 
        libraries = ['fftw3f'], 
        include_dirs=[np.get_include(), '/usr/local/include'], # modify this 
        extra_f77_compile_args=['-fcheck=all', '-fallow-argument-mismatch'])
```
I've listed some examples on different machines in [setup.py examples](#setup.py-examples). 
Once you've modified `setup.py` run 

```bash 
python setup.py install
```

## Coming Soon
* power spectrum quadrupole for periodic box
* less janky install 

## Contact
If you have any questions or need help using the package, feel free to contact me at changhoon.hahn@utexas.edu


-----
## setup.py examples 
**mac with FFTW installed using homebrew**
```python 
ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'], 
        language='f77', 
        library_dirs = ["/opt/homebrew/Cellar/fftw/3.3.10_2/lib"],
        libraries = ['fftw3f'], 
        include_dirs=[np.get_include(), '/opt/homebrew/Cellar/fftw/3.3.10_2/include'], 
        extra_f77_compile_args=['-fcheck=all', '-fallow-argument-mismatch'])
```

**NERSC Cori**
```python
ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'], 
        language='f77', 
        library_dirs = ["/opt/cray/pe/fftw/default/x86_64/lib"],
        libraries = ['fftw3f'], 
        include_dirs=[np.get_include(), "/opt/cray/pe/fftw/default/x86_64/include"])
```

**Princeton Tiger cluster**
```python 
ext = Extension(name='estimator', 
        sources=['pyspectrum/estimator.f'], 
        language='f77', 
        library_dirs = ["/usr/local/fftw/intel-16.0/3.3.4/lib64"], 
        libraries = ['fftw3f'], 
        include_dirs=[np.get_include(), "/usr/local/fftw/intel-16.0/3.3.4/include"])
```

