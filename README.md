# pySpectrum
Python package for calculating the galaxy/halo/dark matter power spectrum and bispectrum using [the Scoccimarro (2015)](https://ui.adsabs.harvard.edu/abs/2015PhRvD..92h3532S/abstract) estimator. 

- [Installation](#installation)
- [Coming Soon](#coming-soon)
- [Contact](#contact) 

## Installation
To install the package, simply run 
```bash
python -m pip install git+https://github.com/changhoonhahn/pySpectrum.git
```

There's some complications with using FFTW3 installs. I've listed some examples on different HPCs in [install examples](#install-examples). 

required packages: 
* numpy 
* scipy
* astropy
* matplotlib
* f2py

Also requires [`FFTW3`](http://www.fftw.org/install/mac.html). On macs, you can install `FFTW3` using [homebrew](https://formulae.brew.sh/formula/fftw).

## Coming Soon

## Contact
If you have any questions or need help using the package, feel free to contact me at changhoon.hahn@utexas.edu


-----
## install examples 

### TACC 
```
module load fftw3/3.3.10

conda activate ENVNAME

python -m pip install git+https://github.com/changhoonhahn/pySpectrum.git

```
