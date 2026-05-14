# pySpectrum
Python package for calculating the galaxy/halo/dark matter power spectrum and bispectrum using the [Scoccimarro (2015)](https://ui.adsabs.harvard.edu/abs/2015PhRvD..92h3532S/abstract) estimator. 

- [Installation](#installation)
- [Coming Soon](#coming-soon)
- [Citaton](#citation) 
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

## Citation 
If you use the package please consider citing 

```
@article{Scoccimarro_2015,
   title={Fast estimators for redshift-space clustering},
   volume={92},
   ISSN={1550-2368},
   url={http://dx.doi.org/10.1103/PhysRevD.92.083532},
   DOI={10.1103/physrevd.92.083532},
   number={8},
   journal={Physical Review D},
   publisher={American Physical Society (APS)},
   author={Scoccimarro, Román},
   year={2015},
   month=Oct }
```
and
```
@article{Hahn_2020,
   title={Constraining <i>M</i><sub>ν</sub> with the bispectrum. Part I. Breaking parameter degeneracies},
   volume={2020},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2020/03/040},
   DOI={10.1088/1475-7516/2020/03/040},
   number={03},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Hahn, ChangHoon and Villaescusa-Navarro, Francisco and Castorina, Emanuele and Scoccimarro, Roman},
   year={2020},
   month=Mar, pages={040–040} }
```

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
