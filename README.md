# pySpectrum
Python package for calculating the galaxy/halo/dark matter power spectrum and bispectrum. 

- [requirements](#requirements) 
- [installation](#installation)
- [To-do](#to-do)

## requirements
required packages: 
* numpy 
* scipy
* astropy
* matplotlib
* f2py

## Installation
clone the repo and run setpy.py
```bash
git clone https://github.com/changhoonhahn/pySpectrum.git
cd pySpectrum
python setup.py install
```
Also make sure to set the environment variable $PYSPEC_CODEDIR to the location of the repo 
in your .bashrc or .bash_profile --- e.g. 
```bash
export PYSPEC_CODEDIR=/location/of/pySpectrum/
```

## To-do 
coming soon: 
* implement test scripts 
* release power spectrum quadrupole for periodic box
