__all__ = ['test_pySpec'] 

import os 
import h5py 
import pytest 
from itertools import product 
from pyspectrum import dat_dir
from pyspectrum import pyspectrum as pySpec


@pytest.mark.parametrize(("pt", "space"), product((2, 3), ('real', 'rsd')))
def test_pySpec(pt, space): 
    # read test data 
    fbox    = h5py.File(os.path.join(dat_dir(), 'test_box.hdf5'), 'r') 
    xyz     = fbox['xyz'][...]
    vxyz    = fbox['vxyz'][...]

    Lbox    = 2600.         # box size
    Ngrid   = 360           # fft grid size

    if space == 'rsd': 
        xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=Lbox) 

    if pt == 2:  # 2pt (power spectrum)
        if space == 'real':
            spec = pySpec.Pk_periodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
        elif space == 'rsd': 
            spec = pySpec.Pk_periodic(xyz_s, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
        assert spec['p0k'].mean() > 0.
    elif pt == 3: # 3pt (bispectrum) 
        if space == 'real': 
            spec = pySpec.Bk_periodic(xyz.T, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
        elif space == 'rsd': 
            spec = pySpec.Bk_periodic(xyz_s, Lbox=Lbox, Ngrid=Ngrid, silent=False) 
        print(spec['p0k1']) 
        print(spec['counts']) 
        print(spec['b123'])
        assert spec['b123'].mean() > 0.
