'''

General utility functions 

'''
import numpy as np 

def ijl_order(i_k, j_k, l_k, typ='GM'): 
    ''' triangle configuration ordering, returns indices
    '''
    i_bq = np.arange(len(i_k))
    if typ == 'GM': # same order as Hector's
        i_bq_new = [] 

        l_usort = np.sort(np.unique(l_k))
        for l in l_usort: 
            j_usort = np.sort(np.unique(j_k[l_k == l]))
            for j in j_usort: 
                i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                for i in i_usort: 
                    i_bq_new.append(i_bq[(i_k == i) & (j_k == j) & (l_k == l)])
    else: 
        raise NotImplementedError
    return np.array(i_bq_new)


def radecz_to_cartesian(radecz, cosmo=None):  
    ''' convert RA, Dec, z values to cartesian coordinates using given
    cosmology
    
    :param radecz: 
        [3, N] numpy array of right ascension, declination, redshift. RA and
        Dec are assumed to be in degrees. 

    '''
    assert radecz.shape[0] == 3, "radecz has to be have shape [3,N]"

    ra, dec, z = radecz 
    
    # convert to radian 
    ra  *= np.pi / 180. 
    dec *= np.pi / 180. 
    
    # calculate radial comoving distance in Mpc/h
    rad = cosmo.comoving_distance(z) 

    xyz = np.array([
        rad * np.cos(dec) * np.cos(ra),
        rad * np.cos(dec) * np.sin(ra),
        rad * np.sin(dec)])
    return xyz 


def applyRSD(xyz, vxyz, redshift, h=0.7, omega0_m=0.3, LOS=None, Lbox=None): 
    ''' Calculate redshift-space positions using the real-space position,
    velocities, and LOS direction for periodic box.
    '''
    assert xyz.shape[0] == 3 # xyz and vxyz should be 3 x Ngal arrays
    assert vxyz.shape[0] == 3 
    if LOS is None: raise ValueError("specify line of sight") 
    if Lbox is None: raise ValueError("specify box size") 
    _los = {'x': 0, 'y': 1, 'z': 2} 
    i_los = _los[LOS]

    # cosmology
    H0 = 100. * h 
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega0_m)  

    # the RSD normalization factor
    rsd_factor = (1+redshift) / (100 * cosmo.efunc(redshift))

    xyz_rsd = xyz.copy() 
    xyz_rsd[i_los] += rsd_factor * vxyz[i_los] + Lbox 
    xyz_rsd[i_los] = (xyz_rsd[i_los] % Lbox) 
    return xyz_rsd


def read_fortFFT(file=None): 
    ''' Read FFT grid from fortran output and return delta[i_k,j_k,l_k]
    '''
    f = FortranFile(file, 'r')
    Ngrid = f.read_ints(dtype=np.int32)[0]
    delt = f.read_reals(dtype=np.complex64) 
    delt = np.reshape(delt, (Ngrid/2+1, Ngrid, Ngrid), order='F')

    # reflect half field
    delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.complex64)
    # i = 0 - Ngrid // 2 (confirmed correct) 
    delta[:Ngrid//2+1, :, :] = delt[:,:,:]
    
    # i = Ngrid // 2 - Ngrid (confirmed corect)
    delta[:Ngrid//2:-1, Ngrid:0:-1, Ngrid:0:-1] = np.conj(delt[1:Ngrid//2,1:Ngrid,1:Ngrid])
    delta[:Ngrid//2:-1, Ngrid:0:-1, 0] = np.conj(delt[1:Ngrid//2,1:Ngrid,0])
    delta[:Ngrid//2:-1, 0, Ngrid:0:-1] = np.conj(delt[1:Ngrid//2,0,1:Ngrid])
    # reflect the x-axis
    delta[:Ngrid//2:-1,0,0] = np.conj(delt[1:Ngrid//2,0,0])

    hg = Ngrid//2
    delta[hg,0,0]    = np.real(delt[hg,0,0])
    delta[0,hg,0]    = np.real(delt[0,hg,0])
    delta[0,0,hg]    = np.real(delt[0,0,hg])
    delta[0,hg,hg]   = np.real(delt[0,hg,hg])
    delta[hg,0,hg]   = np.real(delt[hg,0,hg])
    delta[hg,hg,0]   = np.real(delt[hg,hg,0])
    delta[hg,hg,hg]  = np.real(delt[hg,hg,hg])
    return delta 

