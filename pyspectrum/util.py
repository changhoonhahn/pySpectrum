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
