import os 
import time
import pyfftw
import numpy as np 
from scipy.io import FortranFile
from astropy.cosmology import FlatLambdaCDM
# -- local -- 
import estimator as fEstimate
from . import dat_dir 


def Bk_periodic(xyz, Lbox=2600, Ngrid=360, step=3, Ncut=3, Nmax=40, fft='pyfftw', nthreads=1, silent=True): 
    ''' calculate the bispectrum for periodic box. this function is a wrapper for FFTperiodic
    and Bk123_periodic, which contains the actual calculations. 

    :param xyz: 
        3xN dimensional array of the positions of objects (e.g. galaxies, halos) 

    :param Lbox: (default: 2600.)
        box size 

    :param Ngrid: (default: Ngrid) 
        grid size 

    :param step: (default: 3)
        step size in units of fundamental mode 

    :param Ncut: (default: 3)
        k minimum in units of fundamental mode 

    :param Nmax: (default: 40) 
        number of steps to include (i.e. number of modes) 
    
    :return bispec: 
        dictionary containing the bispectrum. 
        bispec.keys() = 'meta', 'i_k1','i_k2','i_k3','p0k1','p0k2','p0k3','b123','q123','counts'
        bispec['meta'] is a dictionary with meta data regarding how the bispectrum is calculated
        and such.  i_k1, i_k2, i_k3 are the triangle side lenghts in units of fundmental mode
        p0k1, p0k2, p0k3 are corresponding powerspectrum values shot noise corrected. b123 is 
        the bispectrum also shot noise corrected and q123 is the reduced bispectrum (where both
        b123 and pk1pk2pk3 are shot noise corrected. 'counts' are the number of modes. 
    '''
    N = xyz.shape[1] # number of positions 
    nbar = float(N)/Lbox**3 
    kf = 2 * np.pi / Lbox 
    if not silent: 
        print('------------------') 
        print('%i positions in %i box' % (N, Lbox))  
        print('nbar = %f' % nbar)  
    assert Ngrid == 360, "currently only tested for 360; I'm being lazy..."

    if not silent: print('--- calculating the FFT ---') 
    delta = FFTperiodic(xyz, Lbox=Lbox, Ngrid=Ngrid, fft=fft, silent=silent) 
    delta_fft = reflect_delta(delta, Ngrid=Ngrid) 

    if not silent: print('--- calculating the bispectrum ---') 
    bispec = Bk123_periodic(delta_fft, step=step, Ncut=Ncut, Nmax=Nmax, fft=fft, nthreads=nthreads, silent=silent) 

    # store some meta data for completeness  
    meta = {'Lbox': Lbox, 'Ngrid': Ngrid, 'step': step, 'Ncut': Ncut, 'Nmax': Nmax, 'N': N, 'nbar': nbar, 'kf': kf} 
    bispec['meta'] = meta 
    if not silent: print('--- correcting for shotnoise ---') 
    # convert any outputs to sensible units and apply shot noise correction! 
    bispec['p0k1'] = bispec['p0k1'] * (2*np.pi)**3/kf**3 - 1./nbar
    bispec['p0k2'] = bispec['p0k2'] * (2*np.pi)**3/kf**3 - 1./nbar
    bispec['p0k3'] = bispec['p0k3'] * (2*np.pi)**3/kf**3 - 1./nbar
    bispec['p0k_sn'] = 1./nbar

    b_shotnoise = (bispec['p0k1'] + bispec['p0k2'] + bispec['p0k3'])/nbar + 1./nbar**2
    bispec['b123'] = bispec['b123'] * (2*np.pi)**6 / kf**6 - b_shotnoise 
    bispec['b123_sn'] = b_shotnoise 

    bispec['q123'] = bispec['b123'] / (bispec['p0k1']*bispec['p0k2']+bispec['p0k1']*bispec['p0k3']+bispec['p0k2']*bispec['p0k3'])
    return bispec


def Pk_periodic_rsd(xyz, Lbox=2600, Ngrid=360, rsd=2, Nmubin=10, fft='pyfftw', code='fortran', silent=True): 
    '''calculate the powerspectrum multipole for periodic box with redshift-space distortions along `rsd` axes. 
    
    :param xyz: 
        3xN array of object positions (e.g. galaxies, halos, DM particles) 
    :param Lbox: 
        size of periodic box in Mpc/h. (default: 2600) 
    :param Ngrid: 
        FFT grid size. (default:360)  
    :param rsd:
        rsd direction {'x': 0, 'y': 1, 'z': 2}. (default: 2) 
    :param Nmubin: 
        number of Mu bins for (k, mu) binning. (default: 10) 
    :param fft:
        fftw version to use. Options are 'pyfftw' and 'fortran'. (default: pyfftw) 
    :param code: 
        If code == 'fortran' use wrapped fortran code. If code == 'python' use python code. 
        Python code is slow an inefficient and mainly there for pedagogical reasons. So 
        unless you know better use fortran. (default: fortran) 
    :param silent: 
        if True nothing is printed. 

    :return pspec:
        output dictionary with the meta data and the following colums
        * k : average k in k bin  
        * p0k : monopole
        * p2k : quadrupole
        * p4k : hexadecapole
        * counts : number of modes in k bin 
        * k_kmu : average k of (k,mu) bin  
        * mu_kmu : average mu of (k,mu) bin 
        * p_kmu : P(k,mu) 
        * counts_kmu : number of modes in (k, mu) bin
    '''
    N = xyz.shape[1] # number of positions 
    nbar = float(N)/Lbox**3 
    kf = 2 * np.pi / Lbox 
    if not silent: 
        print('------------------') 
        print('%i positions in %i box' % (N, Lbox))  
        print('nbar = %f' % nbar)  

    delta = FFTperiodic(xyz, Lbox=Lbox, Ngrid=Ngrid, fft=fft, silent=silent) 
    k, p0k, p2k, p4k, n_k, k_kmu, mu_kmu, p_kmu, n_kmu = \
            _Pk_periodic_rsd(delta, Lbox=Lbox, rsd=rsd, Nmubin=Nmubin, code=code)

    # store some meta data for completeness  
    meta = {'Lbox': Lbox, 'Ngrid': Ngrid, 'N': N, 'nbar': nbar, 'kf': kf} 
    pspec = {} 
    pspec['meta'] = meta 
    if not silent: print('--- correcting for shotnoise ---') 
    # convert any outputs to sensible units and apply shot noise correction! 
    pspec['k'] = k
    pspec['p0k'] = p0k - 1./nbar
    pspec['p2k'] = p2k
    pspec['p4k'] = p4k
    pspec['p_sn'] = np.repeat(1./nbar, len(k))
    pspec['counts'] = n_k 
    pspec['k_kmu']  = k_kmu
    pspec['mu_kmu'] = mu_kmu
    pspec['p_kmu']  = p_kmu - 1./nbar
    pspec['counts_kmu']  = n_kmu
    return pspec 


def Pk_periodic(xyz, Lbox=2600, Ngrid=360, fft='pyfftw', silent=True): 
    ''' calculate the powerspectrum for periodic box in **real**-space. For redshift-space see
    `Pk_periodic_rsd`. This function is a wrapper for FFTperiodic and _Pk_periodic
    
    :param xyz: 
        3xN array of object positions. 
    :param Lbox: 
        box size in Mpc/h. (default: 2600) 
    :param Ngrid: 
        FFT grid size. (default:360)  
    :param fft:
        fftw version to use. Options are 'pyfftw' and 'fortran'. (default: pyfftw) 
    :param silent: 
        if True nothing is printed. 
    :return pspec:
        output dictionary 
    '''
    N = xyz.shape[1] # number of positions 
    nbar = float(N)/Lbox**3 
    kf = 2 * np.pi / Lbox 
    if not silent: 
        print('------------------') 
        print('%i positions in %i box' % (N, Lbox))  
        print('nbar = %f' % nbar)  

    if not silent: print('--- calculating the FFT ---') 
    delta = FFTperiodic(xyz, Lbox=Lbox, Ngrid=Ngrid, fft=fft, silent=silent) 
    delta_fft = reflect_delta(delta, Ngrid=Ngrid) 

    k, p0k, counts = _Pk_periodic(delta_fft, Lbox=Lbox) 

    # store some meta data for completeness  
    meta = {'Lbox': Lbox, 'Ngrid': Ngrid, 'N': N, 'nbar': nbar, 'kf': kf} 
    pspec = {} 
    pspec['meta'] = meta 
    if not silent: print('--- correcting for shotnoise ---') 
    # convert any outputs to sensible units and apply shot noise correction! 
    pspec['k'] = k
    pspec['p0k'] = p0k - 1./nbar
    pspec['counts'] = counts 
    pspec['p0k_sn'] = 1./nbar
    return pspec 


def FFTperiodic(xyz, Lbox=2600., Ngrid=360, fft='pyfftw', silent=True): 
    ''' Put galaxies in a grid and FFT it. This function wraps some of
    the functions in estimator.f and does the same thing as roman's 
    zmapFFTil4_aniso_gen.f 

    :param xyz: 
        3 x N dimensional array

    :param Lbox: (default: 2600.) 
        Box size

    :param Ngrid: (default: 360) 
        grid size 

    :param fft: (default: 'pyfftw') 
        determines which fftw version to use. Options are 'pyfftw' 
        and 'fortran'. 

    :param silent: (default: True) 
        if not silent then it'll output stuff  
    '''
    kf_ks = np.float32(float(Ngrid) / Lbox)
    N = np.int32(xyz.shape[1]) # number of objects 

    # position of galaxies (checked with fortran) 
    xyzs = np.zeros([3, N], dtype=np.float32, order='F') 
    xyzs[0,:] = np.clip(xyz[0,:], 0., Lbox*(1.-1e-6))
    xyzs[1,:] = np.clip(xyz[1,:], 0., Lbox*(1.-1e-6))
    xyzs[2,:] = np.clip(xyz[2,:], 0., Lbox*(1.-1e-6))
    if not silent: print('%i positions' % N) 
    
    # assign galaxies to grid (checked with fortran) 
    _delta = np.zeros([2*Ngrid, Ngrid, Ngrid], dtype=np.float32, order='F') # even indices (real) odd (complex)
    fEstimate.assign2(xyzs, _delta, kf_ks, N, Ngrid) 
    
    if fft == 'pyfftw': 
        delta = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')
    elif fft == 'fortran': 
        delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F')
    delta.real = _delta[::2,:,:] 
    delta.imag = _delta[1::2,:,:] 
    if not silent: print('positions assigned to grid') 

    # FFT delta (checked with fortran code, more or less matches)
    if fft == 'pyfftw': 
        fftw_ob = pyfftw.builders.ifftn(delta, planner_effort='FFTW_ESTIMATE') # axes=(0,1,2,))
        #ifft_delta = fftw_ob(normalise_idft=False)
        ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F') 
        ifft_delta[:,:,:] = fftw_ob(normalise_idft=False)
    elif fft == 'fortran': 
        ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.complex64, order='F') 
        fEstimate.ffting(delta, N, Ngrid) 
        ifft_delta[:,:,:] = delta[:,:,:]

    if not silent: print('position grid FFTed') 
    # combine fields 
    fEstimate.fcomb(ifft_delta,N,Ngrid) 
    if not silent: print('fcomb complete') 
    return ifft_delta[:Ngrid//2+1,:,:]


def Bk123_periodic(delta, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=True): 
    ''' Calculate the bispectrum for periodic box given delta(k) 3D field.
    i,j,l are in units of k_fundamental (2pi/Lbox) 
    b123 is in units of 1/kf^3/(2pi)^3

    # add documentation here!
    # add documentation here!
    # add documentation here!
    # add documentation here!
    # add documentation here!
    '''
    Ngrid = delta.shape[0]
    
    # FFT convention: array of |kx| values #
    a = np.array([min(i,Ngrid-i) for i in range(Ngrid)])

    # FFT convention: rank three field of |r| values #
    rk = ((np.sqrt(a[:,None,None]**2 + a[None,:,None]**2 + a[None,None,:]**2)))
    
    irk = (rk/step+0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT

    Nk = np.array([np.sum(irk == i) for i in np.arange(Nmax+1)])#grid)])#Nmax - Ncut/step+2)])

    if not silent: print("--- calculating delta(k) shells ---") 
    
    deltaKshellX = np.zeros((Nmax+1, Ngrid, Ngrid, Ngrid),dtype=float) #default double prec
    p0k = np.zeros(Nmax)

    for j in range(Ncut // step, Nmax + 1):
        if fft == 'pyfftw':
            tempK = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')
            tempK[:] = 0.
        else: 
            tempK = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.complex64)
        tempK[irk == j] = delta[irk == j] 

        if fft == 'pyfftw': 
            if j == (Ncut // step): 
                fftw_ob = pyfftw.builders.fftn(tempK, planner_effort='FFTW_ESTIMATE', threads=nthreads)
                pyfftw.interfaces.cache.enable()
            fft_tempK = fftw_ob(tempK)
            deltaKshellX[j] = np.real(fft_tempK)
        elif fft == 'numpy':
            deltaKshellX[j] = np.fft.fftn(tempK)
        
        p0k[j-1] = np.einsum('i,i', deltaKshellX[j].ravel(), deltaKshellX[j].ravel())/Ngrid**3/Nk[j] # 10 ms

    # counts for normalizing  
    counts = _counts_Bk123(Ngrid=Ngrid, Nmax=Nmax, Ncut=Ncut, step=step, fft=fft, silent=silent) 

    if not silent: print("--- summing over k1,k2,k3 configurations ---") 
    # ch: undoing the for-loops does not speed up anything
    # and unless very clever increases the memory footprint.
    i_arr, j_arr, l_arr = [], [], []
    p0k_i, p0k_j, p0k_l = [], [], [] 
    b123_arr, q123_arr, cnts_arr = [], [], [] 
    for i in range(Ncut//step, Nmax+1): 
        for j in range(Ncut//step, i+1):
            for l in range(max(i-j, Ncut//step), j+1):
                fac = 1. 
                if (j == l) and (i == j): fac=6.
                if (i == j) and (j != l): fac=2.
                if (i == l) and (l != j): fac=2.
                if (j == l) and (l != i): fac=2.
                if counts[i-1,j-1,l-1] > 0: 
                    i_arr.append(i) 
                    j_arr.append(j) 
                    l_arr.append(l) 
                    bisp_ijl = np.einsum('i,i,i', 
                            deltaKshellX[i].ravel(), 
                            deltaKshellX[j].ravel(), 
                            deltaKshellX[l].ravel())
                    p0k_i.append(p0k[i-1])
                    p0k_j.append(p0k[j-1])
                    p0k_l.append(p0k[l-1])

                    b123_arr.append(bisp_ijl/counts[i-1,j-1,l-1]) 
                    q123_arr.append(bisp_ijl/counts[i-1,j-1,l-1]/(p0k[i-1]*p0k[j-1] + p0k[j-1]*p0k[l-1] + p0k[l-1]*p0k[i-1]))
                    cnts_arr.append(counts[i-1,j-1,l-1]/(fac*float(Ngrid**3)))
                else: 
                    p0k_i.append(0.)
                    p0k_j.append(0.)
                    p0k_l.append(0.)

                    b123_arr.append(0.) 
                    q123_arr.append(0.) 
                    cnts_arr.append(0.) 
    
    output = {} 
    output['i_k1'] = np.array(i_arr) * step 
    output['i_k2'] = np.array(j_arr) * step 
    output['i_k3'] = np.array(l_arr) * step 
    output['p0k1'] = np.array(p0k_i)
    output['p0k2'] = np.array(p0k_j)
    output['p0k3'] = np.array(p0k_l)
    output['b123'] = np.array(b123_arr)
    output['q123'] = np.array(q123_arr) 
    output['counts'] = np.array(cnts_arr)
    return output 


def _counts_Bk123(Ngrid=360, Nmax=40, Ncut=3, step=3, fft='pyfftw', silent=True): 
    ''' return bispectrum normalization 
    @chh explain nmax, ncut, and step below 
    '''
    fcnt = ''.join(['counts', '.Ngrid', str(Ngrid), '.Nmax', str(Nmax), '.Ncut', str(Ncut), '.step', str(step), '.', fft]) 
    f_counts = os.path.join(dat_dir(), fcnt) 

    if os.path.isfile(f_counts): 
        f = FortranFile(f_counts, 'r') 
        cnts = f.read_reals() 
        counts = np.reshape(cnts, (Nmax, Nmax, Nmax))
        f.close() 
        #counts = pickle.load(open(f_counts), 'rb') 
    else: 
        if not silent: print("--- calculating %s ---" % f_counts) 
        delta = np.ones((Ngrid, Ngrid, Ngrid))
        
        # FFT convention: array of |kx| values #
        a = np.array([min(i,Ngrid-i) for i in range(Ngrid)])

        # FFT convention: rank three field of |r| values #
        rk = ((np.sqrt(a[:,None,None]**2 + a[None,:,None]**2 + a[None,None,:]**2)))
        
        irk = (rk/step+0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT

        Nk = np.array([np.sum(irk == i) for i in np.arange(Nmax+1)])#grid)])#Nmax - Ncut/step+2)])

        if not silent: print("--- calculating delta(k) shells ---") 
        deltaKshellK = np.zeros((irk.max()+1, Ngrid, Ngrid, Ngrid), dtype=complex)
        for i in range(Ngrid):
            for j in range(Ngrid):
                for l in range(Ngrid):
                    ak = irk[i,j,l]  # binning operation
                    #if (ak <= Nmax/step):
                    #if (ak <= Nmax):
                    deltaKshellK[ak,i,j,l] = delta[i,j,l]
        
        if fft == 'pyfftw':
            tempK = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')

        deltaKshellX = np.zeros((Nmax+1, Ngrid, Ngrid, Ngrid),dtype=float) #default double prec
        for j in range(Ncut // step, Nmax + 1):
            tempK = deltaKshellK[j,:,:,:]
        
            if fft == 'pyfftw': 
                if j == (Ncut // step): 
                    fftw_ob = pyfftw.builders.fftn(tempK, planner_effort='FFTW_ESTIMATE')
                    pyfftw.interfaces.cache.enable()
                fft_tempK = fftw_ob(tempK)
                deltaKshellX[j] = np.real(fft_tempK)
            elif fft == 'numpy':
                deltaKshellX[j] = np.fft.fftn(tempK)
        
        if not silent: print("--- calculating counts ---") 
        counts = np.zeros((Nmax, Nmax, Nmax), dtype=float) #default double prec
        for i in range(Ncut//step, Nmax+1): 
            for j in range(Ncut//step, i+1):
                for l in range(max(i-j, Ncut//step), j+1):
                    counts[i-1,j-1,l-1] = np.einsum('i,i,i', 
                            deltaKshellX[i].ravel(), 
                            deltaKshellX[j].ravel(), 
                            deltaKshellX[l].ravel())

        # save to file  
        f = FortranFile(f_counts, 'w') 
        f.write_record(counts) # double prec 
        f.close() 
        #pickle.dump(counts, open(f_counts, 'wb'))
    return counts 


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


def delta_quadrupole(delt, Ngrid=360, rsd=None, silent=True): 
    ''' given half field build quadrupole (what's done in inputNB) 
    '''
    if rsd is None: raise ValueError("choose rsd direction 'x', 'y', or 'z'") 
    irsd = {'x': 1, 'y': 2, 'z': 3} 
    i_rsd = irsd[rsd]
    if not silent: print('building quadrupole') 
    _delt1 = np.zeros((Ngrid//2+1, Ngrid, Ngrid), dtype=np.complex64, order='F') 
    _delt2 = np.zeros((Ngrid//2+1, Ngrid, Ngrid), dtype=np.complex64, order='F') 
    _delt1[:,:,:] = delt[:,:,:]

    fEstimate.build_quad(_delt1, _delt2, i_rsd, Ngrid) 
    return reflect_delta(_delt2, Ngrid=Ngrid, silent=silent) 


def reflect_delta(delt, Ngrid=360, silent=True): 
    ''' reflect half field that's output from the code 
    '''
    if not silent: print('reflecting the half field') 
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


def _Pk_periodic(delta, Lbox=None):
    ''' calculate the powerspecturm for periodic box given 3d fourier density grid, delta(k). 
    output k is in units of k_fundamental 
    '''
    Ngrid = delta.shape[0]
    Nbins = Ngrid // 2 
    if Lbox is None: 
        kf = 1. # in units of fundamental mode 
    else: 
        kf = 2*np.pi / float(Lbox) 
    phys_nyq = kf * float(Ngrid) / 2. # physical Nyquist

    # FFT convention: array of |kx| values #
    _i = np.array([min(i,Ngrid-i) for i in range(Ngrid)])
    # FFT convention: rank three field of |r| values #
    rk = kf * np.sqrt(_i[:,None,None]**2 + _i[None,:,None]**2 + _i[None,None,:]**2)
    irk = (Nbins * rk / phys_nyq + 0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT
    
    ks = np.zeros(Nbins) 
    p0k = np.zeros(Nbins) 
    nks = np.zeros(Nbins) 
    for i in np.arange(1, Nbins+1): 
        inkbin = (irk == i) 
        Nk = np.sum(inkbin) 
        if Nk > 0: 
            ks[i-1] = np.sum(rk[inkbin])/float(Nk)
            p0k[i-1] = np.sum(np.absolute(delta[inkbin])**2)/float(Nk)/kf**3
            nks[i-1] = float(Nk) 

    return ks, (2.*np.pi)**3 * p0k, nks


def _Pk_periodic_rsd(delta, Lbox=None, rsd=2, Nmubin=5, code='fortran'):
    ''' calculate the powerspecturm for periodic box given 3d fourier density grid, delta(k). 
    output k is in units of k_fundamental 
    '''
    if code == 'python': 
        Ngrid = delta.shape[0]
        Nbins = int(Ngrid / 2) 

        dmu = 1./float(Nmubin)
        
        # fundamental model 
        if Lbox is None: 
            kf = 1. # in units of fundamental mode 
        else: 
            kf = 2*np.pi / float(Lbox) 
        phys_nyq = kf * float(Ngrid) / 2. # physical Nyquist

        # FFT convention: array of |kx| values #
        _i = np.array([min(i,Ngrid-i) for i in range(Ngrid)])
        # FFT convention: rank three field of |r| values #
        rk = kf * np.sqrt(_i[:,None,None]**2 + _i[None,:,None]**2 + _i[None,None,:]**2)
        irk = (Nbins * rk / phys_nyq + 0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT

        # theta and phi of observer
        theta_obs = [0.5*np.pi, 0.5*np.pi, 0.][rsd]
        phi_obs = [0., 0.5*np.pi, 0.][rsd]

        rkx = kf * _i[:,None,None]
        rky = kf * _i[None,:,None]
        rkz = kf * _i[None,None,:]

        cos_theta = rkz / rk 
        sin_theta = np.sqrt(1. - cos_theta**2)  

        cc = np.zeros((Ngrid, Ngrid, Ngrid))
        cos_phi = rkx / (rk * sin_theta) 
        sin_phi = rky / (rk * sin_theta) 
        cc[sin_theta > 0.] = np.sin(phi_obs) * sin_phi[sin_theta > 0.] + np.cos(phi_obs) * cos_phi[sin_theta > 0.]
        mu = np.cos(theta_obs) * cos_theta + np.sin(theta_obs) * sin_theta * cc 
        imu = (np.ceil(mu / dmu)).astype(int)
        imu[0,0,0] = -999 

        Leg2 = -0.5 + 1.5 * mu**2
        Leg4 = 0.375 - 3.75 * mu**2 + 4.375 * mu**4

        ks      = np.zeros(Nbins) # k 
        p0k     = np.zeros(Nbins) # monopole
        p2k     = np.zeros(Nbins) # quadrupole 
        p4k     = np.zeros(Nbins) # hexadecapole 
        nks     = np.zeros(Nbins) # number of modes 
        
        N_kmu   = np.zeros((Nbins, Nmubin)) # number of modes in (k, mu) bin 
        k_kmu   = np.zeros((Nbins, Nmubin)) # average k of modes in (k, mu) bin 
        mu_kmu  = np.zeros((Nbins, Nmubin)) # average mu of modes in (k, mu) bin
        p_kmu   = np.zeros((Nbins, Nmubin)) # p(k, mu) 

        for i in np.arange(1, Nbins+1): 
            inkbin = (irk == i) 
            Nk = np.sum(inkbin) 

            if Nk > 0: 
                ks[i-1]     = np.sum(rk[inkbin])/float(Nk)
                p0k[i-1]    = np.sum(np.absolute(delta[inkbin])**2)/float(Nk)/kf**3
                nks[i-1]    = float(Nk) 

                for j in np.arange(1, Nmubin+1): 
                    print(i,j)
                    inkmubin = inkbin & (imu == j)
                    Nkmu = np.sum(inkmubin) 
                    if Nkmu > 0: 
                        # quadrupole and hexadecapole contributions
                        p2k[i-1] += np.sum(np.absolute(delta[inkmubin])**2 * Leg2[inkmubin])/float(Nk)/kf**3 * 5. 
                        p4k[i-1] += np.sum(np.absolute(delta[inkmubin])**2 * Leg4[inkmubin])/float(Nk)/kf**3 * 9. 

                        k_kmu[i-1, j-1]     = np.sum(rk[inkmubin])/float(Nkmu)
                        mu_kmu[i-1, j-1]    = np.sum(mu[inkmubin])/float(Nkmu)
                        p_kmu[i-1, j-1]     = np.sum(np.absolute(delta[inkmubin])**2)/float(Nkmu)/kf**3
                        N_kmu[i-1, j-1]     = float(Nkmu) 
        
        pk_norm = (2.*np.pi)**3
        p0k *= pk_norm
        p2k *= pk_norm
        p4k *= pk_norm
        p_kmu *= pk_norm
        
        return ks, p0k, p2k, p4k, nks, k_kmu, mu_kmu, p_kmu, N_kmu 

    elif code == 'fortran': 
        Ngrid = delta.shape[1]
        Nbins = (Ngrid // 2) 

        dtl = np.zeros((Ngrid//2+1, Ngrid, Ngrid), dtype=np.complex64, order='F') 
        dtl[:,:,:] = delta[:,:,:]
        ks, p0k, p2k, p4k, nk, k_kmu, mu_kmu, p_kmu, n_kmu = fEstimate.pk_pbox_rsd(dtl, rsd, Lbox, Nbins, Nmubin, Ngrid)

        pk_norm = (2.*np.pi)**3
        p0k *= pk_norm
        p2k *= pk_norm
        p4k *= pk_norm
        p_kmu *= pk_norm
        return ks, p0k, p2k, p4k, nk, k_kmu, mu_kmu, p_kmu, n_kmu


def Pk_periodic_f77(delta, Lbox=None):
    ''' calculate the powerspecturm for periodic box given 3d fourier density grid. 
    output k is in units of k_fundamental 
    '''
    Ngrid = delta.shape[1]
    Nbins = Ngrid / 2 
    # allocate arrays
    dtl = np.zeros((Ngrid//2+1, Ngrid, Ngrid), dtype=np.complex64, order='F') 
    dtl[:,:,:] = delta[:,:,:]
    ks = np.zeros(Nbins, dtype=np.float64, order='F') 
    p0k = np.zeros(Nbins, dtype=np.float64, order='F') 
    fEstimate.pk_periodic(dtl,ks,p0k,Ngrid,Nbins)
    return ks, (2.*np.pi)**3 * p0k 


def _counts_Bk123_f77(Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True): 
    ''' return bispectrum normalization 
    @chh explain nmax, ncut, and step below 
    '''
    fcnt = ''.join(['counts', '.Ngrid', str(Ngrid), '.Nmax', str(Nmax), '.Ncut', str(Ncut), '.step', str(step), '.fort77']) 
    f_counts = os.path.join(dat_dir(), fcnt) 

    if os.path.isfile(f_counts): 
        f = FortranFile(f_counts, 'r') 
        cnts = f.read_reals(dtype=np.float64) 
        counts = np.reshape(cnts, (Nmax, Nmax, Nmax), order='F')
        f.close() 
    else: 
        if not silent: 
            print('-- %s does not exist --' % f_counts) 
            print('-- computing %s --' % f_counts) 
        counts = np.zeros((40,40,40), dtype=np.float64, order='F')  
        fEstimate.bk_counts(counts,Ngrid,float(step),Ncut,Nmax) 

        # save to file  
        f = FortranFile(f_counts, 'w') 
        f.write_record(counts) # double prec 
        f.close() 
    return counts 


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


def _pk_Bk123_periodic(delta, Nmax=40, Ncut=3, step=3, fft='pyfftw', nthreads=1, silent=True): 
    ''' this is a hacked thing to get powerspectrum 
    '''
    Ngrid = delta.shape[0]
    
    # FFT convention: array of |kx| values #
    a = np.array([min(i,Ngrid-i) for i in range(Ngrid)])

    # FFT convention: rank three field of |r| values #
    rk = ((np.sqrt(a[:,None,None]**2 + a[None,:,None]**2 + a[None,None,:]**2)))
    
    irk = (rk/step+0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT

    if not silent: print("--- calculating delta(k) shells ---") 
    
    p0k = np.zeros(Nmax)
    for j in range(Ncut // step, Nmax + 1):
        inkbin = (irk == j) 
        Nk = np.sum(inkbin) 
        p0k[j-1] = np.sum(np.absolute(delta[inkbin])**2)/float(Nk)
    
    # counts for normalizing  
    counts = _counts_Bk123(Ngrid=Ngrid, Nmax=Nmax, Ncut=Ncut, step=step, fft=fft, silent=silent) 

    if not silent: print("--- summing over k1,k2,k3 configurations ---") 
    # ch: undoing the for-loops does not speed up anything
    # and unless very clever increases the memory footprint.
    p0k_i, p0k_j, p0k_l = [], [], [] 
    for i in range(Ncut//step, Nmax+1): 
        for j in range(Ncut//step, i+1):
            for l in range(max(i-j, Ncut//step), j+1):
                if counts[i-1,j-1,l-1] > 0: 
                    p0k_i.append(p0k[i-1])
                    p0k_j.append(p0k[j-1])
                    p0k_l.append(p0k[l-1])
                else: 
                    p0k_i.append(0.)
                    p0k_j.append(0.)
                    p0k_l.append(0.)

    return np.array(p0k_i), np.array(p0k_j), np.array(p0k_l)
