import os 
import time
import pyfftw
import numpy as np 
from scipy.io import FortranFile
from astropy.cosmology import FlatLambdaCDM
# -- local -- 
import estimator as fEstimate
from . import util as UT


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


def FFTperiodic(gals, Lbox=2600., Ngrid=360, fft='pyfftw', silent=True): 
    ''' Put galaxies in a grid and FFT it. This function wraps some of
    the functions in estimator.f and does the same thing as roman's 
    zmapFFTil4_aniso_gen.f 
    '''
    kf_ks = np.float32(float(Ngrid) / Lbox)
    Ng = np.int32(gals.shape[1]) # number of galaxies

    # position of galaxies (checked with fortran) 
    xyz_gals = np.zeros([3, Ng], dtype=np.float32, order='F') 
    xyz_gals[0,:] = np.clip(gals[0,:], 0., Lbox*(1.-1e-6))
    xyz_gals[1,:] = np.clip(gals[1,:], 0., Lbox*(1.-1e-6))
    xyz_gals[2,:] = np.clip(gals[2,:], 0., Lbox*(1.-1e-6))
    if not silent: print('%i galaxy positions saved' % Ng) 
    
    # assign galaxies to grid (checked with fortran) 
    _delta = np.zeros([2*Ngrid, Ngrid, Ngrid], dtype=np.float32, order='F') # even indices (real) odd (complex)
    fEstimate.assign2(xyz_gals, _delta, kf_ks, Ng, Ngrid) 
    
    if fft == 'pyfftw': 
        delta = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')
    #elif fft == 'fftw3': 
    #    delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex128')
    elif fft == 'fortran': 
        delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F')
    delta.real = _delta[::2,:,:] 
    delta.imag = _delta[1::2,:,:] 
    if not silent: print('galaxy positions assigned to grid') 

    # FFT delta (checked with fortran code, more or less matches)
    if fft == 'pyfftw': 
        fftw_ob = pyfftw.builders.ifftn(delta, planner_effort='FFTW_ESTIMATE') # axes=(0,1,2,))
        #ifft_delta = fftw_ob(normalise_idft=False)
        ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex64', order='F') 
        ifft_delta[:,:,:] = fftw_ob(normalise_idft=False)
    #elif fft == 'fftw3': 
    #    _ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype='complex128')#, order='F') 
    #    ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.complex64, order='F') 
    #    fftplan = fftw3.Plan(delta, _ifft_delta, direction='backward', flags=['estimate'])
    #    fftplan.execute() 
    #    ifft_delta[:,:,:] = _ifft_delta[:,:,:]
    elif fft == 'fortran': 
        ifft_delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.complex64, order='F') 
        fEstimate.ffting(delta, Ng, Ngrid) 
        ifft_delta[:,:,:] = delta[:,:,:]

    if not silent: print('galaxy grid FFTed') 
    # fcombine 
    fEstimate.fcomb(ifft_delta,Ng,Ngrid) 
    if not silent: print('fcomb complete') 
    return ifft_delta[:Ngrid/2+1,:,:]


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


def Pk_periodic(delta, Lbox=None):
    ''' calculate the powerspecturm for periodic box given 3d fourier density grid. 
    output k is in units of k_fundamental 
    '''
    Ngrid = delta.shape[0]
    Nbins = Ngrid / 2 
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


def Bk123_periodic(delta, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', nthreads=1, silent=True): 
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
        if fft_method == 'pyfftw':
            tempK = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')
        else: 
            tempK = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.complex64)
        tempK[irk == j] = delta[irk == j] 

        if fft_method == 'pyfftw': 
            if j == (Ncut // step): 
                fftw_ob = pyfftw.builders.fftn(tempK, planner_effort='FFTW_ESTIMATE', threads=nthreads)
                pyfftw.interfaces.cache.enable()
            fft_tempK = fftw_ob(tempK)
            deltaKshellX[j] = np.real(fft_tempK)
        elif fft_method == 'numpy':
            deltaKshellX[j] = np.fft.fftn(tempK)
        
        p0k[j-1] = np.einsum('i,i', deltaKshellX[j].ravel(), deltaKshellX[j].ravel())/Ngrid**3/Nk[j] # 10 ms

    # counts for normalizing  
    counts = _counts_Bk123(Ngrid=Ngrid, Nmax=Nmax, Ncut=Ncut, step=step, fft_method=fft_method, silent=silent) 
    #counts[counts == 0] = np.inf

    if not silent: print("--- calculating B(k1,k2,k3) ---") 
    #bisp = np.zeros((Nmax, Nmax, Nmax), dtype=float) #default double prec
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
                    #bisp[i-1,j-1,l-1] = np.einsum('i,i,i', 
                    # deltaKshellX[i].ravel(), deltaKshellX[j].ravel(), deltaKshellX[l].ravel())
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


def _Bk123_periodic_old(delta, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', nthreads=1, bit=32, silent=True): 
    ''' Calculate the bispectrum for periodic box given delta(k) 3D field.
    i,j,l are in units of k_fundamental (2pi/Lbox) 
    b123 is in units of 1/kf^3/(2pi)^3
    '''
    Ngrid = delta.shape[0]
    
    # FFT convention: array of |kx| values #
    a = np.array([min(i,Ngrid-i) for i in range(Ngrid)])

    # FFT convention: rank three field of |r| values #
    rk = ((np.sqrt(a[:,None,None]**2 + a[None,:,None]**2 + a[None,None,:]**2)))
    
    irk = (rk/step+0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT

    #Nk = np.array([len(np.where(irk == i)[0]) for i in np.arange(Nmax - Ncut/step+2)])
    #Nk = np.array([len(np.where(irk == i)[0]) for i in np.arange(2*Ngrid)])#Nmax - Ncut/step+2)])
    Nk = np.array([np.sum(irk == i) for i in np.arange(Nmax+1)])#grid)])#Nmax - Ncut/step+2)])

    if not silent: print("--- calculating delta(k) shells ---") 
    if bit == 32: 
        deltaKshellK = np.zeros((irk.max()+1, Ngrid, Ngrid, Ngrid), dtype=np.complex64)
    elif bit == 64: 
        deltaKshellK = np.zeros((irk.max()+1, Ngrid, Ngrid, Ngrid), dtype=complex)
    for i in range(Ngrid):
        for j in range(Ngrid):
            for l in range(Ngrid):
                ak = irk[i,j,l]  # binning operation
                #if (ak <= Nmax/step):
                #if (ak <= Nmax):
                deltaKshellK[ak,i,j,l] = delta[i,j,l]
    
    if fft_method == 'pyfftw':
        tempK = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')

    if not silent: print("--- calculating delta(X) shells and P(k) ---") 
    #deltaKshellX = np.zeros((Nmax//step+1, Ngrid, Ngrid, Ngrid),dtype=float) #default double prec
    deltaKshellX = np.zeros((Nmax+1, Ngrid, Ngrid, Ngrid),dtype=float) #default double prec
    p0k = np.zeros(Nmax)
    for j in range(Ncut // step, Nmax + 1):
        tempK = deltaKshellK[j,:,:,:]
    
        if fft_method == 'pyfftw': 
            if j == (Ncut // step): 
                fftw_ob = pyfftw.builders.fftn(tempK, planner_effort='FFTW_ESTIMATE', threads=nthreads)
                pyfftw.interfaces.cache.enable()
            fft_tempK = fftw_ob(tempK)
            deltaKshellX[j] = np.real(fft_tempK)
        elif fft_method == 'numpy':
            deltaKshellX[j] = np.fft.fftn(tempK)
        
        p0k[j-1] = np.einsum('i,i', deltaKshellX[j].ravel(), deltaKshellX[j].ravel())/Ngrid**3/Nk[j] # 10 ms
    
    # counts for normalizing  
    counts = _counts_Bk123(Ngrid=Ngrid, Nmax=Nmax, Ncut=Ncut, step=step, fft_method=fft_method, silent=silent) 
    #counts[counts == 0] = np.inf

    if not silent: print("--- calculating B(k1,k2,k3) ---") 
    #bisp = np.zeros((Nmax, Nmax, Nmax), dtype=float) #default double prec
    i_arr, j_arr, l_arr = [], [], []
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
                    #bisp[i-1,j-1,l-1] = np.einsum('i,i,i', 
                    # deltaKshellX[i].ravel(), deltaKshellX[j].ravel(), deltaKshellX[l].ravel())
                    bisp_ijl = np.einsum('i,i,i', 
                            deltaKshellX[i].ravel(), 
                            deltaKshellX[j].ravel(), 
                            deltaKshellX[l].ravel())
                    b123_arr.append(bisp_ijl/counts[i-1,j-1,l-1]) 
                    q123_arr.append(bisp_ijl/counts[i-1,j-1,l-1]/(p0k[i-1]*p0k[j-1] + p0k[j-1]*p0k[l-1] + p0k[l-1]*p0k[i-1]))
                    cnts_arr.append(counts[i-1,j-1,l-1]/(fac*float(Ngrid**3)))
                else: 
                    b123_arr.append(0.) 
                    q123_arr.append(0.) 
                    cnts_arr.append(0.) 

    i_arr = np.array(i_arr) * step 
    j_arr = np.array(j_arr) * step 
    l_arr = np.array(l_arr) * step 
    b123_arr = np.array(b123_arr)
    q123_arr = np.array(q123_arr) 
    cnts_arr = np.array(cnts_arr)
    return i_arr, j_arr, l_arr, b123_arr, q123_arr, cnts_arr 


def _counts_Bk123(Ngrid=360, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=True): 
    ''' return bispectrum normalization 
    @chh explain nmax, ncut, and step below 
    '''
    f_counts = ''.join([UT.code_dir(), 'dat/', 
        'counts', 
        '.Ngrid', str(Ngrid),
        '.Nmax', str(Nmax),
        '.Ncut', str(Ncut),
        '.step', str(step),
        '.', fft_method]) 

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
        
        if fft_method == 'pyfftw':
            tempK = pyfftw.n_byte_align_empty((Ngrid, Ngrid, Ngrid), 16, dtype='complex64')

        deltaKshellX = np.zeros((Nmax+1, Ngrid, Ngrid, Ngrid),dtype=float) #default double prec
        for j in range(Ncut // step, Nmax + 1):
            tempK = deltaKshellK[j,:,:,:]
        
            if fft_method == 'pyfftw': 
                if j == (Ncut // step): 
                    fftw_ob = pyfftw.builders.fftn(tempK, planner_effort='FFTW_ESTIMATE')
                    pyfftw.interfaces.cache.enable()
                fft_tempK = fftw_ob(tempK)
                deltaKshellX[j] = np.real(fft_tempK)
            elif fft_method == 'numpy':
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


def Bk123_periodic_f77(delta, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=True): 
    ''' Calculate the bispectrum for periodic box given delta(k) 3D field.
    '''
    raise NotImplementedError


def _counts_Bk123_f77(Ngrid=360, Nmax=40, Ncut=3, step=3, silent=True): 
    ''' return bispectrum normalization 
    @chh explain nmax, ncut, and step below 
    '''
    f_counts = ''.join([UT.code_dir(), 'dat/', 
        'counts', 
        '.Ngrid', str(Ngrid),
        '.Nmax', str(Nmax),
        '.Ncut', str(Ncut),
        '.step', str(step),
        '.fort77']) 

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
