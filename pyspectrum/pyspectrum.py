import os 
import pyfftw
import numpy as np 
from scipy.io import FortranFile
# -- local -- 
from . import util as UT


def read_fortFFT(file='/Users/ChangHoon/data/pyspectrum/FFT_Q_CutskyN1.fidcosmo.dat.grid360.P020000.box3600'): 
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
    for i in np.arange(1, Nbins+1): 
        inkbin = (irk == i) 
        Nk = np.sum(inkbin) 
        if Nk > 0: 
            ks[i-1] = np.sum(rk[inkbin])/float(Nk)
            p0k[i-1] = np.sum(np.absolute(delta[inkbin])**2)/float(Nk)/kf**3
    return ks, (2.*np.pi)**3 * p0k 


def Bk123_periodic(delta, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=True): 
    ''' Calculate the bispectrum for periodic box given delta(k) 3D field.
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
                fftw_ob = pyfftw.builders.fftn(tempK, planner_effort='FFTW_ESTIMATE')
                pyfftw.interfaces.cache.enable()
            fft_tempK = fftw_ob(tempK)
            deltaKshellX[j] = np.real(fft_tempK)
        elif fft_method == 'numpy':
            deltaKshellX[j] = np.fft.fftn(tempK)
        
        p0k[j-1] = np.einsum('i,i', deltaKshellX[j].ravel(), deltaKshellX[j].ravel())/Ngrid**3/Nk[j] # 10 ms
    
    # counts for normalizing  
    counts = _counts_Bk123(Ngrid=Ngrid, Nmax=Nmax, Ncut=Ncut, step=step, fft_method=fft_method, silent=silent) 
    counts[counts == 0] = np.inf

    if not silent: print("--- calculating B(k1,k2,k3) ---") 
    #bisp = np.zeros((Nmax//step+1, Nmax//step+1, Nmax//step+1), dtype=float) #default double prec
    #for i in range(Ncut // step, Nmax // step+1): 
    i_arr, j_arr, l_arr = [], [], []
    bisp_arr = [] 
    #bisp = np.zeros((Nmax, Nmax, Nmax), dtype=float) #default double prec
    for i in range(Ncut//step, Nmax+1): 
        for j in range(Ncut//step, i+1):
            for l in range(max(i-j, Ncut//step), j+1):
                i_arr.append(i) 
                j_arr.append(j) 
                l_arr.append(l) 
                #bisp[i-1,j-1,l-1] = np.einsum('i,i,i', deltaKshellX[i].ravel(), deltaKshellX[j].ravel(), deltaKshellX[l].ravel())
                bisp_ijl = np.einsum('i,i,i', deltaKshellX[i].ravel(), deltaKshellX[j].ravel(), deltaKshellX[l].ravel())
                bisp_arr.append(bisp_ijl/counts[i-1,j-1,l-1]) 
    
    i_arr = np.array(i_arr) * step 
    j_arr = np.array(j_arr) * step 
    l_arr = np.array(l_arr) * step 
    bisp_arr = np.array(bisp_arr)
    return i_arr, j_arr, l_arr, bisp_arr


def _counts_Bk123(Ngrid=360, Nmax=40, Ncut=3, step=3, fft_method='pyfftw', silent=True): 
    ''' return bispectrum normalization 
    @chh explain nmax, ncut, and step below 
    '''
    f_counts = ''.join([UT.dat_dir(), 'counts', 
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
