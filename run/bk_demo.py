'''
make demo video of bispectrum
'''
import os 
import h5py
import pickle
import numpy as np 
# -- pyspectrum -- 
from pyspectrum import util as UT 
from pyspectrum import pyspectrum as pySpec
# -- plotting -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def egBk(): 
    ''' calculate example Bk
    '''
    Lbox = 2600. # box size
    Ngrid = 360  # fft grid size

    fnbox = h5py.File(os.path.join(UT.dat_dir(), 'BoxN1.hdf5'), 'r') 
    xyz = fnbox['xyz'].value 
    vxyz = fnbox['vxyz'].value 
    xyz_s = pySpec.applyRSD(xyz.T, vxyz.T, 0.5, h=0.7, omega0_m=0.3, LOS='z', Lbox=2600.) 
    bisp = pySpec.Bk_periodic(xyz_s, Lbox=Lbox, Ngrid=Ngrid, step=3, Ncut=3, Nmax=40, fft='pyfftw') 
    pickle.dump(bisp, open(os.path.join(UT.dat_dir(), 'egBk.p'), 'wb'))
    return None 


def demoBk(): 
    '''
    '''
    kmax = 0.5
    kf = 2.*np.pi/2600.
    # read in Bk 
    bisp = pickle.load(open(os.path.join(UT.dat_dir(), 'egBk.p'), 'rb'))
    i_k, j_k, l_k, bk = bisp['i_k1'], bisp['i_k2'], bisp['i_k3'], bisp['b123'] 
    
    klim = ((i_k*kf <= kmax) & (j_k*kf <= kmax) & (l_k*kf <= kmax))
    i_k, j_k, l_k = i_k[klim], j_k[klim], l_k[klim]
    ijl = UT.ijl_order(i_k, j_k, l_k, typ='GM') # order of triangles 
    
    for itri in range(1500): 
        fig = plt.figure(figsize=(15,4.8))
        sub = fig.add_subplot(111)
        sub.plot(range(1500), bk[klim][ijl][:1500], c='k', lw=0.5)
        sub.scatter(range(1500), bk[klim][ijl][:1500], c='k', s=2)
        sub.scatter(itri, bk[klim][ijl][itri], c='C1', s=30, zorder=10) 

        itri_i, itri_j, itri_l = i_k[ijl][itri], j_k[ijl][itri], l_k[ijl][itri]
        trixy = np.zeros((3,2))
        trixy[0,:] = np.array([1400, 2e10])
        trixy[1,:] = np.array([1400 - 3.33*itri_i, 2e10]) 
        theta23 = np.arccos(-0.5*(itri_l**2 - itri_i**2 - itri_j**2)/itri_i/itri_j)[0]
        trixy[2,:] = np.array([1400-3.33*itri_j[0]*np.cos(theta23), 
            10**(10.301 - 0.022*itri_j[0]*np.sin(theta23))])
        tri = plt.Polygon(trixy, fill=None, edgecolor='k')
        fig.gca().add_patch(tri)
        
        sub.text(0.5*(trixy[0,0]+trixy[1,0]), 0.5*(trixy[0,1]+trixy[1,1]), '$k_3$', 
                fontsize=8, ha='center', va='bottom') 
        sub.text(0.5*(trixy[0,0]+trixy[2,0]), 0.5*(trixy[0,1]+trixy[2,1]), '$k_2$', 
                fontsize=8, ha='left', va='top') 
        sub.text(0.5*(trixy[1,0]+trixy[2,0]), 0.5*(trixy[1,1]+trixy[2,1]), '$k_1$', 
                fontsize=8, ha='right', va='top') 

        sub.set_xlabel('triangle configurations', fontsize=25)
        sub.set_xlim([0, 1500])
        sub.set_ylabel(r'$B(k_1, k_2, k_3)$', fontsize=25)
        sub.set_yscale('log') 
        sub.set_ylim([1e7, 3e10])
        fig.savefig(os.path.join(UT.dat_dir(), 'demo', 'demoBk%i.png' % itri), bbox_inches='tight')
        plt.close() 
    return None 


if __name__=="__main__": 
    #egBk()
    demoBk()
