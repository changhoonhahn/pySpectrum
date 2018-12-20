'''


some convenience functions for plotting powerspectrum or bispectrum 


'''
import numpy as np 
import matplotlib as mpl
import matplotlib.cm as cm
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


def plotBk(plottype, k1, k2, k3, BorQ, cnts, nbin=20, fig=None, ax=None, norm=True, cmap='viridis'): 
    ''' 
    by default k1 >= k2 >= k3 

    if plottype == 'amplitude': 
        B(k1,k2,k3) or Q(k1,k2,k3) as a function of triangle index
    
    if plottype == 'shape': 
        Make triangle plot that illustrates the shape of the bispectrum
    '''
    if fig is None and ax is None: 
        fig = plt.figure()

    if plottype == 'amplitude': 
        if ax is not None: raise ValueError
        x_bins = np.linspace(0., k1.max()+1, int(nbin)+1)
        y_bins = np.linspace(0., k1.max()+1, int(nbin)+1)
        theta12 = np.arccos((k1**2 + k2**2 - k3**2)/(2.*k1*k2))
        for i in range(3): 
            sub = fig.add_subplot(1,3,i+1) 
            theta_bin = (theta12 >= i*np.pi/6.) & (theta12 < (i+1)*np.pi/6.)
        
            bk1k2 = np.zeros((len(x_bins)-1, len(y_bins)-1))
            for i_x in range(len(x_bins)-1): 
                for i_y in range(len(y_bins)-1): 
                    lim = (theta_bin & 
                            (k1 >= x_bins[i_x]) & (k1 < x_bins[i_x+1]) &
                            (k2 >= y_bins[i_y]) & (k2 < y_bins[i_y+1])) 
                    if np.sum(lim) > 0: 
                        bk1k2[i_x, i_y] = np.average(BorQ[lim], weights=cnts[lim]) 
                    else: 
                        bk1k2[i_x, i_y] = -np.inf
            cont = sub.pcolormesh(x_bins, y_bins, bk1k2.T, vmin=BorQ.min(), vmax=BorQ.max(), cmap=cmap)
            if i == 0: 
                sub.text(0.05, 0.85, r'$\theta_{12} < \frac{\pi}{3}$', transform=sub.transAxes, fontsize=25)
            elif i == 1: 
                sub.set_yticklabels([]) 
                sub.text(0.05, 0.85, r'$\frac{\pi}{3} \le \theta_{12} < \frac{2\pi}{3}$', 
                        transform=sub.transAxes, fontsize=25)
            elif i == 2:
                sub.set_yticklabels([]) 
                sub.text(0.05, 0.85, r'$\frac{2\pi}{3} \le \theta_{12}$', transform=sub.transAxes, fontsize=25)

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.025, 0.02, 0.9])
        fig.colorbar(cont, format='%.0e', cax=cbar_ax)

        #plt.colorbar(cont, format='%.0e')
        bkgd = fig.add_subplot(111, frameon=False)
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel(r'$k_1$ [$h$/Mpc]', fontsize=25)
        bkgd.set_ylabel(r'$k_2$ [$h$/Mpc]', fontsize=25)
        fig.subplots_adjust(wspace=0.1)

    elif plottype == 'shape': 
        if ax is None:
            sub = fig.add_subplot(111)
        else: 
            sub = ax 
        k3k1 = k3/k1
        k2k1 = k2/k1

        x_bins = np.linspace(0., 1., int(nbin)+1)
        y_bins = np.linspace(0.5, 1., int(0.5*nbin)+1)
        BorQ_grid = _BorQgrid(k3k1, k2k1, BorQ, cnts, x_bins, y_bins)
        if norm: BorQ_grid /= BorQ_grid.max() 

        bplot = plt.pcolormesh(x_bins, y_bins, BorQ_grid.T, vmin=0., vmax=1., cmap=cmap)
        cbar = plt.colorbar(bplot, orientation='vertical') 

        sub.set_xlabel(r'$k_3/k_1$', fontsize=25)
        sub.set_xlim([0.0, 1.0]) 
        sub.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]) 
        sub.set_ylabel(r'$k_2/k_1$', fontsize=25)
        sub.set_ylim([0.5, 1.0]) 
        sub.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.]) 
    if ax is not None: 
        return sub 
    else: 
        return fig


def _BorQgrid(k3k1, k2k1, BorQ, cnts, x_bins, y_bins): 
    BorQ_grid = np.zeros((len(x_bins)-1, len(y_bins)-1))
    for i_x in range(len(x_bins)-1): 
        for i_y in range(len(y_bins)-1): 
            lim = ((k2k1 >= y_bins[i_y]) & 
                    (k2k1 < y_bins[i_y+1]) & 
                    (k3k1 >= x_bins[i_x]) & 
                    (k3k1 < x_bins[i_x+1]))
            if np.sum(lim) > 0: 
                BorQ_grid[i_x, i_y] = np.average(BorQ[lim], weights=cnts[lim])
            else: 
                BorQ_grid[i_x, i_y] = -np.inf
    return BorQ_grid
