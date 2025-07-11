## Inflation Data Interpretations v2.0.0
##
## 2/15/2021
## Connor Painter, The University of Texas at Austin
##
## The purpose of this document is to organize and clarify the outputs of simulations from Painter and Bunn (2021).



"""
IMPORTS AND SETTINGS
"""



import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import copy
import warnings
from scipy.interpolate import interp1d
from scipy.integrate import quad
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
saveImagesHere = "/Users/lwolski/Downloads"



D = {'A_s':(1e-12,1e-8), 
     'A_t':(1e-12,10**(-9.5)),
     'A_iso':(1e-20,1e-8),
     'n_s':(0.9,1), 
     'n_t':(-0.04,0),
     'n_iso':(0.85,1.05),
     'r':(1e-2,1e0),
     'r_iso':(1e-10,1e0),
     'beta_iso':(1e-10,1e-0)}

LTX = {'A_s':r"$A_S$",
       'A_t':r"$A_T$",
       'A_iso':r"$A_{\mathrm{iso}}$",
       'n_s':r"$n_S$",
       'n_t':r"$n_T$",
       'n_iso':r"$n_{\mathrm{iso}}$",
       'r':r"$r$",
       'r_iso':r"$(A_{\mathrm{iso}}/A_S)$",
       'beta_iso':r"$\beta_{\mathrm{iso}}$"}

logit = ['A_s', 'A_t', 'A_iso', 'r', 'r_iso', 'beta_iso']
iso = ['A_iso', 'n_iso', 'r_iso', 'beta_iso']

warnings.filterwarnings('ignore', '.*The maximum number.*')



"""
IMPORT SIMULATION OUTPUT
"""



path = "/Users/lwolski/Documents/Inflation Principled Forgetting Files/megatest1_07-08-25"



"""
MODEL CLASS
"""



class Model():
    
    """
    Extracts and organizes output data from a particular set of parameters on the random potential (s, V*, N).
    """
    
    def __init__(self, path, s, V0, N):
        
        self.path = path
        self.s = s
        self.V0 = V0
        self.N = N
        
        subs = np.array([f.path for f in os.scandir(path)])
        phi0_bins = [0,0]
        yes = np.array([], dtype=int)
        
        for i in range(len(subs)):
            
            b = os.path.basename(subs[i])
            _ = [c for c in range(len(b)) if b[c]=='_']
            if len(_)==5:
                if float(b[_[0]+1:_[1]]) == s and float(b[_[1]+1:_[2]]) == V0 and float(b[_[2]+1:_[3]]) == N:
                    phi0_bins = np.vstack(( phi0_bins, [float(b[_[3]+1:_[4]]), float(b[_[4]+1:])] ))
                    yes = np.append(yes, [i])
        
        subs = subs[yes]
        phi0_bins = phi0_bins[1:]
        i_sort = np.argsort(phi0_bins, axis=0)[:,0]
        self.subs = subs[i_sort]
        self.phi0_bins = phi0_bins[i_sort]
        self.n_shells = len(self.phi0_bins)
        self.c = mpl.cm.get_cmap('plasma')(np.linspace(0,1,self.n_shells))
        
        assert len(self.subs)==len(self.phi0_bins), "Subfolders and shell boundaries are not corresponding."
        
        self.df = [pd.read_csv(os.path.join(sub, 'major_quantities.csv')) for sub in self.subs]
        self.rows = [len(df) for df in self.df]
        self.maxrows = np.max([np.shape(df)[0] for df in self.df])
        self.id = self.get('id')
        self.from0 = self.get('from0')
        self.N_e = self.get('N_e')
        
        self.A_t = self.get('A_t')
        self.n_t = self.get('n_t')
        self.A_s = self.get('A_s')
        self.n_s = self.get('n_s')
        self.r = self.get('r')
        self.A_iso = self.get('A_iso')
        self.n_iso = self.get('n_iso')
        self.r_iso = self.A_iso/self.A_s
        self.beta_iso = self.A_iso/(self.A_s + self.A_iso)
        
        self.simtime = self.get('sim_time')
        self.forgets = self.get('forgets')
        self.upscale = self.get('upscale')
        
        self.fails_inwardness = self.get('fails:inwardness')
        self.fails_N_e = self.get('fails:e-folds')
        self.fails_total = self.get('fails:total')
        self.fails_exception = self.get('fails:exception')
        self.fails_overall = self.get('fails:overall')
        self.attempts = np.nansum(self.fails_total, axis=1) + self.rows
        self.success_convergence = self.get('success:convergence')
        self.success_N_e = self.get('success:e-folds')
        self.success_overall = (self.rows - np.nansum(self.fails_overall, axis=1))/self.attempts
        self.bowl = np.reshape([self.get(f'bowl{i}') for i in range(N)], (self.n_shells, self.maxrows, N))
        
        self.w = self.binWeights()
        
        return
        
        
    
    def get(self, q, i=None, log10=False, nan_outliers=True):
        
        if i is None: i = np.arange(self.n_shells)
        if isinstance(i, int): i = [i]
        
        data = np.empty((len(i), self.maxrows),dtype=object)
        data[:] = np.nan

        if q in self.df[0].columns:

            for j in range(len(i)):
                this = self.df[i[j]][q]
                data[j][:len(this)] = np.array(this)
        
        elif q=='r_iso':
            try: data = self.r_iso
            except:
                A_iso = self.get('A_iso')
                A_s = self.get('A_s')
                self.r_iso = A_iso/A_s
                data = self.r_iso
        
        elif q=='beta_iso':
            try: data = self.beta_iso
            except:
                A_iso = self.get('A_iso')
                A_s = self.get('A_s')
                self.beta_iso = A_iso/(A_s + A_iso)
                data = self.beta_iso
        
        if log10:
            data = np.float32(data)
            data = np.log10(data)
        if nan_outliers and q in D: 
            domain = np.log10(D[q]) if log10 else D[q]
            data = np.where((data<domain[0]) | (data>=domain[1]), np.nan, data)
        
        return data
    
    
    
    def stats(self, q, i=None, log10=None):
        
        if i is None: i = np.arange(self.n_shells)
        if isinstance(i, int): i = [i]
        if log10 is None: log10 = (q in logit)
        
        data = self.get(q, i, log10)
        data = np.float32(data)
        mu = np.nanmean(data, axis=1)
        var = np.nanstd(data, axis=1)**2
        
        return mu, var
    
    
    
    def binWeights(self, i=None, PDF_domain=(0,4)):
        
        if i is None: i = np.arange(self.n_shells)
        if isinstance(i, int): i = [i]
        
        from0 = np.mean(self.phi0_bins, axis=1)
        from0 = np.concatenate(( [PDF_domain[0]], from0, [PDF_domain[1]] ))
        success = np.concatenate(( [0], self.success_overall, [0] ))
        PDF_unnormed = interp1d(from0, success*from0**self.N)
        norm = quad(PDF_unnormed, PDF_domain[0], PDF_domain[1])[0]
        PDF = interp1d(from0, success*from0**(self.N-1)/norm)
        
        w = np.zeros(len(i))
        for b in range(len(i)): 
            start = self.phi0_bins[i[b]][0]
            end = self.phi0_bins[i[b]][1]
            w[b] = quad(PDF, start, end)[0]
        
        return w
    


    def cultivation(self, q, log10=None, w=None):
        
        if log10 is None: log10 = (q in logit)
        if w is None: w = self.w
        
        mu, var = self.stats(q, log10=log10)
        w_mu = np.nansum(w * mu)/np.nansum(w)
        w_var = np.nansum(w * mu**2)/np.nansum(w) - w_mu**2
        
        return w_mu, w_var
    
    
    
    def hist(self, q, i=None, log10=None, sigma=False, ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 400)
        bins = kwargs.pop('bins', 100)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', f"Histogram - N{self.N} - {q}")
        ext = kwargs.pop('ext', '.pdf')
        
        if i is None: i = np.arange(self.n_shells)
        if isinstance(i, int): i = [i]
        if log10 is None: log10 = (q in logit)
        
        data = self.get(q, i, log10)
        weights = np.repeat(self.w, self.maxrows).reshape(np.shape(data), order='F')
        labels = [r"$\phi_i \in [{}s,{}s]$".format(*self.phi0_bins[i[b]]) for b in range(len(i))]
        
        if ax is None: fig, ax = plt.subplots(dpi=dpi)
        n, bs, patches = ax.hist(data.T, weights=weights.T, bins=bins, histtype='barstacked', color=self.c, label=labels, **kwargs)
        
        mu, var = self.cultivation(q, log10)
        ylim = ax.get_ylim()
        ax.vlines([mu], *ylim, linestyle='dashed', color='k', label="Weighted Mean")
        if sigma: ax.vlines([mu-2*np.sqrt(var), mu+2*np.sqrt(var)], *ylim, linestyle='dashed', color='k')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.5)
        
        ax.set_xlabel(log10*r"$\log_{10}$" + LTX[q], fontsize=16)
        ax.set_ylabel("Normalized Frequency", fontsize=16)
        ax.text(0.9, 0.9, rf"$N = {self.N}$", fontsize=20, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.tick_params(labelsize=16)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=4)
        
        if save: plt.savefig(saveImagesHere + '/' + filename + ext, bbox_inches='tight')
        
        return
    
    
    
    def scatterCloud(self, q1, q2, i=None, log10=[None,None], ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 400)
        figsize = kwargs.pop('figsize', (8,6))
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', f"Scatterplot - N{self.N} - {q1} v {q2}")
        ext = kwargs.pop('ext', '.pdf')
        
        if i is None: i = np.arange(self.n_shells)
        if isinstance(i, int): i = [i]
        if log10[0] is None: log10[0] = (q1 in logit)
        if log10[1] is None: log10[1] = (q2 in logit)
        
        data1 = self.get(q1, i, log10[0])
        data2 = self.get(q2, i, log10[1])
        labels = [r"$\phi_i \in [{}s,{}s]$".format(*self.phi0_bins[b]) for b in range(len(i))]
        
        if ax is None: fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        
        [ax.scatter(data1[b], data2[b], marker='+', color=self.c[i[b]], label=labels[b]) for b in range(len(i))]
        mu1, var1 = self.cultivation(q1, log10[0])
        mu2, var2 = self.cultivation(q2, log10[1])
        ax.scatter([mu1], [mu2], marker='*', color='k', label="Weighted Mean")
        ax.legend()
        ax.grid(alpha=0.5)
        
        ax.set_xlabel(log10[0]*r"$\log_{10}$" + LTX[q1], fontsize=16)
        ax.set_ylabel(log10[1]*r"$\log_{10}$" + LTX[q2], fontsize=16)
        ax.text(0.9, 0.9, rf"$N = {self.N}$", fontsize=20, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.tick_params(labelsize=16)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=6)
        
        if save: plt.savefig(saveImagesHere + '/' + filename + ext, bbox_inches='tight')
        
        return
    
    
    
    def KDE(self, q1, q2, i=None, log10=[None,None], mupoint=False, ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 400)
        figsize = kwargs.pop('figsize', (8,6))
        levels = kwargs.pop('levels', 2)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', f"KDE - N{self.N} - {q1} v {q2}")
        ext = kwargs.pop('ext', '.pdf')
        
        if i is None: i = np.arange(self.n_shells)
        if isinstance(i, int): i = [i]
        if log10[0] is None: log10[0] = (q1 in logit)
        if log10[1] is None: log10[1] = (q2 in logit)
        
        data1 = self.get(q1, i, log10[0])
        data2 = self.get(q2, i, log10[1])
        w = np.repeat(self.w, self.maxrows)
        data1flat, data2flat = data1.flatten(), data2.flatten()
        i_notnan = ~np.isnan(data1flat) & ~np.isnan(data2flat)
        data1flat, data2flat, w = data1flat[i_notnan], data2flat[i_notnan], w[i_notnan]
        
        kde = sp.stats.gaussian_kde((data1flat, data2flat), weights=w)
        x_bounds = np.log10(D[q1]) if log10[0] else D[q1]
        y_bounds = np.log10(D[q2]) if log10[1] else D[q2]
        x, y = np.mgrid[x_bounds[0]:x_bounds[1]:200*1j, y_bounds[0]:y_bounds[1]:200*1j]
        z = kde(np.vstack([x.flatten(), y.flatten()]))
        
        if ax is None: fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        ax.pcolormesh(x, y, z.reshape(x.shape), shading='gouraud', zorder=0)
        ax.contour(x, y, z.reshape(x.shape), levels=levels, colors='k', linewidths=3, zorder=5)
        ax.grid(alpha=0.5)
        if mupoint:
            mu1, var1 = self.cultivation(q1, log10[0])
            mu2, var2 = self.cultivation(q2, log10[1])
            ax.scatter([mu1], [mu2], marker='*', color='r', label="Weighted Mean", zorder=10)
        
        ax.set_xlabel(log10[0]*r"$\log_{10}$" + LTX[q1], fontsize=16)
        ax.set_ylabel(log10[1]*r"$\log_{10}$" + LTX[q2], fontsize=16)
        ax.text(0.9, 0.9, rf"$N = {self.N}$", color='white', fontsize=20, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax.tick_params(labelsize=16)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=6)
        
        if save: plt.savefig(saveImagesHere + '/' + filename + ext, bbox_inches='tight')
        
        return
    
    
    
    def scatterAndKDE(self, q1, q2, i=None, log10=[None,None], ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 400)
        figsize = kwargs.pop('figsize', (16,6))
        levels = kwargs.pop('levels', 2)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', f"Scatterplot and KDE - N{self.N} - {q1} v {q2}")
        ext = kwargs.pop('ext', '.pdf')
        
        if ax is None: 
            fig, ax = plt.subplots(ncols=2, sharey=True, dpi=dpi, figsize=figsize)
            fig.tight_layout(pad=0)
        self.scatterCloud(q1, q2, i, log10, ax[0])
        self.KDE(q1, q2, i, log10, ax[1], levels=levels)
        ax[1].set_ylabel(None)
        
        if save: plt.savefig(saveImagesHere + '/' + filename + ext, bbox_inches='tight')
        
        return
    
    
    
    def triangleOfCorrelations(self, qs=['A_s', 'n_s', 'n_t', 'r', 'beta_iso'], i=None, log10=[None]*5, **kwargs):
        
        dpi = kwargs.pop('dpi', 400)
        figsize = kwargs.pop('figsize', (8.5*2.5,8.5*2.5))
        levels = kwargs.pop('levels', 2)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', f"Triangle of Correlations - N{self.N}")
        ext = kwargs.pop('ext', '.pdf')
        
        log10 = np.array(log10)
        fig, ax = plt.subplots(len(qs), len(qs), sharex='col', sharey='row', dpi=dpi, figsize=figsize)
        for j in range(len(qs)):
            for k in range(len(qs)):
                q1, q2 = qs[k], qs[j]
                print(f"Handling {q1} vs. {q2} subplot...")
                if k<j:
                    if (q1 not in iso and q2 not in iso) or self.N != 1:
                        self.KDE(q1, q2, i, log10[[k,j]], ax=ax[j,k], levels=levels)
                    else:
                        ax[j,k].text(0.5, 0.5, "N/a", color='k', fontsize=20, horizontalalignment='center', verticalalignment='center', transform = ax[j,k].transAxes)   
                    ax[j,k].get_shared_y_axes().remove(ax[j,j])
                    if k!=0: ax[j,k].set_ylabel(None)
                    if j!=len(qs)-1: ax[j,k].set_xlabel(None)
                if k==j:
                    if q1 not in iso or self.N != 1:
                        self.hist(q1, i, log10[j], ax=ax[j,k])
                    else:
                        ax[j,k].text(0.5, 0.5, "N/a", color='k', fontsize=20, horizontalalignment='center', verticalalignment='center', transform = ax[j,k].transAxes)   
                    if j!=len(qs)-1: ax[j,k].set_xlabel(None)
                if k>j:
                    fig.delaxes(ax[j,k])
        
        fig.tight_layout(pad=0)
        
        if save: plt.savefig(saveImagesHere + '/' + filename + ext, bbox_inches='tight')
        
        return
                
        
    
    
    
def histGrid(Ms, qs=['A_s', 'n_s', 'n_t', 'r', 'beta_iso'], **kwargs):
    
    save = kwargs.pop('save', True)
    filename = kwargs.pop('filename', "Quantity Histograms")
    ext = kwargs.pop('ext', '.pdf')
    
    fig, ax = plt.subplots(len(qs), len(Ms), dpi=400, figsize=(8.5*2,11*2))
    
    for i in range(len(qs)):
        
        q = qs[i]
        
        for j in range(len(Ms)):
            
            M = Ms[j]
            if q not in iso or M.N != 1:
                M.hist(q, ax=ax[i,j], **kwargs)
    
    fig.delaxes(ax[4,0])
    
    if save: plt.savefig(saveImagesHere + '/' + filename + ext)
    
    return








            
        
        
        
        
        
    

    
"""
RELEVANT GLOBAL MODEL VARIABLES
"""



#M1 = Model(path, 30, 5e-9, 1)
M2 = Model(path, 30, 5e-9, 2)
#M3 = Model(path, 30, 5e-9, 3)
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
