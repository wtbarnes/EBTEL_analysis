#ebtel_plot.py

#Will Barnes
#7 May 2015

#Import necessary modules
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
    
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

#Resolve Python 2/3 exception problem
exc = getattr(__builtin__,"IOError","FileNotFoundError")

class Plotter(object):

    def __init__(self,lvl0_filename=None,dpi=1000,fontsize=18,alfs=0.65,figsize=(8,8),fformat='eps',two_fluid=True,**kwargs):
        #configure logger
        self.logger = logging.getLogger(type(self).__name__)
        #configure keyword arguments
        self.dpi = dpi
        self.fontsize = fontsize
        self.alfs = alfs
        self.figsize = figsize
        self.fformat = fformat
        self.two_fluid = two_fluid
        #plotting helpers
        self.linestyles = ('-','--','-.',':')
        #load variables
        if lvl0_filename is not None:
            self.load_variables(lvl0_filename)
        else:
            self.logger.warning("No file specified. Variable namespace will not be populated.")

    def load_variables(self,lvl0_filename,**kwargs):
        #load plasma parameters
        try:
            index_offset = 0
            data = np.loadtxt(lvl0_filename+'.txt')
            if self.two_fluid:
                index_offset += 1 
                self.temp_i = data[:,2]
                self.temp_apex_i = data[:,8]
            self.time = data[:,0]
            self.temp_e = data[:,1]
            self.dens = data[:,2+index_offset]
            self.temp_apex_e = data[:,5+2*index_offset]
            self.dens_apex = data[:,6+3*index_offset]
            self.heat = data[:,10+5*index_offset]
        except exc:
            self.logger.warning("Unable to load plasma parameters from %s."%(lvl0_filename+'.txt'))
            pass

        #load dem parameters
        try:
            data = np.loadtxt(lvl0_filename+'_dem.txt')
            self.temp_dem = data[:,0]
            self.dem_tr = data[:,1]
            self.dem_cor = data[:,2]
            self.dem_tot = data[:,3]
            self.em_cor = data[:,4]
        except exc:
            self.logger.warning("Unable to load DEM parameters from %s."%(lvl0_filename+'_dem.txt'))
            pass
            
        #load heat parameters
        try:
            self.events = np.loadtxt(lvl0_filename+'_heat_amp.txt')
        except exc:
            self.logger.warning("Unable to load heating event amplitudes from %s."%(lvl0_filename+'_heat_amp.txt'))
            pass


    def plot_params(self,print_fig_filename=None,**kwargs):
        #set up figure
        fig,ax = plt.subplots(3,1,figsize=(1.5*self.figsize[0],self.figsize[1]),sharex=True)
        ax_n = ax[1].twinx()
        ax_na = ax[2].twinx()
        
        if self.two_fluid:
            tlab = r'$T_e$'
        else:
            tlab = r'$T$'

        #plot heating
        ax[0].plot(self.time,self.heat,color=sns.color_palette('deep')[0])
        ax[0].set_ylabel(r'$h$ (erg cm$^{-3}$ s$^{-1}$)',fontsize=self.fontsize)
        ax[0].set_xlim([self.time[0],self.time[-1]])
        ax[0].locator_params(nbins=5)
        ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-2,2) )
        ax[0].tick_params(axis='both',labelsize=self.alfs*self.fontsize,pad=8)
        #plot average temperature and density
        line_te = ax[1].plot(self.time,self.temp_e/10**6,label=tlab,color=sns.color_palette('deep')[0])
        if self.two_fluid:
            line_ti = ax[1].plot(self.time,self.temp_i/10**6,color=sns.color_palette('deep')[2],label=r'$T_i$')
        ax[1].set_ylabel(r'$T$ (MK)',fontsize=self.fontsize)
        ax[1].yaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax[1].locator_params(nbins=5)
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2) )
        ax[1].tick_params(axis='both',labelsize=self.alfs*self.fontsize,pad=8)
        line_n = ax_n.plot(self.time,self.dens/10**8,color=sns.color_palette('deep')[0],linestyle='--',label=r'$n$')
        ax_n.set_ylabel(r'$n$ (10$^8$ cm$^{-3}$)',fontsize=self.fontsize)
        ax_n.yaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax_n.locator_params(nbins=5)
        ax_n.ticklabel_format(axis='y', style='sci', scilimits=(-2,2) )
        ax_n.tick_params(axis='both',labelsize=self.alfs*self.fontsize,pad=8)
        ax[1].set_xlim([self.time[0],self.time[-1]])
        #plot apex temperature and density
        ax[2].plot(self.time,self.temp_apex_e/10**6,color=sns.color_palette('deep')[0])
        if self.two_fluid:
            ax[2].plot(self.time,self.temp_apex_i/10**6,color=sns.color_palette('deep')[2])
        ax[2].set_ylabel(r'$T_a$ (MK)',fontsize=self.fontsize)
        ax[2].yaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax[2].locator_params(nbins=5)
        ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-2,2) )
        ax[2].tick_params(axis='both',labelsize=self.alfs*self.fontsize,pad=8)
        ax_na.plot(self.time,self.dens_apex/10**8,color=sns.color_palette('deep')[0],linestyle='--')
        ax_na.set_ylabel(r'$n_a$ (10$^8$ cm$^{-3}$)',fontsize=self.fontsize)
        ax_na.yaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax_na.locator_params(nbins=5)
        ax_na.ticklabel_format(axis='y', style='sci', scilimits=(-2,2) )
        ax_na.tick_params(axis='both',labelsize=self.alfs*self.fontsize,pad=8)
        ax[2].set_xlim([self.time[0],self.time[-1]])
        ax[2].set_xlabel(r'$t$ (s)',fontsize=self.fontsize)

        #configure legend
        lines = line_te + line_n
        if self.two_fluid:
            lines = line_te + line_ti + line_n
        labels = [l.get_label() for l in lines]
        ax[1].legend(lines,labels,loc=1)

        #Check if output filename is specified
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
        else:
            plt.show()


    def plot_dem(self,print_fig_filename=None,**kwargs):
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        #plot dem curves
        ax.plot(self.temp_dem,self.dem_tr,label=r'TR')
        ax.plot(self.temp_dem,self.dem_cor,label=r'corona')
        ax.plot(self.temp_dem,self.dem_tot,label=r'total')
        ax.plot(self.temp_dem,self.em_cor,label=r'EM$_{corona}$')
        ax.legend()
        ax.set_xlabel(r'$\log{T}$ (K)',fontsize=self.fontsize)
        ax.set_ylabel(r'$\log{\mathrm{DEM}}$ (cm$^{-5}$ K$^{-1}$)',fontsize=self.fontsize)
        ax.set_xlim([5.5,7.5])

        #Check if output filename is specified
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
        else:
            plt.show()


    def plot_event_distribution(self,print_fig_filename=None,noise_thresh=0.01,return_params=True,**kwargs):
        """Fit event energy distribution with a power-law and plot it."""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        #Create a histogram
        num_bins = self._freedman_diaconis()
        n,bins,patches = ax.hist(self.events,num_bins,histtype='stepfilled',facecolor='blue',alpha=0.25,label=r'Events')
        bin_centers = np.log10(np.diff(bins)/2.0+bins[0:-1])
        
        #Fit with "graphical method"
        #check for bins with no entries in them; below these entries (if they exist), don't calculate fit
        noise = np.where(n <= int(np.max(n)*noise_thresh))
        if len(noise[0]) > 0:
            n = n[0:noise[0][0]]
            bin_centers = bin_centers[0:noise[0][0]]

        #calculate fit
        pars,covar = curve_fit(self._power_law_curve,bin_centers,np.log10(n),sigma=np.sqrt(np.log10(n)))
        pl_fit = self._power_law_curve(bin_centers,*pars)

        #exception for when uncertainty calculation fails
        try:
            sigma = np.sqrt(np.diag(covar))
        except:
            sigma = [0.0,0.0]
            print("Uncertainty calculation failed. Resulting value is a placeholder.")
            pass
            
        #estimate power-law fit using maximum likelihood estimation (see D'Huys et al., 2016, Sol. Phys.)
        xmin = np.min(self.events)
        alpha_mle = 1. + len(self.events)*1.0/(np.sum(np.log(self.events/xmin)))
        sigma_mle = (alpha_mle - 1.)/np.sqrt(len(self.events))

        #plot fit
        ax.plot(10**bin_centers,10**pl_fit,'--r',label=r'Fit',linewidth=2.0)
        ax.set_xlabel(r'$E_H$ (erg cm$^{-3}$ s$^{-1}$)',fontsize=self.fontsize)
        ax.set_ylabel(r'Number of Events',fontsize=self.fontsize)
        ax.set_title(r'Graphical: $\alpha$ = %.2f $\pm$ %.2e, MLE: $\alpha$= %.2f $\pm$ %.2e' % (pars[1], sigma[1], alpha_mle, sigma_mle),fontsize=self.fontsize)
        ax.set_yscale('log',nonposy='clip')
        ax.set_xscale('log')
        ax.set_xlim([np.min(self.events),np.max(self.events)])
        ax.tick_params(axis='both',labelsize=0.75*self.fontsize)
        ax.legend(fontsize=0.75*self.fontsize,loc=1)

        #Check if output filename is specified
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
            
        if return_params:
            return {'graphical':{'alpha':pars[1],'sigma':sigma[1]},'mle':{'alpha':alpha_mle,'sigma':sigma_mle}}


    def _power_law_curve(self,x,a,b):
        return a + b*x
    
    def _freedman_diaconis(self,**kwargs):
        q75,q25 = np.percentile(self.events,[75,25])
        iqr = q75 - q25
        w = 2.0*iqr*(len(self.events))**(-1.0/3.0)
        return int((np.max(np.array(self.events)) - np.min(np.array(self.events)))/w)


    def plot_surface(self,param_1,param_2,surf_list,**kwargs):
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        #set colorbar limits
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = np.min(np.array(surf_list))
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = np.max(np.array(surf_list))

        #set up mesh
        p1_mesh,p2_mesh = np.meshgrid(np.array(param_1),np.array(param_2))
        surf = ax.pcolormesh(p1_mesh,p2_mesh,np.array(surf_list),cmap='hot',vmin=vmin,vmax=vmax)
        fig.colorbar(surf,ax=ax)

        #set limits
        if 'xlim' in kwargs:
            ax.set_xlim([kwargs['xlim'][0],kwargs['xlim'][1]])
        else:
            ax.set_xlim([param_1[0],param_1[-1]])
        if 'ylim' in kwargs:
            ax.set_ylim([kwargs['ylim'][0],kwargs['ylim'][1]])
        else:
            ax.set_ylim([param_2[0],param_2[-1]])

        #set labels
        if 'ylab' in kwargs:
            ax.set_ylabel(kwargs['ylab'],fontsize=self.fontsize)
        if 'xlab' in kwargs:
            ax.set_xlabel(kwargs['xlab'],fontsize=self.fontsize)
        if 'plot_title' in kwargs:
            ax.set_title(kwargs['plot_title'],fontsize=self.fontsize)

        #Check if output filename is specified
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
        else:
            plt.show()
