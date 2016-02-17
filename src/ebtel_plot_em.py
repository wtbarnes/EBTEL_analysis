#ebtel_plot_em.py

#Will Barnes
#14 May 2015

#Import needed modules
import logging
import dill as pickle
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

class DEMPlotter(object):

    def __init__(self,em,em_stats,fits,fits_stats,Tn = np.arange(250,5250,250),dpi=1000,fontsize=18.,figsize=(8,8),alfs=0.75,fformat='eps',**kwargs):
        #set up logger
        self.logger = logging.getLogger(type(self).__name__)
        #arguments
        self.em = em
        self.em_stats = em_stats
        self.fits = fits
        self.fits_stats = fits_stats
        #keyword arguments
        self.dpi,self.fontsize,self.figsize,self.alfs,self.fformat = dpi,fontsize,figsize,alfs,fformat
        self.Tn = Tn
        self.Tndelta = self.Tn[1] - self.Tn[0]
        #static parameters for plot styling
        self.linestyles = (':','-.','--','-')
        self.colors = []
        [self.colors.extend(len(self.linestyles)*[sns.color_palette('deep')[i]]) for i in range(int(len(self.Tn)/len(self.linestyles)))]
        if len(self.colors) != len(self.Tn):
            self.logger.warning("Number of colors does not match number of wait-time values. Reconfigure one or the other before plotting.")


    def plot_em_curves(self,delta_em = 0.2,y_limits=[10**26.,10**32.],print_fig_filename=None,**kwargs):
        """Plot mean EM distributions for each Tn with superimposed fit lines"""
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        for i in range(len(self.em_stats)):
            #em
            ax.plot(self.em_stats[i]['T_mean'], 10.**(i*delta_em)*self.em_stats[i]['em_mean'], linestyle=self.linestyles[i%len(self.linestyles)], color=self.colors[i], label=r'$%d$'%self.Tn[i] )
            #fit lines
            if self.fits_stats[i]['cool'] is not None:
                t = np.array([self.fits_stats[i]['cool']['limits'][0],self.fits_stats[i]['cool']['limits'][1]])
                ax.plot(t,10.**(i*delta_em)*10.**self.fits_stats[i]['cool']['b']*t**self.fits_stats[i]['cool']['a'],color=sns.color_palette('bright')[0], linewidth=2, linestyle='solid')
            if self.fits_stats[i]['hot'] is not None:
                t = np.array([self.fits_stats[i]['hot']['limits'][0],self.fits_stats[i]['hot']['limits'][1]])
                ax.plot(t,10.**(i*delta_em)*10.**self.fits_stats[i]['hot']['b']*t**self.fits_stats[i]['hot']['a'],color=sns.color_palette('bright')[2], linewidth=2, linestyle='solid')

        #plot options
        ax.set_xlabel(r'$\log{T}$ $\mathrm{(K)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$\log\mathrm{EM}$ $\mathrm{(cm}^{-5}\mathrm{)}$',fontsize=self.fontsize)
        ax.set_xlim([10**5.5,10**7.5])
        ax.set_ylim(y_limits)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        #legend
        leg = ax.legend(loc=2,fontsize=self.alfs*self.fontsize,title=r'$T_N$ $\mathrm{(s)}$',ncol=2,bbox_to_anchor=(-0.1,1.07))
        plt.setp(leg.get_title(),fontsize=self.alfs*self.fontsize)
        #avoid cutting off labels
        plt.tight_layout()

        #save or show the figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.format,format=self.format,dpi=self.dpi,bbox_extra_artists=[leg],bbox_inches='tight')
        else:
            plt.show()


    def plot_em_curve(self,tn_val,y_limits=[10**25.,10**28.],**kwargs):
        """Plot MC and mean EM curves for single value of Tn"""
        
        tn_index = np.where(self.Tn==tn_val)[0]
        if len(tn_index) != 1:
            raise ValueError("Invalid wait time %f. Use a valid value from self.Tn"%(tn_val))
        else:
            tn_index = tn_index[0]
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        #standard deviation
        ax.fill_between(self.em_stats[tn_index]['T_mean'], self.em_stats[tn_index]['em_mean']-self.em_stats[tn_index]['em_std'], self.em_stats[tn_index]['em_mean']+self.em_stats[tn_index]['em_std'], facecolor=sns.color_palette('deep')[2], edgecolor=sns.color_palette('deep')[2], alpha=0.75)
        #MC histograms
        for h in self.em[tn_index]:
            ax.hist(h['T'],bins=h['bins'],weights=h['em'],histtype='step',color=sns.color_palette('deep')[0],linestyle=self.linestyles[-1],alpha=0.1)
        #mean
        ax.plot(self.em_stats[tn_index]['T_mean'],self.em_stats[tn_index]['em_mean'],color='black',linestyle=self.linestyles[-1],linewidth=2)
        
        #set labels
        ax.set_xlabel(r'$\log{T}$ $\mathrm{(K)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$\log{\mathrm{EM}}$ $\mathrm{(cm}^{-5}\mathrm{)}$',fontsize=self.fontsize)
        ax.set_xlim([10**5.5,10**7.5])
        ax.set_ylim(y_limits)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()


    def plot_em_max(self,temp_max,em_max,**kwargs):
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        ax_twin = ax.twinx()

        for i in range(len(self.Tn)):
            #calculate average and std
            mean_temp_max = np.mean(temp_max[i])
            std_temp_max = np.std(temp_max[i])
            mean_em_max = np.mean(em_max[i])
            std_em_max = np.std(em_max[i])
            #plot points
            ax.errorbar(self.Tn[i],mean_temp_max,yerr=std_temp_max,fmt='o',color='black')
            ax_twin.errorbar(self.Tn[i],mean_em_max,yerr=std_em_max,fmt='*',color='black')

        #set labels
        ax.set_xlabel(r'$T_N$ $\mathrm{(s)}$',fontsize=self.fs)
        ax.set_ylabel(r'$\log{T_{max}}$ $\mathrm{(K)}$',fontsize=self.fs)
        ax.set_ylim([5.5,7.0])
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.tick_params(axis='both',labelsize=self.alfs*self.fs)
        ax_twin.set_ylabel(r'$\log{\mathrm{EM}(T_{max})}$ $\mathrm{(cm}^{-5}\mathrm{)}$',fontsize=self.fs)
        ax_twin.set_ylim([28,30])
        ax_twin.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fs)
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()


    def plot_em_slopes(self,**kwargs):
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        marker_cool,marker_hot = [],[]
        for i in range(len(self.cool_fits)):
            if self.cool_fits[i][0] is not False and self.cool_fits[i][2] is not False:
                marker_cool = ax.errorbar(self.Tn[i],self.cool_fits[i][0],yerr=self.cool_fits[i][2][0],fmt='o',color='blue',label=r'$\mathrm{cool}$')

            if self.hot_fits[i][0] is not False and self.hot_fits[i][2] is not False:
                marker_hot = ax.errorbar(self.Tn[i],np.fabs(self.hot_fits[i][0]),yerr=self.hot_fits[i][2][0],fmt='o',color='red',label=r'$\mathrm{hot}$')

        #set labels
        ax.set_xlabel(r'$T_N$ $\mathrm{(s)}$',fontsize=self.fs)
        ax.set_ylabel(r'$\mathrm{EM}$ $\mathrm{scaling}$',fontsize=self.fs)
        ax.axhline(y=2,color='k',linestyle='--')
        ax.axhline(y=3,color='k',linestyle='-')
        ax.axhline(y=5,color='k',linestyle='-.')
        ax.set_ylim([0,8])
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.set_yticks(self.tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fs)
        
        #legend
        if marker_cool and marker_hot:
            ax.legend([marker_cool,marker_hot],[r'$a$, $\mathrm{cool}$',r'$b$, $\mathrm{hot}$'],loc=1,fontsize=self.alfs*self.fs,numpoints=1)
            
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def plot_em_derivs(self,**kwargs):
        """Plot d(log(EM))/d(log(T)) as a function of T for all values of Tn"""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        #Iterate over t_wait values
        for i in range(len(self.em_mean)):    
            #remove data below given threshold of mean EM and T
            filter_inds = np.where(self.em_mean[i] > (self.em_cutoff))
            em_filter = np.array(self.em_mean[i])[filter_inds[0]]
            temp_filter = np.array(self.temp_mean[i])[filter_inds[0]]
            #compute derivative 
            dem_dt = np.gradient(em_filter,np.gradient(temp_filter))
            #plotting
            ax.plot(temp_filter, dem_dt, color=self.colors[i], linestyle=self.linestyles[i%len(self.linestyles)])
            
        #set labels
        ax.set_xlabel(r'$\log{T}$ $\mathrm{(K)}$',fontsize=self.fs)
        ax.set_ylabel(r'$d\log{\mathrm{EM}}/d\log{T}$',fontsize=self.fs)
        ax.axhline(y=2,color='k',linestyle=':')
        ax.axhline(y=3,color='k',linestyle=':')
        ax.axhline(y=-2.5,color='k',linestyle=':')
        ax.axhline(y=-5.5,color='k',linestyle=':')
        if 'y_limits' in kwargs:
            ax.set_ylim(kwargs['y_limits'])
        else:
            ax.set_ylim([-10,6])
        ax.set_xlim([5.5,7.5])
        ax.set_yticks(self.tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fs)
        #avoid cutting off labels
        plt.tight_layout()
        
        #save or show figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def tick_maker(self,old_ticks,n,**kwargs):
        if n < 2:
            raise ValueError('n must be greater than 1')
        
        n = n-1
        delta = (old_ticks[-1] - old_ticks[0])/n
        new_ticks = []
        for i in range(n):
            new_ticks.append(old_ticks[0] + i*delta)
        
        new_ticks.append(old_ticks[0] + n*delta)
        return new_ticks
        
        
class EMHistoBuilder(object):
    """Class to build histograms of slope values to compare across heating functions and heating frequencies"""
    
    def __init__(self,species,loop_length,tpulse,alpha,**kwargs):
        
        #Load in heating function values and set needed variables
        self.alpha = alpha
        if len(self.alpha) == 1:
            print("Warning: You have only specified one heating function.")
        if 'group' in kwargs:
            self.group = kwargs['group']
        else:
            self.group = 'by_alpha'
        if 'root_dir' in kwargs:
            root_dir = kwargs['root_dir']
        else:
            root_dir = '/data/datadrive2/EBTEL-2fluid_figs/'
        #Plotting options
        if 'dpi' in kwargs:
            self.dpi = kwargs['dpi']
        else:
            self.dpi = 1000
        if 'format' in kwargs:
            self.format = kwargs['format']
        else:
            self.format = 'eps'
        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        else:
            self.fs = 18.0
        if 'alfs' in kwargs:
            self.alfs = kwargs['alfs']
        else:
            self.alfs = 0.75
        if 'figsize' in kwargs:
            self.figsize = kwargs['figsize']
        else:
            self.figsize = (12,12)
        #Assemble temp file name
        self.fn_temp = root_dir + species + '_heating_runs/alpha%s/ebtel_L' +str(loop_length) + '_tpulse' + str(tpulse) + '_alpha%s%s_' + species + '_heating_all_a.fits'
        #Initialize dictionary to store separate histograms
        self.histo_dict_cool = {}
        self.histo_dict_hot = {}
        
            
    def loader(self,interval,**kwargs):
        """Load in data and create dictionaries with slope values grouped according to 'group' option"""
        
        #Loop over (alpha,b) values 
        for ab in self.alpha:
            #Unpickle the file
            with open(self.fn_temp%(ab[0],ab[0],ab[1]),'rb') as f: 
                cool,hot = pickle.load(f)
            f.close()
            #Group by alpha method
            if self.group is 'by_alpha':
                self.histo_dict_cool[''.join(ab)] = list(itertools.chain(*cool))
                self.histo_dict_hot[''.join(ab)] = list(itertools.chain(*hot))
            #Group by Tn method
            elif self.group is 'by_t_wait':
                for i in np.arange(0 + interval,20+interval,1+interval):
                    try:
                        self.histo_dict_cool[str(i)] = self.histo_dict_cool[str(i)] + cool[i]
                        self.histo_dict_hot[str(i)] = self.histo_dict_hot[str(i)] + hot[i]
                    except KeyError:
                        self.histo_dict_cool[str(i)] = []
                        self.histo_dict_cool[str(i)] = self.histo_dict_cool[str(i)] + cool[i]
                        self.histo_dict_hot[str(i)] = []
                        self.histo_dict_hot[str(i)] = self.histo_dict_hot[str(i)] + hot[i]
            else:
                raise ValueError("Unknown grouping option. Use either 'by_alpha' or 'by_t_wait'.")
                
        #Filter out False values that get put in when fitting cannot be performed
        for key in self.histo_dict_cool:
            self.histo_dict_cool[key] = np.fabs([x for x in self.histo_dict_cool[key] if x is not False])
        for key in self.histo_dict_hot:
            self.histo_dict_hot[key] = np.fabs([x for x in self.histo_dict_hot[key] if x is not False])
            
                
    def histo_maker(self,temp_choice,histo_opts,**kwargs):
        """Build histograms from hot and cool dictionaries built up by self.loader()"""
        
        #Choose hot or cool
        if temp_choice is 'cool':
            hist_dict = self.histo_dict_cool
        elif temp_choice is 'hot':
            hist_dict = self.histo_dict_hot
        else:
            raise ValueError("Invalid choice of histogram dictionary.")

        #Set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        #Initialize y-limits values to find max values
        ylims_final = [0.0,0.0]
        #Loop over histograms
        for key in hist_dict:
            if len(hist_dict[key]) < 10:
                pass
            else:
                ax.hist(hist_dict[key], self.freedman_diaconis(hist_dict[key]), histtype='step',**histo_opts[key])
                ylims = ax.get_ylim()
                if ylims[1] > ylims_final[1]:
                    ylims_final[1] = ylims[1]
                if ylims[0] < ylims_final[0]:
                    ylims_final[0] = ylims[0]
            
        #Labels and styling
        #Check for normalization in just one set; assumed all or none are normed
        if 'normed' in list(histo_opts.values())[0] and list(histo_opts.values())[0]['normed'] is True:
            ax.set_ylabel(r'$\mathrm{Normalized}$ $\mathrm{Frequency}$',fontsize=self.fs)            
        else:
            ax.set_ylabel(r'$\mathrm{Frequency}$',fontsize=self.fs)
        ax.set_ylim(ylims_final)
        ax.set_yticks(self.tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fs)
        if 'x_limits' in kwargs:
            ax.set_xlim(kwargs['x_limits'])
        if temp_choice is 'cool':
            ax.set_xlabel(r'$a$',fontsize=self.fs)
            ax.axvline(x=2,color='k',linestyle='-.',linewidth=2)
            ax.axvline(x=5,color='k',linestyle='-.',linewidth=2)
        else:
            ax.set_xlabel(r'$b$',fontsize=self.fs)
            ax.axvline(x=5.5,color='k',linestyle='-.',linewidth=2)
            
        if 'leg_off' not in kwargs or kwargs['leg_off']is False:
            if 'leg_loc' in kwargs:
                leg_loc = kwargs['leg_loc']
            else:
                leg_loc = 'best'
            if self.group is 'by_alpha':
                leg_title = r'$\alpha$'
            elif self.group is 'by_t_wait':
                leg_title = r'$T_N$ $\mathrm{(s)}$'
            leg = ax.legend(fontsize=self.alfs*self.fs,loc=leg_loc,ncol=1,title=leg_title)
            plt.setp(leg.get_title(),fontsize=self.alfs*self.fs)
        
        #Print or show figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
        
        
    def freedman_diaconis(self,hist,**kwargs):
        q75,q25 = np.percentile(hist,[75,25])
        iqr = q75 - q25
        w = 2.0*iqr*(len(hist))**(-1.0/3.0)
        return int((np.max(np.array(hist)) - np.min(np.array(hist)))/w)
        
    
    def tick_maker(self,old_ticks,n,**kwargs):
        if n < 2:
            raise ValueError('n must be greater than 1')
        
        n = n-1
        delta = (old_ticks[-1] - old_ticks[0])/n
        new_ticks = []
        for i in range(n):
            new_ticks.append(old_ticks[0] + i*delta)
        
        new_ticks.append(old_ticks[0] + n*delta)
        return new_ticks
                    
