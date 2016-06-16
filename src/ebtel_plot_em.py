"""
Plot EM and EM slope for many EBTEL runs.
"""

import os
import logging
import pickle
import itertools

import numpy as np
from astroML import density_estimation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit


def tick_maker(old_ticks,n,**kwargs):
    if n < 2:
        raise ValueError('n must be greater than 1')
    
    n = n-1
    delta = (old_ticks[-1] - old_ticks[0])/n
    new_ticks = []
    for i in range(n):
        new_ticks.append(old_ticks[0] + i*delta)
    
    new_ticks.append(old_ticks[0] + n*delta)
    return new_ticks
    

class DEMPlotter(object):

    def __init__(self, em_res_top_dir, em_stats, diagnostics, diagnostics_stats, Tn = np.arange(250,5250,250), dpi=1000, fontsize=18., figsize=(8,8), alfs=0.75, fformat='eps', **kwargs):
        #set up logger
        self.logger = logging.getLogger(type(self).__name__)
        #arguments
        self.em_res_top_dir = em_res_top_dir
        self.em_stats = em_stats
        self.diagnostics = diagnostics
        self.diagnostics_stats = diagnostics_stats
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
            if self.diagnostics_stats[i]['cool'] is not None:
                t = np.array([self.diagnostics_stats[i]['cool']['limits'][0],self.diagnostics_stats[i]['cool']['limits'][1]])
                ax.plot(t,10.**(i*delta_em)*10.**self.diagnostics_stats[i]['cool']['b']*t**self.diagnostics_stats[i]['cool']['a'],color=sns.color_palette('bright')[0], linewidth=2, linestyle='solid')
            if self.diagnostics_stats[i]['hot'] is not None:
                t = np.array([self.diagnostics_stats[i]['hot']['limits'][0],self.diagnostics_stats[i]['hot']['limits'][1]])
                ax.plot(t,10.**(i*delta_em)*10.**self.diagnostics_stats[i]['hot']['b']*t**self.diagnostics_stats[i]['hot']['a'],color=sns.color_palette('bright')[2], linewidth=2, linestyle='solid')

        #plot options
        ax.set_xlabel(r'$T$ $\mathrm{(K)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$\mathrm{EM}$ $\mathrm{(cm}^{-5}\mathrm{)}$',fontsize=self.fontsize)
        ax.set_xlim([10**5.5,10**7.5])
        ax.set_ylim(y_limits)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        #legend
        leg = ax.legend(loc=2,fontsize=self.alfs*self.fontsize,title=r'$t_N$ $\mathrm{(s)}$',ncol=2,bbox_to_anchor=(-0.1,1.07))
        plt.setp(leg.get_title(),fontsize=self.alfs*self.fontsize)
        #avoid cutting off labels
        plt.tight_layout()

        #save or show the figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi,bbox_extra_artists=[leg],bbox_inches='tight')
        else:
            plt.show()
            
            
    def plot_em_curves_grid(self, y_limits=[10**26.,10**32.], num_cols=4, print_fig_filename=None, **kwargs):
            """Make subplot grid of EM curves"""
            
            num_rows=int(np.ceil(len(self.Tn)/num_cols))
            fig,axes=plt.subplots(num_rows,num_cols,figsize=(1.5*self.figsize[0],1.5*self.figsize[1]),sharex=True,sharey=True)
            plt.subplots_adjust(hspace=0.0,wspace=0.0,left=0.05,bottom=0.05)
            
            for ax,i in zip(axes.flatten(),range(len(self.Tn))):
                #plot histograms and mean
                self._make_em_axes(ax,self.Tn[i])
                #fit lines
                if self.diagnostics_stats[i]['cool'] is not None:
                    t = np.array([self.diagnostics_stats[i]['cool']['limits'][0],self.diagnostics_stats[i]['cool']['limits'][1]])
                    ax.plot(t,10.**self.diagnostics_stats[i]['cool']['b']*t**self.diagnostics_stats[i]['cool']['a'],color=sns.color_palette('bright')[0], linewidth=2, linestyle='solid')
                if self.diagnostics_stats[i]['hot'] is not None:
                    t = np.array([self.diagnostics_stats[i]['hot']['limits'][0],self.diagnostics_stats[i]['hot']['limits'][1]])
                    ax.plot(t,10.**self.diagnostics_stats[i]['hot']['b']*t**self.diagnostics_stats[i]['hot']['a'],color=sns.color_palette('bright')[2], linewidth=2, linestyle='solid')
                #text
                ax.text(0.7,0.9,r'$t_N=%d$ $\mathrm{s}$'%(self.Tn[i]),transform=ax.transAxes,fontsize=self.alfs*self.fontsize)
                if self.diagnostics_stats[i]['cool'] is not None:
                    ax.text(0.12,0.9,r'$a=%.3f$'%(self.diagnostics_stats[i]['cool']['a']), fontsize=self.alfs*self.fontsize, transform=ax.transAxes, color=sns.color_palette('bright')[0], ha='left', va='center')
                if self.diagnostics_stats[i]['hot'] is not None:
                    ax.text(0.12,0.8,r'$b=%.3f$'%(np.fabs(self.diagnostics_stats[i]['hot']['a'])), fontsize=self.alfs*self.fontsize, transform=ax.transAxes, color=sns.color_palette('bright')[2], ha='left', va='center')
                #limits and scale
                ax.set_xlim([10**5.5,10**7.5])
                ax.set_ylim(y_limits)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
            
            #labels    
            fig.text(0.5, 0.005, r'$T$ $\mathrm{(K)}$', ha='center', va='center',fontsize=self.fontsize) #xlabel
            fig.text(0.005, 0.5, r'$\mathrm{EM}$ $\mathrm{(cm}^{-5}\mathrm{)}$', ha='center', va='center', rotation='vertical',fontsize=self.fontsize) #ylabel
                            
            #save or show figure
            if print_fig_filename is not None:
                plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
                plt.close('all')
            else:
                plt.show()
            
            
    def _make_em_axes(self,ax,tn_val):
        """Plot single em curve"""
        
        if tn_val not in self.Tn:
            raise ValueError("Invalid wait time %f. Use a valid value from self.Tn"%(tn_val))
            
        tn_index = np.where(self.Tn==tn_val)[0][0]
            
        #standard deviation
        ax.fill_between(self.em_stats[tn_index]['T_mean'], self.em_stats[tn_index]['em_mean']-self.em_stats[tn_index]['em_std'], self.em_stats[tn_index]['em_mean']+self.em_stats[tn_index]['em_std'], facecolor=sns.color_palette('deep')[2], edgecolor=sns.color_palette('deep')[2], alpha=0.75)
        #MC histograms
        for pfile in os.listdir(os.path.join(self.em_res_top_dir,'tn%d'%tn_val)):
            with open(os.path.join(self.em_res_top_dir,'tn%d'%tn_val,pfile),'rb') as f:
                h=pickle.load(f)
            ax.hist(h['T'], bins=h['bins'], weights=h['em'], histtype='step', color=sns.color_palette('deep')[0], linestyle=self.linestyles[-1], alpha=0.1)
        #mean
        ax.plot(self.em_stats[tn_index]['T_mean'], self.em_stats[tn_index]['em_mean'], color='black', linestyle=self.linestyles[-1], linewidth=2)


    def plot_em_curve(self, tn_val, y_limits=[10**25.,10**28.], print_fig_filename=None, **kwargs):
        """Plot MC and mean EM curves for single value of Tn"""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        self._make_em_axes(ax,tn_val)
        
        #set labels
        ax.set_xlabel(r'$T$ $\mathrm{(K)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$\mathrm{EM}$ $\mathrm{(cm}^{-5}\mathrm{)}$',fontsize=self.fontsize)
        ax.set_xlim([10**5.5,10**7.5])
        ax.set_ylim(y_limits)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()


    def plot_em_max(self, t_ref_line=4e+6, y_limits_em=[10**26.,10**28.], y_limits_t=[10**6.,10**7.], print_fig_filename=None, **kwargs):
        """Plot max(EM) and corresponding temperature with error bars"""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        ax_twin = ax.twinx()

        for tw,ems in zip(self.Tn,self.em_stats):
            #plot points
            ax.errorbar(tw,ems['T_max_mean'],yerr=ems['T_max_std'],fmt='o',color='black')
            ax_twin.errorbar(tw,ems['em_max_mean'],yerr=ems['em_max_std'],fmt='*',color='black')

        #set labels
        ax.set_xlabel(r'$t_N$ $\mathrm{(s)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$T_m$ $\mathrm{(K)}$',fontsize=self.fontsize)
        ax.axhline(y=t_ref_line,color='black',linestyle=':')
        ax.set_ylim(y_limits_t)
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.set_yscale('log')
        ax.tick_params(axis='both',labelsize=self.alfs*self.fontsize)
        ax_twin.set_ylabel(r'$\mathrm{EM}_{\mathrm{max}}$ $\mathrm{(cm}^{-5}\mathrm{)}$',fontsize=self.fontsize)
        ax_twin.set_ylim(y_limits_em)
        ax_twin.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax_twin.set_yscale('log')
        ax_twin.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()


    def plot_em_slopes(self, y_limits=[0,8], print_fig_filename=None, **kwargs):
        """Plot cool and hot slopes versus heating frequency with error bars"""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        marker_cool,marker_hot = [],[]
        for tw,fs in zip(self.Tn,self.diagnostics_stats):
            if fs['cool'] is not None:
                marker_cool = ax.errorbar(tw,fs['cool']['a'],yerr=fs['cool']['sigma_a'],fmt='o',color=sns.color_palette('deep')[0],label=r'$\mathrm{cool}$')
            if fs['hot'] is not None:
                marker_hot = ax.errorbar(tw,np.fabs(fs['hot']['a']),yerr=fs['hot']['sigma_a'],fmt='o',color=sns.color_palette('deep')[2],label=r'$\mathrm{hot}$')

        #set labels
        ax.set_xlabel(r'$t_N$ $\mathrm{(s)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$\mathrm{EM}$ $\mathrm{slope}$',fontsize=self.fontsize)
        ax.axhline(y=2,color='k',linestyle=':')
        ax.axhline(y=2.5,color='k',linestyle=':')
        ax.axhline(y=3,color='k',linestyle=':')        
        ax.axhline(y=5.5,color='k',linestyle=':')
        ax.set_ylim(y_limits)
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.set_yticks(tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        
        #legend
        if marker_cool and marker_hot:
            ax.legend([marker_cool,marker_hot],[r'$a$, $\mathrm{cool}$',r'$b$, $\mathrm{hot}$'],loc=1,fontsize=self.alfs*self.fontsize,numpoints=1)
            
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def plot_em_ratios(self, y_limits=[0,2], print_fig_filename=None, legend_labels=None, **kwargs):
        """Plot EM(T_hot)/EM(T_cool) ratio as a function of Tn"""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        for tw,fs in zip(self.Tn,self.diagnostics_stats):
            for i in range(len(fs['ratio']['tpairs'])):
                ax.errorbar(tw, fs['ratio']['mean'][i], yerr=fs['ratio']['sigma'][i], fmt='o', color=sns.color_palette('deep')[i], 
                label=r'$T_c=%.2f$ $\mathrm{MK}$, $T_h=%.2f$ $\mathrm{MK}$'%(fs['ratio']['tpairs'][i][0]/1e+6,fs['ratio']['tpairs'][i][1]/1e+6))
            
        #set labels
        ax.set_xlabel(r'$t_N$ $\mathrm{(s)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$\mathrm{EM}$ $\mathrm{ratio}$',fontsize=self.fontsize)
        ax.set_ylim(y_limits)
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.set_yticks(tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        
        #legend
        handles,labels=ax.get_legend_handles_labels()
        leghand=handles[0:i+1]
        if legend_labels and len(legend_labels)==len(leghand):
            leglab=legend_labels
        else:
            leglab=labels[0:i+1]
        ax.legend(leghand,leglab,fontsize=self.alfs*self.fontsize, numpoints=1, loc=2)
        
        #avoid cutting off labels
        plt.tight_layout()

        #save or show figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def plot_em_derivs(self, em_cutoff=1e+23, y_limits=[-10,6], print_fig_filename=None, **kwargs):
        """Plot d(log(EM))/d(log(T)) as a function of T for all values of Tn"""
        
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        i = 0
        #Iterate over t_wait values
        for ems in self.em_stats:
            #filter and compute derivative 
            em_tmp = np.log10(ems['em_mean'][ems['em_mean']>em_cutoff]) 
            t_tmp = np.log10(ems['T_mean'][ems['em_mean']>em_cutoff])
            dem_dt = np.gradient(em_tmp,np.gradient(t_tmp))
            #plotting
            ax.plot(t_tmp, dem_dt, color=self.colors[i], linestyle=self.linestyles[i%len(self.linestyles)])
            #increment counter
            i += 1
            
        #set labels
        ax.set_xlabel(r'$\log{T}$ $\mathrm{(K)}$',fontsize=self.fontsize)
        ax.set_ylabel(r'$d\log{\mathrm{EM}}/d\log{T}$',fontsize=self.fontsize)
        ax.axhline(y=2,color='k',linestyle=':')
        ax.axhline(y=3,color='k',linestyle=':')
        ax.axhline(y=-2.5,color='k',linestyle=':')
        ax.axhline(y=-5.5,color='k',linestyle=':')
        ax.set_ylim(y_limits)
        ax.set_xlim([5.5,7.5])
        ax.set_yticks(tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        #avoid cutting off labels
        plt.tight_layout()
        
        #save or show figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
        
        
        
class EMHistoBuilder(object):
    """Class to build histograms of slope values to compare across heating functions and heating frequencies"""
    
    def __init__(self,files=[],labels=[], group='by_alpha', fontsize=18., figsize=(8,8), alfs=0.75, fformat='eps', dpi = 1000, **kwargs):
        
        #configure logger
        self.logger = logging.getLogger(type(self).__name__)
        #Load in heating function values and set needed variables
        self.files = files
        self.labels = labels
        if len(files) != len(labels):
            raise ValueError("File list and label list must be of equal length.")
        elif len(files) == 1:
            self.logger.warning("Only one data set specified.")
        #keyword options
        self.group = group
        self.dpi = dpi
        self.fformat = fformat
        self.fontsize = fontsize
        self.alfs = alfs
        self.figsize = figsize
        #Initialize dictionary to store separate histograms
        self.histo_dict = {}
        self.histo_dict['cool'],self.histo_dict['hot'],self.histo_dict['ratio'] = {},{},{}
        
            
    def load_diagnostics(self, t_wait_interval=0, t_wait_length=20, ratio_index=0, include_nei=False, **kwargs):
        """Load in data and create dictionaries with slope values grouped according to 'group' option"""
        
        #Loop over (alpha,b) values 
        for fn,lab in zip(self.files,self.labels):
            #Unpickle the file
            with open(fn,'rb') as f: 
                fits_dict = pickle.load(f)
            #Group by alpha method
            if self.group is 'by_alpha':
                self.histo_dict['cool'][lab] = [np.fabs(d['cool']['a']) for d in list(itertools.chain(*fits_dict)) if d['cool'] is not None]
                self.histo_dict['hot'][lab] = [np.fabs(d['hot']['a']) for d in list(itertools.chain(*fits_dict)) if d['hot'] is not None]
                self.histo_dict['ratio'][lab] = [d['ratio'][ratio_index] for d in list(itertools.chain(*fits_dict)) if d['ratio'][ratio_index] is not None]
            #Group by Tn method
            elif self.group is 'by_t_wait':
                for i in np.arange(0 + t_wait_interval,t_wait_length+t_wait_interval,1+t_wait_interval):
                    try:
                        if include_nei and lab.lower() == 'nei':
                            self.histo_dict['cool'][str(i)+'-nei'].extend([np.fabs(d['cool']['a']) for d in fits_dict[i] if d['cool'] is not None])
                            self.histo_dict['hot'][str(i)+'-nei'].extend([np.fabs(d['hot']['a']) for d in fits_dict[i] if d['hot'] is not None])
                            self.histo_dict['ratio'][str(i)+'-nei'].extend([d['ratio'][ratio_index] for d in fits_dict[i] if d['ratio'][ratio_index] is not None])
                        else:
                            self.histo_dict['cool'][str(i)].extend([np.fabs(d['cool']['a']) for d in fits_dict[i] if d['cool'] is not None])
                            self.histo_dict['hot'][str(i)].extend([np.fabs(d['hot']['a']) for d in fits_dict[i] if d['hot'] is not None])
                            self.histo_dict['ratio'][str(i)].extend([d['ratio'][ratio_index] for d in fits_dict[i] if d['ratio'][ratio_index] is not None])
                    except KeyError:
                        if include_nei and lab.lower() == 'nei':
                            self.histo_dict['cool'][str(i)+'-nei'] = [np.fabs(d['cool']['a']) for d in fits_dict[i] if d['cool'] is not None]
                            self.histo_dict['hot'][str(i)+'-nei'] = [np.fabs(d['hot']['a']) for d in fits_dict[i] if d['hot'] is not None]
                            self.histo_dict['ratio'][str(i)+'-nei'] = [d['ratio'][ratio_index] for d in fits_dict[i] if d['ratio'][ratio_index] is not None]
                        else:
                            self.histo_dict['cool'][str(i)] = [np.fabs(d['cool']['a']) for d in fits_dict[i] if d['cool'] is not None]
                            self.histo_dict['hot'][str(i)] = [np.fabs(d['hot']['a']) for d in fits_dict[i] if d['hot'] is not None]
                            self.histo_dict['ratio'][str(i)] = [d['ratio'][ratio_index] for d in fits_dict[i] if d['ratio'][ratio_index] is not None]
            else:
                raise ValueError("Unknown grouping option. Use either 'by_alpha' or 'by_t_wait'.")
            
                
    def make_fit_histogram(self, temp_choice, histo_opts={}, x_limits=None, y_limits=None, leg=False, leg_loc=None, bin_tool='freedman', print_fig_filename=None, ncols_leg=1,min_stats = 10,**kwargs):
        """Build histograms from hot and cool dictionaries built up by self.loader()"""

        #Set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        #set minimum number of statistics to show result
        if self.group is not 'by_alpha':
            min_stats=20*len(self.files)
        #Initialize y-limits values to find max values
        ylims_final = [0.0,0.0]
        #Loop over histograms
        for key in self.histo_dict[temp_choice]:
            if len(self.histo_dict[temp_choice][key]) < min_stats:
                pass
            else:
                ax.hist(self.histo_dict[temp_choice][key], bins=self._size_bins(self.histo_dict[temp_choice][key],bin_tool), histtype='step',**histo_opts[key])
                ylims = ax.get_ylim()
                if ylims[1] > ylims_final[1]:
                    ylims_final[1] = ylims[1]
                if ylims[0] < ylims_final[0]:
                    ylims_final[0] = ylims[0]
            
        #Labels and styling
        if temp_choice == 'cool':
            ax.set_xlabel(r'$a$',fontsize=self.fontsize)
        elif temp_choice == 'hot':
            ax.set_xlabel(r'$b$',fontsize=self.fontsize)
        else:
            ax.set_xlabel(r'$\mathrm{EM}\,\,\mathrm{ratio}$',fontsize=self.fontsize)
        #Check for normalization in just one set; assumed all or none are normed
        if 'normed' in list(histo_opts.values())[0] and list(histo_opts.values())[0]['normed'] is True:
            ax.set_ylabel(r'$\mathrm{Normalized}$ $\mathrm{Frequency}$',fontsize=self.fontsize)            
        else:
            ax.set_ylabel(r'$\mathrm{Frequency}$',fontsize=self.fontsize)
        if y_limits is None:
            ax.set_ylim(ylims_final)
        else:
            ax.set_ylim(y_limits)
        ax.set_yticks(tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',pad=8,labelsize=self.alfs*self.fontsize)
        if x_limits is not None:
            ax.set_xlim(x_limits)
        if leg:
            if leg_loc is None:
                leg_loc = 'best'
            if self.group == 'by_alpha':
                leg_title = r'$\alpha$'
            else:
                leg_title = r'$t_N$ $\mathrm{(s)}$'
            handles,labels = ax.get_legend_handles_labels()
            short_hand,short_lab = [],[]
            for hand,lab in zip(handles,labels):
                if lab.lower() != 'nei':
                    short_hand.append(hand)
                    short_lab.append(lab) 
            #sort legend entries
            short_lab,short_hand = zip(*sorted(zip(short_lab,short_hand), key=lambda t: t[0]))
            leg = ax.legend(short_hand,short_lab,fontsize=self.alfs*self.fontsize,loc=leg_loc,ncol=ncols_leg,title=leg_title)
            plt.setp(leg.get_title(),fontsize=self.alfs*self.fontsize)
        
        #Print or show figure
        if print_fig_filename is not None:
            plt.savefig(print_fig_filename+'.'+self.fformat,format=self.fformat,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
        
        
    def _size_bins(self,hist,bin_tool,**kwargs):
        """Use astroML routines to choose bin edges"""
        
        if bin_tool == 'freedman':
            dx,bins = density_estimation.freedman_bin_width(hist,return_bins=True)
        elif bin_tool == 'scotts':
            dx,bins = density_estimation.scotts_bin_width(hist,return_bins=True)
        elif bin_tool == 'knuth':
            dx,bins = density_estimation.knuth_bin_width(hist,return_bins=True)
        else:
            self.logger.warning("Unrecognized bin_tool option. Using Freedman-Diaconis rule.")
            dx,bins = density_estimation.freedman_bin_width(hist,return_bins=True)
            
        return bins
        
                    

def make_top_em_grid(files=[],labels=[],colors=sns.color_palette('deep'),linestyles=[],tw_select=np.arange(250,5250,250),nrows=5,ncols=4, fontsize=18., figsize=(8,8), alfs=0.75, fformat='eps', dpi = 1000, xlims=[10**5.5,10**7.5], ylims=[10**26.,10**29.], xlab_pos=[0.5, 0.005], ylab_pos=[0.005, 0.5], show_sigma=False, show_leg=[], print_fig_filename=None):
    """Plot a grid of EM curves for a select number of waiting times"""
    
    #input check
    if len(files) != len(labels):
        raise ValueError('Length of label list and length of file list must be equal.')
        
    if not linestyles:
        linestyles = len(labels)*['solid']
    if not show_leg:
        show_leg = len(labels)*[True]
    
    #get needed indices (NOTE: assuming wait times are from this selection)
    waiting_times = np.arange(250,5250,250)
    try:
        i_tws = [np.where(waiting_times==tws)[0][0] for tws in tw_select]
    except IndexError:
        logging.exception('Invalid wait time selection.')
        
    #prep dictionaries
    em_dict = {}
    for tws in tw_select:
        em_dict[str(tws)] = {}

    #load the data
    for df,l in zip(files,labels):
        with open(df,'rb') as f:
            ems,emb = pickle.load(f)
        for i,tws in zip(i_tws,tw_select):
            em_dict[str(tws)][l] = ems[i]
            
    #plotting
    fig,axes = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)
    for ax,i_ax,tws in zip(axes.flatten(),range(len(axes.flatten())),tw_select):
        for l,i in zip(labels,range(len(labels))):
            ax.plot(em_dict[str(tws)][l]['T_mean'], em_dict[str(tws)][l]['em_mean'], color=colors[i], linestyle=linestyles[i], label=l)
            if show_sigma:
                ax.fill_between(em_dict[str(tws)][l]['T_mean'], em_dict[str(tws)][l]['em_mean']-em_dict[str(tws)][l]['em_std'], em_dict[str(tws)][l]['em_mean']+em_dict[str(tws)][l]['em_std'], facecolor=sns.color_palette('deep')[i], edgecolor=sns.color_palette('deep')[i], alpha=0.25)
        #plot options
        ax.text(0.7, 0.85, r'$t_N=%d$ $\mathrm{s}$'%tws, fontsize=alfs*fontsize, transform = ax.transAxes)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.tick_params(axis='both',pad=8,labelsize=alfs*fontsize)
        #if i_ax < ncols*(nrows-1):
        #    ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
        #if i_ax >= ncols:
        #    ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))
        
    #axes labels
    fig.text(xlab_pos[0], xlab_pos[1], r'$T$ $\mathrm{(K)}$', ha='center', va='center',fontsize=fontsize) #xlabel
    fig.text(ylab_pos[0], ylab_pos[1], r'$\mathrm{EM}$ $\mathrm{(cm}^{-5}\mathrm{)}$', ha='center', va='center', rotation='vertical',fontsize=fontsize) #ylabel
    #legend
    handles,labels = axes.flatten()[0].get_legend_handles_labels()
    short_handles,short_labels = [],[]
    for hand,lab,lb in zip(handles,labels,show_leg):
        if lb:
            short_handles.append(hand)
            short_labels.append(lab.replace(', $\mathrm{IEQ}$',''))
    axes.flatten()[0].legend(short_handles, short_labels, loc='best', fontsize=alfs*fontsize)
    
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    
    #save or show figure
    if print_fig_filename is not None:
        plt.savefig(print_fig_filename+'.'+fformat,format=fformat,dpi=dpi)
        plt.close('all')
    else:
        plt.show()