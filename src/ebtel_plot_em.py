#ebtel_plot_em.py

#Will Barnes
#14 May 2015

#Import needed modules
import dill as pickle
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

class DEMPlotter(object):

    def __init__(self,temp_list,em_list,temp_mean,em_mean,em_sigma,cool_fits,hot_fits,**kwargs):
        #static parameters
        self.linestyles = ('-','--','-.',':')
        #check for custom parameters
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
        if 'figsize' in kwargs:
            self.figsize = kwargs['figsize']
        else:
            self.figsize = (12,12)
        #arguments
        self.temp_list = temp_list
        self.em_list = em_list
        self.temp_mean = np.array(temp_mean)
        self.em_mean = np.array(em_mean)
        self.em_sigma = np.array(em_sigma)
        self.cool_fits = cool_fits
        self.hot_fits = hot_fits
        #keyword arguments
        if 'Tn' in kwargs:
            self.Tn = kwargs['Tn']
        else:
            self.Tn = np.arange(250,5250,250)
        self.Tndelta = self.Tn[1] - self.Tn[0]


    def plot_em_curves(self,**kwargs):
        #spacing between tn curves (artificial)
        delta_em = 0.2

        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        #print lines
        for i in range(len(self.em_mean)):
            ax.plot(self.temp_mean[i], self.em_mean[i]+i*delta_em, linestyle=self.linestyles[i%len(self.linestyles)], color='black')
            if 'fit_lines' in kwargs:
                try:
                    ax.plot(kwargs['fit_lines']['t_cool'][i], (self.cool_fits[i][0]*kwargs['fit_lines']['t_cool'][i] + self.cool_fits[i][1]) + i*delta_em, linewidth=2.0, color='blue')
                except:
                    pass
                    
                try:
                    ax.plot(kwargs['fit_lines']['t_hot'][i], (self.hot_fits[i][0]*kwargs['fit_lines']['t_hot'][i] + self.hot_fits[i][1]) + i*delta_em, linewidth=2.0, color='red')
                except:
                    pass

        #set labels
        ax.set_xlabel(r'$\log T$ (K)',fontsize=self.fs)
        ax.set_ylabel(r'$\log$EM (cm$^{-5}$)',fontsize=self.fs)
        ax.set_xlim([5.5,7.5])
        ax.set_ylim([27,33])
        ax.tick_params(axis='both',labelsize=0.75*self.fs)

        #save or show the figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
        else:
            plt.show()


    def plot_em_curve(self,tn_index,**kwargs):
        #get single list
        em_list = self.em_list[tn_index]
        temp_list = self.temp_list[tn_index]

        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        #print lines
        ax.fill_between(self.temp_mean[tn_index], self.em_mean[tn_index]-self.em_sigma[tn_index], self.em_mean[tn_index]+self.em_sigma[tn_index], facecolor='red', edgecolor='red', alpha=0.35)
        for i in range(len(temp_list)):
            ax.plot(temp_list[i],em_list[i],color='blue',linestyle=self.linestyles[-1])
        ax.plot(self.temp_mean[tn_index],self.em_mean[tn_index],color='black')

        #set labels
        ax.set_title(r"EBTEL EM, $\langle T_n\rangle$ = "+str(self.Tn[tn_index])+" s",fontsize=self.fs)
        ax.set_xlabel(r'$\log T$ (K)',fontsize=self.fs)
        ax.set_ylabel(r'$\log$EM (cm$^{-5}$)',fontsize=self.fs)
        ax.set_xlim([5.5,7.5])
        ax.set_ylim([27,30])
        ax.tick_params(axis='both',labelsize=0.75*self.fs)

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
        ax.set_xlabel(r'$T_N$',fontsize=self.fs)
        ax.set_ylabel(r'$\log(T_{max})$',fontsize=self.fs)
        ax.set_ylim([5.5,7.0])
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.tick_params(axis='both',labelsize=0.75*self.fs)
        ax_twin.set_ylabel(r'$\log$EM($T_{max}$) (cm$^{-5}$)',fontsize=self.fs)
        ax_twin.set_ylim([28,30])
        ax_twin.tick_params(axis='both',labelsize=0.75*self.fs)

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
                marker_cool = ax.errorbar(self.Tn[i],self.cool_fits[i][0],yerr=self.cool_fits[i][2][0],fmt='o',color='blue',label=r'cool')

            if self.hot_fits[i][0] is not False and self.hot_fits[i][2] is not False:
                marker_hot = ax.errorbar(self.Tn[i],np.fabs(self.hot_fits[i][0]),yerr=self.hot_fits[i][2][0],fmt='o',color='red',label=r'hot')

        #set labels
        ax.set_xlabel(r'$T_N$',fontsize=self.fs)
        ax.set_ylabel(r'$a$',fontsize=self.fs)
        ax.plot([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta],[2,2],'--k')
        ax.plot([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta],[3,3],'-k')
        ax.plot([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta],[5,5],'-.k')
        ax.set_ylim([0,8])
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.set_yticks(self.tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',labelsize=0.75*self.fs)
        
        #legend
        if marker_cool and marker_hot:
            ax.legend([marker_cool,marker_hot],[r'cool',r'hot'],loc=1,fontsize=0.75*self.fs,numpoints=1)

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
            self.fn_temp = kwargs['fn_temp']
        else:
            self.fn_temp = '/data/datadrive2/EBTEL-2fluid_figs/' + species + '_heating_runs/alpha%s/ebtel_L' +str(loop_length) + '_tpulse' + str(tpulse) + '_alpha%s%s_' + species + '_heating_all_a.fits'
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
        if 'figsize' in kwargs:
            self.figsize = kwargs['figsize']
        else:
            self.figsize = (12,12)
        #Initialize dictionary to store separate histograms
        self.histo_dict_cool = {}
        self.histo_dict_hot = {}
        
            
    def loader(self,**kwargs):
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
                for i in len(cool):
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
            
                
    def histo_maker(self,temp_choice,**kwargs):
        """Build histograms from hot and cool dictionaries built up by self.loader()"""
        
        #Look for dictionary of histogram options
        if 'histo_opts' not in kwargs:
            raise ValueError("Missing histogram options for styling.")
        
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
        
        #Loop over histograms
        for key in hist_dict:
            #multiplier for uniform case
            print(key)
            if key is 'uniform':
                hist_dict[key] = 100*hist_dict[key]
            ax.hist(hist_dict[key], self.freedman_diaconis(hist_dict[key]), histtype='step',**kwargs['histo_opts'][key])# color=kwargs['histo_opts'][key]['color'], linestyle = kwargs['histo_opts'][key]['style'], label=kwargs['histo_opts'][key]['label'])
            
        #Labels
        ax.set_xlabel(r'$a$',fontsize=self.fs)
        ax.set_ylabel(r'Number of Fits',fontsize=self.fs)
        ax.set_yticks(self.tick_maker(ax.get_yticks(),5))
        ax.tick_params(axis='both',labelsize=0.75*self.fs)
        ax.legend(fontsize=0.75*self.fs,loc='best',ncol=2)
        
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
                    
