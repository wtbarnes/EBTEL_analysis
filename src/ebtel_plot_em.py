#ebtel_plot_em.py

#Will Barnes
#14 May 2015

#Import needed modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

class DEMPlotter(object):

    def __init__(self,temp_list,em_list,alpha,**kwargs):
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
        self.alpha = alpha
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
        for i in range(len(self.em_list)):
            if len(np.shape(np.array(self.em_list[i]))) > 1:
                mean_em = np.mean(self.inf_filter(self.em_list[i]),axis=0)
                temp_mean_em = np.array(mean_em)
                temp_mean_em[np.where(temp_mean_em==0.0)]=-np.float('Inf')
                mean_em = temp_mean_em
                mean_temp = np.mean(self.temp_list[i],axis=0)
                ax.plot(mean_temp,mean_em+i*delta_em,linestyle=self.linestyles[i%len(self.linestyles)],color='black')
            else:
                ax.plot(np.array(self.temp_list[i]),np.array(self.em_list[i])+i*delta_em,linestyle=self.linestyles[i%len(self.linestyles)],color='black')
                
            if 'fit_lines' in kwargs:
                try:
                    ax.plot(kwargs['fit_lines']['t_cool'],(kwargs['fit_lines']['a_cool'][i]*kwargs['fit_lines']['t_cool'] + kwargs['fit_lines']['b_cool'][i]) + i*delta_em,linewidth=2.0,color='blue')
                except:
                    pass
                try:
                    ax.plot(kwargs['fit_lines']['t_hot'],(kwargs['fit_lines']['a_hot'][i]*kwargs['fit_lines']['t_hot'] + kwargs['fit_lines']['b_hot'][i]) + i*delta_em,linewidth=2.0,color='red')
                except:
                    pass

        #set labels
        ax.set_title(r'EBTEL EM, $\alpha$ = '+str(self.alpha),fontsize=self.fs)
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
        mean_em = np.mean(em_list,axis=0)
        std_em = np.std(em_list,axis=0)
        mean_temp = np.mean(temp_list,axis=0)
        ax.fill_between(mean_temp,mean_em-std_em,mean_em+std_em,facecolor='red',edgecolor='red',alpha=0.35)
        for i in range(len(temp_list)):
            ax.plot(temp_list[i],em_list[i],color='blue',linestyle=self.linestyles[-1])
        ax.plot(mean_temp,mean_em,color='black')

        #set labels
        ax.set_title(r'EBTEL EM, $\alpha$ = '+str(self.alpha)+", $T_n$ = "+str(self.Tn[tn_index])+" s",fontsize=self.fs)
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
        ax.set_title(r'EBTEL $T(\mathrm{EM}_{\mathrm{max}})$, $\alpha$ = '+str(self.alpha),fontsize=self.fs)
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


    def plot_em_slopes(self,a_cool_mean,a_cool_std,a_hot_mean,a_hot_std,**kwargs):
        #set up figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        
        marker_cool,marker_hot = [],[]
        for i in range(len(self.Tn)):
            if a_cool_mean[i] is not False and a_cool_std[i] is not False:
                marker_cool = ax.errorbar(self.Tn[i],a_cool_mean[i],yerr=a_cool_std[i],fmt='o',color='blue',label=r'cool')

            if a_hot_mean[i] is not False and a_hot_std[i] is not False:
                marker_hot = ax.errorbar(self.Tn[i],np.fabs(a_hot_mean[i]),yerr=a_hot_std[i],fmt='o',color='red',label=r'hot')

        #set labels
        ax.set_title(r'EBTEL EM Slope',fontsize=self.fs)
        ax.set_xlabel(r'$T_N$',fontsize=self.fs)
        ax.set_ylabel(r'$a$',fontsize=self.fs)
        ax.plot([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta],[2,2],'--k')
        ax.plot([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta],[3,3],'-k')
        ax.plot([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta],[5,5],'-.k')
        ax.set_ylim([0,7])
        ax.set_xlim([self.Tn[0]-self.Tndelta,self.Tn[-1]+self.Tndelta])
        ax.tick_params(axis='both',labelsize=0.75*self.fs)
        
        #legend
        if marker_cool and marker_hot:
            ax.legend(loc=1,fontsize=0.75*self.fs,numpoints=1s)

        #save or show figure
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.format,format=self.format,dpi=self.dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def inf_filter(self,nested_list,**kwargs):
        #preallocate space
        filtered_list = []
        #filter out infs in list and set to zero for averaging
        for i in nested_list:
            temp_array = np.array(i)
            temp_array[np.where(np.isinf(temp_array)==True)]=0.0
            filtered_list.append(temp_array)
        return filtered_list

