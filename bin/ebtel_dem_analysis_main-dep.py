#ebtel_dem_analysis_main.py

#Will Barnes
#15 May 2015

#Import needed modules
import sys
import os
import argparse
import numpy as np
sys.path.append('/home/wtb2/Documents/EBTEL_analysis/src/')
import ebtel_dem as ebd
import ebtel_plot_em as ebpe
import ebtel_plot as ebp

#Declare parser object
parser = argparse.ArgumentParser(description='Script that performs DEM analysis for EBTEL-2fluid runs.')
#Add arguments to parser
parser.add_argument("-s","--species",help="Species to which the heating was applied for particular run.")
parser.add_argument("--root_dir",help="Root directory from which EBTEL output will be read.")
parser.add_argument("--root_dir_figs",help="Root directory to which figures will be output.")
#Declare the parser dictionary
args = parser.parse_args()

#set root directory
if args.root_dir:
    root_dir = args.root_dir
else:
    root_dir = '/data/datadrive2/EBTEL-2fluid_runs/'

if args.root_dir_figs:
    root_dir_figs = args.root_dir_figs
else:
    root_dir_figs = '/data/datadrive2/EBTEL-2fluid_figs/'
    
#set figure file format
figdir = '%s_heating_runs/alpha%s/'
figname = 'ebtel_L%.1f_tpulse%.1f_alpha%s_%s_heating'

#make parameter vectors
loop_length = np.array([20.0,40.0,60.0,80.0,100.0,120.0])
alpha = ['uniform',1.5,2.0,2.5]
#make waiting time vector
Tn = np.arange(250,5250,250)

#set limits on hot/cool slope calculation
slope_limits = {'cool_lower':6.0,'cool_upper':6.6,'hot_lower':6.7,'hot_upper':7.3}
t_cool_static = np.linspace(slope_limits['cool_lower'],slope_limits['cool_upper'],1000)
t_hot_static = np.linspace(slope_limits['hot_lower'],slope_limits['hot_upper'],1000)

#function to calculate variable hot slopes
def calc_upper_hot_lim(mean_temp,mean_em,delta_t):
    ninf_i = np.where(np.isinf(mean_em) == False) #find non-inf indices
    max_i = np.argmax(mean_em) #find index corresponding to max value
    hot_i = ninf_i[0][np.where(ninf_i[0]>max_i)] #indices for hot branch
    em_hot = mean_em[hot_i] #hot branch em
    delta_em_hot = np.fabs(np.diff(em_hot)) #delta(em) of hot branch
    delta_i = np.where(delta_em_hot>0.5)[0][0]
    lim_i = hot_i[delta_i - 1]-1
    t_upper = mean_temp[lim_i]
    t_lower = t_upper - delta_t
    return t_lower,t_upper

#set static parameters
tpulse = 100.0
solver = 'rka4'
delta_lim = 0.4

#declare instance of Plotter class
surf_plot = ebp.Plotter(format='pdf',fs=18.0,figsize=(8,8),dpi=1000)

#iterate over variable parameters
for i in range(len(alpha)):
    temp_max_save = []
    em_max_save = []
    for j in  range(len(loop_length)):
        #print status
        print("Processing L = %.1f, alpha = %s"%(loop_length[j],str(alpha[i])))
        #get data
        if loop_length[j] == 20.0 or alpha[i] == 'uniform':
            solver = 'euler'
        else:
            solver = 'rka4'
            
        dema = ebd.DEMAnalyzer(root_dir,args.species,alpha[i],loop_length[j],tpulse,solver,Tn=Tn,slope_limits=slope_limits)
        dema.process_raw()
        dema.em_statistics()
        #variable limit configuration and temperature fit array
        t_cool,t_hot = [],[]
        if alpha[i] == 'uniform' or args.species == 'ion':
            [(t_cool.append(t_cool_static),t_hot.append(t_hot_static)) for k in range(len(dema.temp_mean))]
            dema.many_slopes()
        else:
            hot_lower,hot_upper = [],[]
            cool_lower,cool_upper = [],[]
            for k in range(len(dema.temp_mean)):
                l,u = calc_upper_hot_lim(dema.temp_mean[k],dema.em_mean[k],delta_lim)
                hot_lower.append(l),hot_upper.append(u)
                cool_lower.append(slope_limits['cool_lower']),cool_upper.append(slope_limits['cool_upper'])
                t_cool.append(t_cool_static)
                t_hot.append(np.linspace(l,u,len(t_cool_static)))
            
            dema.many_slopes(slope_limits={'cool_lower':cool_lower,'cool_upper':cool_upper,'hot_lower':hot_lower,'hot_upper':hot_upper})
            
        dema.slope_statistics()
        dema.find_em_max()
        temp_max_save.append([np.mean(tmax) for tmax in dema.temp_max])
        em_max_save.append([np.mean(emmax) for emmax in dema.em_max])
        #build figure names and make sure directories exist
        if not os.path.exists(root_dir_figs + figdir%(args.species,str(alpha[i]))):
            os.makedirs(root_dir_figs + figdir%(args.species,str(alpha[i])))
            
        figname_temp = figdir%(args.species,str(alpha[i]))+figname%(loop_length[j],tpulse,str(alpha[i]),args.species)
        #plot data
        demp = ebpe.DEMPlotter(dema.temp_em,dema.em,alpha[i],Tn=Tn,format='pdf',fs=18.0,figsize=(8,8),dpi=1000)
        demp.plot_em_max(dema.temp_max,dema.em_max,print_fig_filename=root_dir_figs + figname_temp + '_TmaxVTn')
        demp.figsize = (demp.figsize[0],3/4*demp.figsize[1])
        demp.plot_em_slopes(dema.a_cool_mean,dema.a_cool_std,dema.a_hot_mean,dema.a_hot_std,print_fig_filename=root_dir_figs + figname_temp + '_hs_compare')
        demp.figsize = (demp.figsize[0],4/3*demp.figsize[1])
        fit_lines = {'t_cool':t_cool,'a_cool':dema.a_cool_mean,'b_cool':dema.b_cool_mean,'t_hot':t_hot,'a_hot':dema.a_hot_mean,'b_hot':dema.b_hot_mean}
        demp.plot_em_curves(dema.temp_mean,dema.em_mean,fit_lines=fit_lines,print_fig_filename=root_dir_figs+figname_temp+'_dem')
        #plot all em curves for given tn
        if alpha[i] is not 'uniform':
            if not os.path.exists(root_dir_figs+figname_temp+'_dem_mc/'):
                os.makedirs(root_dir_figs+figname_temp+'_dem_mc/')
                
            for k in range(len(Tn)):
                demp.plot_em_curve(k,print_fig_filename=root_dir_figs + figname_temp + '_dem_mc/' + figname%(loop_length[j],tpulse,str(alpha[i]),args.species) + '_'+str(k) + '_dem')
                
    #build surface plot
    surf_plot.plot_surface(Tn,loop_length,temp_max_save,vmin=6.0,vmax=6.8,ylab=r'$L$ (Mm)',xlab=r'$T_n$ (s)',plot_title=r'$T_{max}$ Surface, $\alpha=$' + str(alpha[i]),print_fig_filename=root_dir_figs + figdir%(args.species,str(alpha[i])) + 't_max_surface_' + args.species + '_alpha' + str(alpha[i]) + '_tpulse' + str(tpulse) + '_' + solver)
    surf_plot.plot_surface(Tn,loop_length,em_max_save,vmin=26.0,vmax=30.0,ylab=r'$L$ (Mm)',xlab=r'$T_n$ (s)',plot_title=r'EM$_{max}$ Surface, $\alpha=$' + str(alpha[i]),print_fig_filename=root_dir_figs + figdir%(args.species,str(alpha[i])) + 'em_max_surface_' + args.species + '_alpha' + str(alpha[i]) + '_tpulse' + str(tpulse) + '_' + solver)