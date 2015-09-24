#ebtel_dem_analyze_main-new.py

#Will Barnes
#18 September 2015

#Import needed libraries
import sys
import os
import argparse
import pandas as pd
import numpy as np
sys.path.append('/home/wtb2/Documents/EBTEL_analysis/src/')
import ebtel_dem as ebd
import ebtel_plot_em as ebpe

#command line argument parsing
parser = argparse.ArgumentParser(description='Script that performs DEM analysis for EBTEL-2fluid runs.')
parser.add_argument("-s","--species",help="Species to which the heating was applied for particular run.")
parser.add_argument("-L","--loop_length",type=float,help="Loop half-length (in Mm)")
parser.add_argument("-a","--alpha",help="Power-law index for amplitude distribution")
parser.add_argument("-S","--solver",help="Solver used in EBTEL")
parser.add_argument("-t","--tpulse",type=float,help="Duration of heating pulses")
parser.add_argument("--t_wait_q_scaling",help="Scaling between T_N and Q")
parser.add_argument("--root_dir",help="Root directory from which EBTEL output will be read.")
parser.add_argument("--root_dir_figs",help="Root directory to which figures will be output.")
args = parser.parse_args()

#optional command line parameters
if args.root_dir:
    root_dir = args.root_dir
else:
    root_dir = '/data/datadrive2/EBTEL-2fluid_runs/'
if args.root_dir_figs:
    root_dir_figs = args.root_dir_figs
else:
    root_dir_figs = '/data/datadrive2/EBTEL-2fluid_figs/'
if args.t_wait_q_scaling:
    t_wait_q_scaling = args.t_wait_q_scaling
    if len(t_wait_q_scaling) > 0:
        t_wait_q_scaling = '-b' + str(t_wait_q_scaling)
else:
    t_wait_q_scaling = ''

    

#figure formatting parameters
fontsize = 18
figsize = (10,10*3/4)
format = 'pdf'
dpi = 1000 

#static parameters
t_wait = np.arange(250,5250,250)
slope_limits = {'cool_lower':6.0,'cool_upper':6.6,'hot_lower':6.8,'hot_upper':7.3}
lim_method = 'dynamic'
fit_method = 'fit_all'
max_percent_drop = 0.92
delta_t = 0.5

#set directories and filenames
figdir = '%s_heating_runs/alpha%s/'
figname = 'ebtel_L%.1f_tpulse%.1f_alpha%s' + str(t_wait_q_scaling) + '_%s_heating'

#Instantiate Process class
processer = ebd.DEMProcess(root_dir,args.species,args.alpha,args.loop_length,args.tpulse,args.solver,t_wait_q_scaling=t_wait_q_scaling,verbose=True)
#Import the data
processer.import_raw(t_wait)
#Compute mean and standard deviation for EM 
processer.calc_stats()

#Instantiate Analyze class
analyzer = ebd.DEMAnalyze(processer.em, processer.temp_em, processer.em_mean, processer.temp_mean, processer.em_std, verbose=True, slope_limits=slope_limits, fit_method=fit_method, lim_method=lim_method, delta_t=delta_t, max_percent_drop=max_percent_drop)
#Filter and interpolate EM curves
analyzer.interp_and_filter()
#Fit all curves
analyzer.many_fits()

#Write all hot and cool slopes to file using pandas/numpy
fits_file=root_dir_figs + figdir%(args.species,args.alpha) + figname%(args.loop_length,args.tpulse,args.alpha,args.species) + '_all_a.fits'
temp = analyzer.cool_fits_all
for i in range(len(temp)):
    for j in range(len(temp[i])):
        if temp[i][j] is False:
            temp[i][j] = np.float('NaN')
np.savetxt(fits_file+'.cool',np.array(pd.DataFrame(temp)))
temp = analyzer.hot_fits_all
for i in range(len(temp)):
    for j in range(len(temp[i])):
        if temp[i][j] is False:
            temp[i][j] = np.float('NaN')
np.savetxt(fits_file+'.hot',np.array(pd.DataFrame(temp)))

#Check for existence of needed directories and create temp names
if not os.path.exists(root_dir_figs + figdir%(args.species,args.alpha)):
    os.makedirs(root_dir_figs + figdir%(args.species,args.alpha))
fn_temp = figdir%(args.species,args.alpha) + figname%(args.loop_length,args.tpulse,args.alpha,args.species)

#Calculate temperature vectors for printing fit lines
th = []
tc = []
for k in range(len(analyzer.hot_fits)):
    sl = analyzer.fit_limits(analyzer.temp_mean[k],analyzer.em_mean[k])
    tc.append(np.linspace(sl['cool_lower'],sl['cool_upper'],10))
    th.append(np.linspace(sl['hot_lower'],sl['hot_upper'],10))
fit_lines = {'t_cool':tc,'t_hot':th}

#Instantiate Plotter class
plotter = ebpe.DEMPlotter(processer.temp_em, processer.em, processer.temp_mean, processer.em_mean, processer.em_std, analyzer.cool_fits, analyzer.hot_fits, dpi=dpi, format=format, fs=fontsize, figsize=figsize)
#Build composite EM plot
plotter.plot_em_curves(fit_lines=fit_lines, print_fig_filename=root_dir_figs + fn_temp + '_dem')
#Build composite slope plot
plotter.plot_em_slopes(print_fig_filename=root_dir_figs + fn_temp + '_hs_compare')
#Check for existence of needed directories to construct MC curves
if args.alpha is not 'uniform':
    if not os.path.exists(root_dir_figs + fn_temp + '_dem_mc/'):
        os.makedirs(root_dir_figs + fn_temp + '_dem_mc/')
    #Build MC plots for each Tn value
    for k in range(len(t_wait)):
        plotter.plot_em_curve(k, print_fig_filename=root_dir_figs + fn_temp + '_dem_mc/' + figname%(args.loop_length,args.tpulse,args.alpha,args.species) + '_%d_dem'%k)
