#ebtel_dem_analyze_main-new.py

#Will Barnes
#18 September 2015

#Import needed libraries
import sys
import os
import logging
import argparse
import pickle
import numpy as np
sys.path.append(os.path.join('/home/wtb2/Documents','EBTEL_analysis/src/'))
import ebtel_dem as ebd
import ebtel_plot_em as ebpe

#command line argument parsing
parser = argparse.ArgumentParser(description='Script that performs DEM analysis for EBTEL-2fluid runs.')
parser.add_argument("-s","--species",help="Species to which the heating was applied for particular run.")
parser.add_argument("-L","--loop_length",type=float,help="Loop half-length (in Mm)")
parser.add_argument("-a","--alpha",help="Power-law index for amplitude distribution")
parser.add_argument("-S","--solver",help="Solver used in EBTEL")
parser.add_argument("-t","--tpulse",type=float,help="Duration of heating pulses")
parser.add_argument("--t_wait_q_scaling",help="Scaling between T_N and Q",default='')
parser.add_argument("--root_dir",help="Root directory from which EBTEL output will be read.",default='/data/datadrive2/EBTEL_runs')
parser.add_argument("--root_dir_figs",help="Root directory to which figures will be output.",default='/data/datadrive2/EBTEL_figs')
parser.add_argument("--read_teff",help="Flag for reading results from IonPopSolver code.", action='store_true')
args = parser.parse_args()

#optional command line parameters
if args.t_wait_q_scaling:
    args.t_wait_q_scaling = '-b'+str(args.t_wait_q_scaling)

#figure formatting parameters
fontsize = 22
format = 'pdf'

#static parameters
t_wait = np.arange(250,5250,250)
cool_limits = [10**6.0,10**6.5]
t_ratio_cool=[10**6.3,1.6e+6,10**6.3]
t_ratio_hot=[10**7.1,8.9e+6,8.9e+6] #Fe XV EIS line, Fe XX-XXII MaGIXS lines

#set directories and filenames
figdir = '%s_heating_runs/alpha%s'
figname = 'ebtel_L%.1f_tpulse%.1f_alpha%s' + str(args.t_wait_q_scaling) + '_%s_heating'

#set up logger
logging.basicConfig(stream=sys.stdout,level=logging.INFO)

#Check for existence of needed directories and create temp names
if not os.path.exists(os.path.join(args.root_dir_figs, figdir%(args.species,args.alpha))):
    os.makedirs(os.path.join(args.root_dir_figs, figdir%(args.species,args.alpha)))
fn_temp = os.path.join(figdir%(args.species,args.alpha), figname%(args.loop_length,args.tpulse,args.alpha,args.species))

#Instantiate Process class
processor = ebd.DEMProcess(args.root_dir, args.root_dir_figs, figname%(args.loop_length,args.tpulse,args.alpha,args.species), t_wait, args.species, args.alpha, args.loop_length, args.tpulse, args.solver, scaling_suffix=args.t_wait_q_scaling, aspect_ratio_factor=10.0, em_peak_falloff=0.7,em_cutoff=1e+24)
#Import the data
if not os.path.exists(processor.em_res_top_dir):
    logging.info("Calculating level 1 EM results and saving to %s. This may take a while..."%(processor.em_res_top_dir))
    processor.import_raw(read_teff=args.read_teff)    

#Statistics and diagnostics
processor.calc_stats()
processor.diagnose_em(cool_limits=cool_limits,t_ratio_cool=t_ratio_cool,t_ratio_hot=t_ratio_hot)
processor.calc_diagnostic_stats()

#Pickle results for building histograms later
with open(os.path.join(args.root_dir_figs, figdir%(args.species,args.alpha), figname%(args.loop_length,args.tpulse,args.alpha,args.species) + '.lvl2_diagnostics.pickle'),'wb') as f:
    pickle.dump(processor.diagnostics,f)

#Instantiate Plotter class
plotter = ebpe.DEMPlotter(processor.em_res_top_dir, processor.em_stats, processor.diagnostics, processor.diagnostics_stats, fformat=format, fontsize=fontsize, alfs=0.65)
plotter.plot_em_curves(print_fig_filename=os.path.join(args.root_dir_figs, fn_temp + '.em_all'))
plotter.plot_em_ratios(print_fig_filename=os.path.join(args.root_dir_figs, fn_temp + '.ratios'))
#Shorten figure size for next two plots
plotter.figsize=(plotter.figsize[0],plotter.figsize[1]/2.0)
#Build remaining plots
plotter.plot_em_slopes(print_fig_filename=os.path.join(args.root_dir_figs, fn_temp + '.slopes'))
plotter.plot_em_derivs(print_fig_filename=os.path.join(args.root_dir_figs, fn_temp + '.derivs'))
plotter.plot_em_max(y_limits_t=[10.**5.5,10.**7.5],print_fig_filename=os.path.join(args.root_dir_figs, fn_temp + '.max'))
#Check for existence of needed directories to construct MC curves
if args.alpha != 'uniform':
    if not os.path.exists(os.path.join(args.root_dir_figs, fn_temp + '_em_mc/')):
        os.makedirs(os.path.join(args.root_dir_figs, fn_temp + '_em_mc/'))
    #Build MC plots for each Tn value
    for tw in t_wait:
        plotter.plot_em_curve(tw, print_fig_filename=os.path.join(args.root_dir_figs, fn_temp + '_em_mc/', figname%(args.loop_length,args.tpulse,args.alpha,args.species) + '_tn%d_em'%(tw)))
        
