#ebtel_dem_histo_main.py

#Will Barnes
#24 September 2015

#Import needed libraries
import sys
import argparse
import numpy as np
sys.path.append('/home/wtb2/Documents/EBTEL_analysis/src/')
import ebtel_plot_em as ebpe
import seaborn.apionly as sns

#Argument parser
parser = argparse.ArgumentParser(description='Script that performs DEM analysis for EBTEL-2fluid runs.')
parser.add_argument("-s","--species",help="Species to which the heating was applied for particular run.")
parser.add_argument("-L","--loop_length",type=float,help="Loop half-length (in Mm)")
parser.add_argument("-t","--tpulse",type=float,help="Duration of heating pulses")
parser.add_argument("--root_dir",help="Root directory from which slope values will be read.")
args = parser.parse_args()

if args.root_dir:
    root_dir = args.root_dir
else:
    root_dir = '/data/datadrive2/EBTEL-2fluid_figs/'

#figure formatting parameters
fontsize = 18
figsize = (10,10*3/4)
format = 'pdf'
dpi = 1000
linewidth = 3
fn_temp = root_dir + '%s_heating_runs/'%(args.species) + 'ebtel_L%.1f_tpulse%.1f_%s_heating'%(args.loop_length,args.tpulse,args.species) + '_all_hist'

#Build colorpalette
xkcd_cols = ['black','windows blue','medium green','fire engine red','barney purple']
cp = sns.xkcd_palette(xkcd_cols)
colors_alpha = []
colors_tn = []
colors_alpha.append(cp[0])
for i in range(1,4):
    [colors_alpha.append(cp[i]) for _ in range(3)]
for c in cp:
    [colors_tn.append(c) for _ in range(4)]
#Build styles
styles_alpha = ['dotted'] + 3*['solid','dashdot','dashed']
styles_tn = 5*['solid','dashed','dashdot','dotted'] 

#Make alpha and tn labels
labels_alpha = [r'$\mathrm{uniform}$',
r'$-1.5$',r'$-1.5$, $b=1$',r'$-1.5$, $b=2$',
r'$-2.0$',r'$-2.0$, $b=1$',r'$-2.0$, $b=2$',
r'$-2.5$',r'$-2.5$, $b=1$',r'$-2.5$, $b=2$']
labels_tn = [r'$%d$'%tn for tn in np.arange(250,5250,250)]

#Set list of unique heating functions; should correspond to alpha labels list
alpha_list = [['uniform',''],
['1.5',''],['1.5','-b1.0'],['1.5','-b2.0'],
['2.0',''],['2.0','-b1.0'],['2.0','-b2.0'],
['2.5',''],['2.5','-b1.0'],['2.5','-b2.0']]

#Instantiate the builder class for the by-alpha histogram
histo_builder = ebpe.EMHistoBuilder(args.species, args.loop_length, args.tpulse, alpha_list, group='by_alpha', fs=fontsize, figsize=figsize, format=format, dpi=dpi, root_dir=root_dir)
#Load data
histo_builder.loader()
#Load histogram options
histo_opts = {}
for i in range(len(alpha_list)):
    histo_opts[''.join(alpha_list[i])] = {'color':colors_alpha[i],'label':labels_alpha[i],'linestyle':styles_alpha[i],'linewidth':linewidth,'normed':True,'stacked':True}
#Build cool histogram
histo_builder.histo_maker('cool',histo_opts,x_limits=[1.9,5.1],leg_loc=1,print_fig_filename=fn_temp+'.alpha.cool')
#Build hot histogram
histo_builder.histo_maker('hot',histo_opts,x_limits=[2.9,6.0],leg_loc=1,print_fig_filename=fn_temp+'.alpha.hot')

#Instantiate the builder class for the by_t_wait
histo_builder = ebpe.EMHistoBuilder(args.species, args.loop_length, args.tpulse, alpha_list, group='by_t_wait', fs=fontsize, figsize=figsize, format=format, dpi=dpi, root_dir=root_dir)
#Load data
histo_builder.loader()
#Load histogram options
histo_opts = {}
for i in range(len(styles_tn)):
    histo_opts[str(i)] = {'color':colors_tn[i],'label':labels_tn[i],'linestyle':styles_tn[i],'linewidth':linewidth,'normed':True,'stacked':True}
#Build cool histogram
histo_builder.histo_maker('cool',histo_opts,x_limits=[1.9,5.1],leg_loc=1,print_fig_filename=fn_temp+'.t_wait.cool')
#Build hot histogram
histo_builder.histo_maker('hot',histo_opts,x_limits=[2.9,6.0],leg_loc=1,print_fig_filename=fn_temp+'.t_wait.hot')
