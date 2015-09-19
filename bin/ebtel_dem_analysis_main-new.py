#ebtel_dem_analyze_main-new.py

#Will Barnes
#18 September 2015

#Import needed libraries
import sys
import os
import argparse
import numpy as np
sys.path.append('/home/wtb2/Documents/EBTEL_analysis/src/')
import ebtel_dem as ebd
import ebtel_plot_em as ebpe

#command line argument parsing

#run parameters
species = 'electron'
alpha = -1.5
loop_length = 20.0 
tpulse = 100.0
solver = 'rka4'
t_wait_q_scaling = ''
t_wait = np.arange(250,5250,250)

#set root directory
root_dir = '/data/datadrive2/EBTEL-2fluid_runs'

#Instantiate Process class
proc = ebd.DEMProcess(root_dir,species,alpha,loop_length,tpulse,solver,t_wait_q_scaling=t_wait_q_scaling,verbose=True)
#Import the data
proc.import_raw(t_wait)
