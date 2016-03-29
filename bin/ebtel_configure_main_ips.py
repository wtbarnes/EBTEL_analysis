#!/bin/env python

#ebtel_configure_main_ips.py
#Will Barnes
#29 February 2016

#Import needed modules
import argparse
import logging
import sys
sys.path.append('../src/')
from ebtel_configure import Configurer

#Declare parser object
parser = argparse.ArgumentParser(description='Script that prints configuration files for EBTEL-2fluid runs.')
#Add arguments to parser
parser.add_argument("-s","--species",help="Species to which the heating was applied for particular run.")
parser.add_argument("-as","--amp_switch",help="Switch to decide between power-law and uniform heating.")
parser.add_argument("-a","--alpha",type=float,help="Spectral index for the power-law distribution used.")
parser.add_argument("-L","--loop_length",type=float,help="Loop half-length.")
parser.add_argument("-t","--t_pulse",type=float,help="Width of the heating pulse used for the particular run.")
parser.add_argument("-S","--solver",help="Solver used to compute solutions.")
parser.add_argument("--root_dir_ebtel",help="Optional root directory for EBTEL data files",default='/data/datadrive2/EBTEL_runs')
parser.add_argument("--root_dir",help="Optional root directory for config files",default='/data/datadrive2/IonPopSolver_runs')
parser.add_argument("--t_wait_scaling",type=float,help="Optional parameter to force scaling between wait time and event amplitude, Q\propto T_N^b; b in Cargill(2014)",default=None)
parser.add_argument("--quiet_logger",help="Optional parameter to set logging level to warning.",action='store_true')
#Declare the parser dictionary
args = parser.parse_args()

#Configure all static dictionary options
config_dict = {'solver':args.solver}
config_dict['t_pulse_half'] = 0.5*args.t_pulse
config_dict['heat_species'] = args.species
config_dict['amp_switch'] = args.amp_switch
config_dict['alpha'] = args.alpha
config_dict['loop_length'] = args.loop_length

#configure logging
if args.quiet_logger:
    logging.basicConfig(stream=sys.stdout,level=logging.WARNING)
else:
    logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

config = Configurer(config_dict,root_dir,build_paths=True,t_wait_q_scaling=args.t_wait_scaling)
config.print_ips_input(root_dir_ebtel)
config.print_ips_config()
