#ebtel_configure_main.py

#Will Barnes
#11 May 2015

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
parser.add_argument("--root_dir",help="Optional root directory for config files")
parser.add_argument("--t_wait_scaling",type=float,help="Optional parameter to force scaling between wait time and event amplitude, Q\propto T_N^b; b in Cargill(2014)")
#Declare the parser dictionary
args = parser.parse_args()

#Check for custom root directory
if args.root_dir:
    root_dir = args.root_dir
else:
    root_dir = '/data/datadrive2/EBTEL_runs'
    
#check for wait time scaling exponent
if  args.t_wait_scaling:
    t_wait_scaling = args.t_wait_scaling
else:
    t_wait_scaling = None

#Set heating parameters--configure power-law bounds such that loop is maintained at an equilibrium temperature of T_peak
tpeak = 4.0e+6 #peak temperature for time-averaged heating rate
Hn = 1.0e-6*tpeak**(3.5)/(args.loop_length*1.0e+8)**2 #time-averaged heating rate needed to maintain peak temperature of 4 MK
delta_q = 100.0 #range over which power-law distribution is constructed (typically one decade)

#Configure all static dictionary options
config_dict = {'usage_option':'no_dem','rad_option':'rk','dem_option':'new','heat_flux_option':'limited','solver':args.solver,'ic_mode':'st_eq','print_plasma_params':'True'}
config_dict['total_time'] = 80000
config_dict['tau'] = 1.0
config_dict['rka_error'] = 1.0e-6
config_dict['index_dem'] = 451
config_dict['sat_limit'] = 1.0/6.0
config_dict['h_back'] = 3.5e-5
config_dict['heating_shape'] = 'triangle'
config_dict['t_start_switch'] = 'file'
config_dict['t_end_switch'] = 'file'
config_dict['T0'] = 1.0e+6
config_dict['n0'] = 1.0e+8
config_dict['t_start'] = 0.0
config_dict['t_pulse_half'] = 0.5*args.t_pulse
config_dict['mean_t_start'] = 1000
config_dict['std_t_start'] = 1000
config_dict['sample_rate'] = 10

#Configure directory-level parameters
config_dict['heat_species'] = args.species
config_dict['amp_switch'] = args.amp_switch
config_dict['alpha'] = args.alpha
config_dict['loop_length'] = args.loop_length

#check whether we need to use MC option
if config_dict['amp_switch'] == 'uniform':
    mc = None
else:
    mc = 1.0e+4
    
#configure logging
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

#instantiate configuration class and print configuration files as well as job configuration file
config = Configurer(config_dict,root_dir,Hn=Hn,delta_q=delta_q,mc=mc,build_paths=True,t_wait_q_scaling=t_wait_scaling)
config.vary_wait_time(250,5000,250)
config.print_job_array_config()
