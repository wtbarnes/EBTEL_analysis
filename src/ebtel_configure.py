#ebtel_configure.py

#Will Barnes
#7 May 2015

#Import needed modules
import numpy as np
import os
import __builtin__
import logging
import itertools
import xml.etree.ElementTree as ET
import xml.dom.minidom as xdm

#Resolve Python 2/3 exception problem
exc = getattr(__builtin__,"IOError","FileNotFoundError")

class Configurer(object):

    def __init__(self, config_dictionary, root_dir, mc=None, t_wait_q_scaling = None, build_paths=False, Hn=None, delta_q=None, constraint_tol = 1e-3, **kwargs):
        """Constructor for Configurer class used to print EBTEL and IonPopSolver configuration files."""

        self.config_dictionary = config_dictionary
        self.root_dir = root_dir
        self.nmc_list = []
        #set optional variables
        self.Hn = Hn
        self.delta_q = delta_q
        self.mc = mc
        self.t_wait_q_scaling = t_wait_q_scaling
        self.tol =  constraint_tol
        #configure logger
        self.logger = logging.getLogger(type(self).__name__)
        #Set up paths
        if build_paths:
            self.path_builder()
        else:
            self.logger.warning("Build paths not constructed! You will not be able to build config files for variable Tn until you run self.path_builder().")


    def path_builder(self,**kwargs):
        """Build path names needed for printing config files and building data folders; check that needed directories exist and create them if necessary."""

        gen_path = self.config_dictionary['heat_species']+'_heating_runs'
        if self.config_dictionary['amp_switch'] == 'uniform':
            gen_path = os.path.join(gen_path,'alpha' + self.config_dictionary['amp_switch'])
        else:
            gen_path = os.path.join(gen_path,'alpha' + str(np.fabs(self.config_dictionary['alpha'])))

        #check for t_wait-q scaling
        if self.t_wait_q_scaling:
            scaling_suffix = '-b'+str(self.t_wait_q_scaling)
        else:
            scaling_suffix = ''

        #set data and config paths
        self.data_path = os.path.join(self.root_dir, gen_path,'data')
        self.config_path = os.path.join(self.root_dir, gen_path,'config')
        self.fn = 'ebtel_L' + str(self.config_dictionary['loop_length']) + '_tn%d' + scaling_suffix + '_tpulse' + str(2.0*self.config_dictionary['t_pulse_half']) + '_' + self.config_dictionary['solver']


    def print_xml_config(self,config_file='../config/ebtel_config.xml',**kwargs):
        """Print EBTEL XML configuration file from dictionary of input parameters"""

        #create root element
        root = ET.Element('root')

        #Loop through dictionary and print to xml file
        for key in self.config_dictionary:
            if (type(self.config_dictionary[key]) is not np.ndarray) and (type(self.config_dictionary[key]) is not list):
                self._set_xml_element(root,key,self.config_dictionary[key])
            else:
                element = ET.SubElement(root,key)
                for i in range(len(self.config_dictionary[key])):
                    self._set_xml_element(element,key+str(i),self.config_dictionary[key][i])

        #print to file
        with open(config_file,'w') as f:
            f.write(self._pretty_print_xml(root))


    def _set_xml_element(self,root,name,val):
        """create element, set tag"""
        element = ET.SubElement(root,name)
        element.text = str(val)


    def _pretty_print_xml(self,element):
        """nicely formatted printing of XML files"""
        unformatted = ET.tostring(element)
        xdmparse = xdm.parseString(unformatted)
        return xdmparse.toprettyxml(indent="    ")


    def print_ips_input(self,ebtel_root_dir,**kwargs):
        """Reshape EBTEL results to be used as IonPopSolver input"""

        #Rebuild paths to read in EBTEL input
        data_stripped = self.data_path.replace(self.root_dir,'')
        config_stripped = self.config_path.replace(self.root_dir,'')
        if data_stripped[0] == '/':
            data_stripped = data_stripped[1:]
        if config_stripped[0] == '/':
            config_stripped = config_stripped[1:]
        path_to_ebtel_results = os.path.join(ebtel_root_dir,data_stripped)
        path_to_ebtel_config = os.path.join(ebtel_root_dir,config_stripped)

        #read in job array config file
        self.config_array = np.loadtxt(os.path.join(path_to_ebtel_config,'ebtel_L' + str(self.config_dictionary['loop_length']) + '_tpulse' + str(2.0*self.config_dictionary['t_pulse_half']) + '_' + self.config_dictionary['solver'] + '_job_array.conf'))
        self.config_array_clean = []

        n_index = 2
        if self.config_dictionary['heat_species'] == 'electron' or self.config_dictionary['heat_species'] == 'ion':
            n_index += 1

        #loop through files, print data as configuration file for IonPopSolver
        for c in self.config_array:
            tmp_fn = os.path.join(path_to_ebtel_results,self.fn%(c[0]),self.fn%(c[0])+'_'+str(int(c[1]))+'.txt')
            self.logger.debug('Reshaping EBTEL results file %s'%(tmp_fn))
            try:
                tmp_data = np.loadtxt(tmp_fn)
            except exc:
                self.logger.exception("%s cannot be loaded"%tmp_fn)
                continue
            t,T,n = tmp_data[:,0],tmp_data[:,1],tmp_data[:,n_index]
            #check for existence of top level directories for config and data
            if not os.path.exists(os.path.join(self.config_path,self.fn%(c[0]))):
                os.makedirs(os.path.join(self.config_path,self.fn%(c[0])))
            if not os.path.exists(os.path.join(self.data_path,self.fn%(c[0]))):
                os.makedirs(os.path.join(self.data_path,self.fn%(c[0])))
            #save reshaped results
            np.savetxt(os.path.join(self.config_path,self.fn%(c[0]),self.fn%(c[0])+'_'+str(int(c[1]))+'.reshape.txt'), np.transpose([t,T,n]), header=str(len(t)),comments='', fmt='%f\t%e\t%e')
            #save successful load parameters to clean config
            self.config_array_clean.append([c[0],c[1]])
            
        self.config_array_clean = np.array(self.config_array_clean)


    def print_ips_config(self,**kwargs):
        """Print file with IonPopSolver configuration options"""

        f = open(os.path.join(self.config_path,'ebtel_L' + str(self.config_dictionary['loop_length']) + '_tpulse' + str(2.0*self.config_dictionary['t_pulse_half']) + '_' + self.config_dictionary['solver'] + '_job_array.conf'),'w')

        for c in self.config_array_clean:
            f.write('%s\t'%(os.path.join(self.config_path,self.fn%(c[0]),self.fn%(c[0])+'_'+str(int(c[1]))+'.reshape.txt')))
            f.write('%s\n'%(os.path.join(self.data_path,self.fn%(c[0]),self.fn%(c[0])+'_'+str(int(c[1]))+'.ips_results.txt')))

        f.close()


    def print_job_array_config(self,**kwargs):
        """Print run number and associated wait time for each unique job to be run according to mc number"""

        try:
            top_list = []
            for i in range(len(self.t_wait_mean)):
                sub_list = []
                [sub_list.append([self.t_wait_mean[i],j]) for j in range(self.nmc_list[i])]
                top_list.append(sub_list)

            top_list_flattened = list(itertools.chain(*top_list))
            np.savetxt(os.path.join(self.config_path,'ebtel_L' + str(self.config_dictionary['loop_length']) + '_tpulse' + str(2.0*self.config_dictionary['t_pulse_half']) + '_' + self.config_dictionary['solver'] + '_job_array.conf'),top_list_flattened,fmt='%d')

        except AttributeError:
            self.logger.exception("Before printing the job_array.conf file, set up the config path with path_builder and then build the t_wait and nmc_list variables by running self.vary_wait_time.")


    def vary_wait_time(self,tn_a,tn_b,delta_tn,**kwargs):
        """Print configuration files for varying wait-time between successive heating events"""

        #Build wait time list
        self.t_wait_mean = np.arange(tn_a,tn_b+delta_tn,delta_tn)
        #Iterate over wait times
        for i in range(len(self.t_wait_mean)):
            #set total number of events
            self.config_dictionary['num_events'] = int(np.ceil(self.config_dictionary['total_time']/(2.0*self.config_dictionary['t_pulse_half'] + self.t_wait_mean[i])))
            #Create directories in data and config if needed
            if not os.path.exists(os.path.join(self.config_path,self.fn%self.t_wait_mean[i])):
                os.makedirs(os.path.join(self.config_path,self.fn%self.t_wait_mean[i]))

            if not os.path.exists(os.path.join(self.data_path,self.fn%self.t_wait_mean[i])):
                os.makedirs(os.path.join(self.data_path,self.fn%self.t_wait_mean[i]))

            #Print config files for each run
            #Check if Monte-Carlo run
            if self.mc:
                num_runs = self._calc_nmc()
            else:
                num_runs = 1

            self.nmc_list.append(num_runs)
            #Iterate over runs
            for j in range(num_runs):
                #build amplitude arrays
                self.amp_arrays(self.t_wait_mean[i])
                #build start and end time arrays
                self.time_arrays(self.t_wait_mean[i])
                #set name of output file
                self.config_dictionary['output_file'] = os.path.join(self.data_path, self.fn%self.t_wait_mean[i], self.fn%self.t_wait_mean[i]+'_'+str(j))
                #print configuration files
                self.print_xml_config(config_file=os.path.join(self.config_path, self.fn%self.t_wait_mean[i], self.fn%self.t_wait_mean[i] + '_' + str(j) + '.xml'))


    def _calc_nmc(self,**kwargs):
        """Calculate number of runs needed to maintain sufficiently large number of heating events"""

        return int(np.ceil(self.mc/self.config_dictionary['num_events']))


    def time_arrays(self,ti,**kwargs):
        """Create start time and end time arrays"""

        #preallocate space for start, end time arrays
        self.config_dictionary['start_time_array'], self.config_dictionary['end_time_array'] = np.empty([self.config_dictionary['num_events']]), np.empty([self.config_dictionary['num_events']])
        #calculate coefficient (xi^(1/b))
        if self.t_wait_q_scaling:
            xi_1ob = (self.config_dictionary['amp_array']**(1.0/self.t_wait_q_scaling)).sum()/(self.config_dictionary['total_time'] - self.config_dictionary['num_events']*2.0*self.config_dictionary['t_pulse_half'])
        #initialize running wait time sum
        t_wait_sum = 0.0
        #configure start and end time for each event
        for i in range(self.config_dictionary['num_events']):
            #calculate start time for static wait time
            self.config_dictionary['start_time_array'][i] = i*(2.0*self.config_dictionary['t_pulse_half']) + t_wait_sum
            #set end time based on pulse duration
            self.config_dictionary['end_time_array'][i] = self.config_dictionary['start_time_array'][i] + 2.0*self.config_dictionary['t_pulse_half']
            #calculate wait time sum
            if self.t_wait_q_scaling:
                #calculate start time for wait time scaled to amplitude (Q=xi*T_N^b)
                t_wait_temp = (self.config_dictionary['amp_array'][i])**(1.0/self.t_wait_q_scaling)/xi_1ob
                t_wait_sum = t_wait_sum + t_wait_temp
            else:
                #increment start time sum
                t_wait_sum = t_wait_sum + ti


    def amp_arrays(self,ti,**kwargs):
        """Configure heating rate amplitudes"""

        #calculate coefficient to convert from energy to heating rate
        if self.config_dictionary['heating_shape'] == 'triangle':
            shape_coeff = 2.0/(2.0*self.config_dictionary['t_pulse_half'])
        elif self.config_dictionary['heating_shape'] == 'square':
            shape_coeff = 1.0/(2.0*self.config_dictionary['t_pulse_half'])
        elif self.config_dictionary['heating_shape'] == 'gaussian':
            shape_coeff = 1.0/(self.config_dictionary['t_pulse_half']*np.sqrt(2.0*np.pi))
        else:
            raise ValueError("Unrecognized heating_shape option. Cannot set heating bounds.")

        #calculate uniform heating amplitude
        self.config_dictionary['h_nano'] = self.Hn*self.config_dictionary['total_time']/(self.config_dictionary['num_events'])*shape_coeff
        #dummy power-law bounds
        self.config_dictionary['amp0'], self.config_dictionary['amp1'] = 0.001,0.1
        if self.config_dictionary['amp_switch'] == 'power_law':
            self.logger.warning("Setting dummy values for power-law bounds.")
        #configure arrays of heating amplitudes from power-law distribution
        if self.config_dictionary['amp_switch'] == 'file':
            np.random.seed()
            self._constrain_distribution(ti)


    def _constrain_distribution(self,ti,**kwargs):
        """Choose events from power-law distribution such that total desired energy input is conserved."""

        #set parameters
        max_tries = 2000
        tries = 0
        err = 1.e+300
        #initial guess of bounds
        a0 = 2./(self.delta_q - 1.)*self.config_dictionary['h_nano']
        a1 = self.delta_q*self.config_dictionary['amp0']
        #save best guesses (helps if routine fails)
        best_err = err
        #begin iteration
        while tries < max_tries and err > self.tol:
            x = np.random.rand(self.config_dictionary['num_events'])
            h = self._power_law_dist(x,a0,a1,self.config_dictionary['alpha'])
            pl_sum = np.sum(h)
            chi = self.config_dictionary['h_nano']*self.config_dictionary['num_events']/pl_sum
            a0 = chi*a0
            a1 = self.delta_q*a0
            err = np.fabs(1.-1./chi)
            if err < best_err:
                best = [a0,a1,h]
                best_err = err
            tries += 1

        self.logger.debug("chi = %f, # of tries = %d, error = %f"%(chi,tries,err))

        if tries >= max_tries:
            self.logger.warning("Power-law constrainer reached max # of tries, using best guess with error = %f"%best_err)

        self.config_dictionary['amp0'] = best[0]
        self.config_dictionary['amp1'] = best[1]
        self.config_dictionary['amp_array'] = best[2]


    def _power_law_dist(self,x,a0,a1,alpha):
        """map uniform variable x to power law distributed variable p for given bounds and index"""

        return ((a1**(alpha + 1.) - a0**(alpha + 1.))*x + a0**(alpha + 1.))**(1./(alpha + 1.))
