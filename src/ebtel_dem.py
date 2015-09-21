#ebtel_dem.py

#Will Barnes
#13 May 2015

#Import needed modules
import numpy as np
import sys
from scipy.optimize import curve_fit

class DEMProcess(object):
    
    em_max_eps_percent = 0.999
    
    def __init__(self,root_dir,species,alpha,loop_length,tpulse,solver,**kwargs):
        #check for wait time scaling option
        if 't_wait_q_scaling' in kwargs:
            scaling_suffix = '-b'+str(kwargs['t_wait_q_scaling'])
        else:
            scaling_suffix = ''
            
        #set up paths
        child_path = root_dir+species+'_heating_runs/alpha'+str(alpha)+'/data/'
        self.file_path = 'ebtel_L'+str(loop_length)+'_tn%d'+scaling_suffix+'_tpulse'+str(tpulse)+'_'+solver
        self.root_path = child_path + self.file_path
        #configure keyword arguments
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = True
        #define variables to be used later
        self.em,self.em_max,self.em_mean,self.em_std = [],[],[],[]
        self.temp_em,self.temp_max,self.temp_mean = [],[],[]
        
        
    def import_raw(self,Tn,**kwargs):
        """Import all runs for given Tn waiting time values; replace NaNs with -Inf if present. EM and T values are returned in nested lists."""
        
        for i in range(len(Tn)):
            #initialize lists
            em = []
            temp_em = []
            #initialize counter and flag
            counter=0
            #failed read-in tolerance parameters
            MAX_FAIL = 5
            fail_count = 0
            #build wait-time specific path
            tn_path = self.root_path%Tn[i]
            while fail_count <= MAX_FAIL:
                try:
                    temp = np.loadtxt(tn_path+'/'+self.file_path%Tn[i]+'_'+str(counter)+'_dem.txt')
                    temp[np.where(np.isnan(temp))] = -np.inf
                    temp_em.append(temp[:,0])
                    em.append(temp[:,4])
                    #reset fail count after success
                    fail_count = 0
                except FileNotFoundError:
                    fail_count += 1
                    if self.verbose:
                        print("Unable to process file for Tn = "+str(Tn[i])+", run = "+str(counter))
                        
                    if fail_count > MAX_FAIL:
                        print("Reached end of list or too many missing files.")
                        print("Estimated percentage of files read = %f %%"%(len(em)/(counter - MAX_FAIL)*100))
                    
                    pass
                #increment counter
                counter += 1
                    
            self.temp_em.append(temp_em)
            self.em.append(em)
                
                
    def calc_stats(self,**kwargs):
        """Calculate mean, standard deviation and max for EM and T."""
        
        if not self.temp_em or not self.em:
            raise ValueError("Before computing EM statistics, run self.import_raw() to process EM,T data.")
        
        for i in range(len(self.em)):
            #first calculate mean
            if len(np.shape(np.array(self.em[i]))) > 1:
                temporary_mean_em = np.array(np.mean(self.inf_filter(self.em[i]),axis=0))
                temporary_std_em = np.array(np.std(self.inf_filter(self.em[i]),axis=0))
                temporary_mean_em[np.where(temporary_mean_em==0.0)]=-np.float('Inf')
                temporary_std_em[np.where(temporary_mean_em==0.0)]=-np.float('Inf')
                self.em_mean.append(temporary_mean_em)
                self.em_std.append(temporary_std_em)
                self.temp_mean.append(np.mean(self.temp_em[i],axis=0))
            else:
                self.em_mean.append(np.array(self.em[i]))
                self.em_std.append(np.zeros(len(self.em[i])))
                self.temp_mean.append(np.array(self.temp_em[i]))
            
            #declare temp lists for max quantities
            temp_max_temp = []
            em_max_temp = []    
            for j in range(len(self.em[i])):
                i_max = np.argmax(self.em[i][j])
                indices_em_max = np.where(np.array(self.em[i][j]) > self.em_max_eps_percent*self.em[i][j][i_max])[0]
                if len(indices_em_max) <= 1:
                    temp_max_temp.append(self.temp_em[i][j][i_max])
                    em_max_temp.append(self.em[i][j][i_max])
                else:
                    em_interp = np.linspace(self.em[i][j][indices_em_max[0]],self.em[i][j][indices_em_max[-1]],100)
                    temp_interp = np.interp(em_interp,np.array(self.em[i][j][indices_em_max[0]:indices_em_max[-1]]),np.array(self.temp_em[i][j][indices_em_max[0]:indices_em_max[-1]]))
                    temp_max_temp.append(np.mean(temp_interp))
                    em_max_temp.append(np.mean(em_interp))
                    
            #append max quantities for each i
            self.temp_max.append(temp_max_temp)
            self.em_max.append(em_max_temp)
        
        
    def inf_filter(self,nested_list,**kwargs):
        #preallocate space
        filtered_list = []
        #filter out infs in list and set to zero for averaging
        for i in nested_list:
            temp_array = np.array(i)
            temp_array[np.where(np.isinf(temp_array)==True)]=0.0
            filtered_list.append(temp_array)
        
        return filtered_list
        
        
class DEMAnalyze(object):
    
    cool_diff = -0.6
    hot_diff = 0.4
    
    def __init__(self,em,temp,em_mean,temp_mean,sigma,**kwargs):
        #get nested lists with EM and T values
        self.em = em
        self.temp_em = temp
        self.sigma_em = sigma
        #keyword arguments
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = True
        if 'slope_limits' in kwargs:
            self.slope_limits = kwargs['slope_limits']
        else:
            self.slope_limits = {}
        if 'fit_method' in kwargs:
            self.fit_method = kwargs['fit_method']
        else:
            self.fit_method = 'fit_all'
        if 'lim_method' in kwargs:
            self.lim_method = kwargs['lim_method']
        else:
            self.lim_method = 'static'
        #class parameters
        if 'em_cutoff' in kwargs:
            self.em_cutoff = kwargs['em_cutoff']
        else:
            self.em_cutoff = 23.0
        if 'delta_t' in kwargs:
            self.delta_t = kwargs['delta_t']
        else:
            self.delta_t = 0.4
        #define variables to be used later
        self.cool_fits = []
        self.hot_fits = []
        
    
    def interp_and_filter(self,**kwargs):
        """Interpolate and filter mean and standard deviation for EM, T arrays; this step is mandatory for later slope calculations."""
        
        for i in range(len(self.em)):
            #find cutoff index
            inf_index = np.where(self.em_mean[i] > self.em_cutoff)
            #reshape temperature and interpolate emission measure
            temp_new = np.linspace(self.temp_mean[i][inf_index[0][0]],self.temp_mean[i][inf_index[0][-1]],2000)
            dem_new = np.interp(temp_new,self.temp_mean[i][inf_index[0][0]:inf_index[0][-1]],self.em_mean[i][inf_index[0][0]:inf_index[0][-1]])
            sigma_new = np.interp(temp_new,self.temp_mean[i][inf_index[0][0]:inf_index[0][-1]],self.em_sigma[i][inf_index[0][0]:inf_index[0][-1]])
            #reassign to nested list
            self.temp_mean[i] = temp_new
            self.em_mean[i] = dem_new
            self.sigma[i] = sigma_new
            #do the same for nested MC entries
            for j in range(len(self.em[i])):
                inf_index = np.where(self.em[i][j] > self.em_cutoff)
                temp_new = np.linspace(self.temp[i][j][inf_index[0][0]],self.temp[i][j][inf_index[0][-1]],2000)
                self.em[i][j] = np.interp(temp_new,self.temp[i][j][inf_index[0][0]:inf_index[0][-1]],self.em[i][j][inf_index[0][0]:inf_index[0][-1]])
                self.temp[i][j] = temp_new
        
        
    def many_fits(self,**kwargs):
        """Calculate fits to hot and cool branches for all EM and T data sets"""
        
        for i in range(len(self.em)):
            #calculate slopes and associated error bars depending on chosen method
            if self.fit_method is 'fit_all':
                cool_temp =[], hot_temp =[]
                for j in range(len(self.em[i])):
                    bound_arrays = self.bounds(self.temp[i][j], self.em[i][j], np.ones(len(self.em[i][j])), self.fit_limits(self.temp[i][j], self.em[i][j]))
                    fits = self.branch_fit(bound_arrays['temp_cool'], bound_arrays['dem_cool'], bound_arrays['temp_hot'], bound_arrays['dem_hot'])
                    cool_temp.append([fits['a_c'],fits['b_c']]), hot_temp.append([fits['a_h'],fits['b_h']])
                #calculate standard deviation and mean and store values
                #a = 
                #b = 
                #sigma_a = 
                #sigma_b = 
                self.cool_fits.append([])
                
            elif self.fit_method is 'fit_plus-minus':
                #mean + sigma
                bound_arrays = self.bounds(self.temp_mean[i], self.em_mean[i]+self.sigma[i], self.sigma[i], self.fit_limits(self.temp_mean[i], self.em_mean[i]+self.sigma[i]))
                fits_plus = self.branch_fit(bound_arrays['temp_cool'],bound_arrays['dem_cool'],bound_arrays['temp_hot'],bound_arrays['dem_hot'])
                #mean - sigma
                bound_arrays = self.bounds(self.temp_mean[i], self.em_mean[i]-self.sigma[i], self.sigma[i], self.fit_limits(self.temp_mean[i], self.em_mean[i]-self.sigma[i]))
                fits_minus = self.branch_fit(bound_arrays['temp_cool'],bound_arrays['dem_cool'],bound_arrays['temp_hot'],bound_arrays['dem_hot'])
                #mean
                bound_arrays = self.bounds(self.temp_mean[i], self.em_mean[i], self.sigma[i], self.fit_limits(self.temp_mean[i], self.em_mean[i]))
                fits = self.branch_fit(bound_arrays['temp_cool'],bound_arrays['dem_cool'],bound_arrays['temp_hot'],bound_arrays['dem_hot'])
                #calculate sigma and store values
                sac = np.max(np.fabs(fits['a_c']-fits_minus['a_c']),np.fabs(fits['a_c']-fits_plus['a_c']))
                sbc = np.max(np.fabs(fits['b_c']-fits_minus['b_c']),np.fabs(fits['b_c']-fits_plus['b_c']))
                sah = np.max(np.fabs(fits['a_h']-fits_minus['a_h']),np.fabs(fits['a_h']-fits_plus['a_h']))
                sbh = np.max(np.fabs(fits['b_h']-fits_minus['b_h']),np.fabs(fits['b_h']-fits_plus['b_h']))
                self.cool_fits.append([fits['a_c'],fits['b_c'],[sac,sbc]]),self.hot_fits.append([fits['a_h'],fits['b_h'],[sah,sbh]])
                
            elif self.fit_method is 'fit_mean_weighted':
                bound_arrays = self.bounds(self.temp_mean[i],self.em_mean[i],self.sigma[i],self.fit_limits(self.temp_mean[i],self.em_mean[i],method='dynamic'))
                fits = self.branch_fit(bound_arrays['temp_cool'], bound_arrays['dem_cool'], bound_arrays['temp_hot'], bound_arrays['dem_hot'], sigma_cool=bound_arrays['sigma_cool'], sigma_hot=bound_arrays['sigma_hot'])
                self.cool_fits.append([fits['a_c'],fits['b_c'],fits['s_c']]),self.hot_fits.append([fits['a_h'],fits['b_h'],fits['s_h']])
                
            else:
                print("Unrecognized fit method. Exiting...")
                sys.exit()
            
            
    def bounds(self,temp,dem,sigma,slope_limits,**kwargs):
        """Create bounded hot and cool branches from given hot and cool branch limits (or default values); interpolation over EM curves should be done before this step."""
        
        #Set default values for hot and cool limits if they have not been specified
        if not slope_limits:
            slope_limits['cool_upper'],slope_limits['hot_lower'] = temp[np.argmax(dem)],temp[np.argmax(dem)]
            slope_limits['cool_lower'] = slope_limits['cool_upper'] + self.cool_diff
            slope_limits['hot_upper'] = slope_limits['hot_lower'] + self.hot_diff
            if self.verbose:
                print("No slope limits specified; using default values:")
                print("    T_cool_upper = "+str(slope_limits['cool_upper'])+" K")
                print("    T_cool_lower = "+str(slope_limits['cool_lower'])+" K")
                print("    T_hot_upper = "+str(slope_limits['hot_upper'])+" K")
                print("    T_hot_lower = "+str(slope_limits['hot_lower'])+" K")
        
        #Construct hot and cool dem and temp arrays for given bounds
        i_cool_lower = np.where(temp<slope_limits['cool_lower'])
        i_cool_upper = np.where(temp>slope_limits['cool_upper'])
        if len(i_cool_lower[0]) > 0 and len(i_cool_upper[0]) > 0 and temp[i_cool_upper[0][0] - 1] <= slope_limits['hot_lower']:
            temp_cool = temp[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
            dem_cool = dem[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
            sigma_cool = sigma[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]    
        else:
            if self.verbose:
                print("Cool bound out of range, T = %.2f > T_limit = %.2f"%(temp[0],slope_limits['cool_lower']))
                print("or T_upper_limit = %.2f > T_max = %.2f"%(slope_limits['cool_upper'],slope_limits['hot_lower']))
            temp_cool = False
            dem_cool = False
            sigma_cool = False

        i_hot_lower = np.where(temp<slope_limits['hot_lower'])
        i_hot_upper = np.where(temp>slope_limits['hot_upper'])
        if len(i_hot_lower[0]) > 0 and len(i_hot_upper[0]) > 0 and temp[i_hot_lower[0][-1] + 1] >= slope_limits['cool_upper']:
            temp_hot = temp[(i_hot_lower[0][-1] + 1):(i_hot_upper[0][0] - 1)]
            dem_hot = dem[(i_hot_lower[0][-1] + 1):(i_hot_upper[0][0] - 1)]
            sigma_hot = sigma[(i_hot_lower[0][-1] + 1):(i_hot_upper[0][0] - 1)]
        else:
            if self.verbose:
                print("Hot bound out of range, T = %.2f < T_limit = %.2f"%(temp[-1],slope_limits['hot_upper']))
                print("or T_lower_limit = %.2f < T_max = %.2f"%(slope_limits['hot_lower'],slope_limits['cool_upper']))
            temp_hot = False
            dem_hot = False
            sigma_hot = False
        
        #Return interpolated arrays and indices
        return {'temp_cool':temp_cool,'dem_cool':dem_cool,'sigma_cool':sigma_cool,'temp_hot':temp_hot,'dem_hot':dem_hot,'sigma_hot':sigma_hot}
        
        
    def branch_fit(self,temp_cool,dem_cool,temp_hot,dem_hot,**kwargs):
        """Linear fit to hot and cool branches of EM curve using hot and cool branches constructed according to hot and cool limits."""
        
        #unpack uncertainties
        if 'sigma_cool' in kwargs:
            sigma_cool = kwargs['sigma_cool']
            absolute_sigma_cool = True
        else:
            sigma_cool = np.ones(len(temp_cool))
            absolute_sigma_cool = False
            
        if 'sigma_hot' in kwargs:
            sigma_hot = kwargs['sigma_hot']
            absolute_sigma_hot = True
        else:
            sigma_hot = np.ones(len(temp_hot))
            absolute_sigma_hot = False
        
        #Function for linear fit
        def linear_fit(x,a,b):
            return a*x + b
            
        #perform linear fit
        #cool
        if temp_cool is False:
            a_coolward = False
            b_coolward = False
            sigma_coolward = False
        else:
            pars_cool,covar_cool = curve_fit(linear_fit,temp_cool,dem_cool,sigma=sigma_cool,absolute_sigma=absolute_sigma_cool)
            a_coolward,b_coolward = pars_cool[0],pars_cool[1]
            #compute residual variance
            res = 1#np.sum((linear_fit(bound_arrays['temp_cool'],*pars_cool) - bound_arrays['dem_cool'])**2)/(len(bound_arrays['dem_cool']) - len(pars_cool))
            sigma_coolward = np.sqrt(np.diag(covar_cool/res))
            
        #hot
        if temp_hot is False:
            a_hotward = False
            b_hotward = False
            sigma_hotward = False
        else:
            pars_hot,covar_hot = curve_fit(linear_fit,temp_hot,dem_hot,sigma=sigma_hot,absolute_sigma=absolute_sigma_hot)
            a_hotward,b_hotward = pars_hot[0],pars_hot[1]
            res = 1#np.sum((linear_fit(bound_arrays['temp_hot'],*pars_hot) - bound_arrays['dem_hot'])**2)/(len(bound_arrays['dem_hot']) - len(pars_hot))
            sigma_hotward = np.sqrt(np.diag(covar_hot/res))
            
        return {'a_c':a_coolward,'b_c':b_coolward,'s_c':sigma_coolward,'a_h':a_hotward,'b_h':b_hotward,'s_h':sigma_hotward}
        
        
    def fit_limits(self,temp,em,**kwargs):
        if self.lim_method is 'dynamic':
            ninf_i = np.where(np.isinf(em) == False) #find non-inf indices
            max_i = np.argmax(em) #find index corresponding to max value
            hot_i = ninf_i[0][np.where(ninf_i[0]>max_i)] #indices for hot branch
            em_hot = em[hot_i] #hot branch em
            delta_em_hot = np.fabs(np.diff(em_hot)) #delta(em) of hot branch
            delta_i = np.where(delta_em_hot>0.5)[0][0]
            lim_i = hot_i[delta_i - 1]-1
            t_upper = temp[lim_i]
            t_lower = t_upper - self.delta_t
        else:
            t_upper = self.slope_limits['hot_upper']
            t_lower = self.slope_limits['hot_lower']
        
        return {'cool_upper':self.slope_limits['cool_upper'],'cool_lower':self.slope_limits['cool_lower'],'hot_upper':t_upper,'hot_lower':t_lower}
        
        
    def branch_fit_statistics(self,**kwargs):
        """Compute mean and standard deviation for fit parameters to hot and cool branches"""
        
        if not self.a_cool or not self.a_hot:
            raise ValueError("Before computing statistics of slopes, first calculate slopes using self.many_fits()")
            
        #compute mean and standard deviation for fit parameters for each T_n value
        for i in range(len(self.a_cool)):
            true_indices_cool = np.where(np.array(self.a_cool[i]) != False)[0]
            if float(len(true_indices_cool))/len(self.a_cool[i]) < 0.9:
                self.a_cool_mean.append(False)
                self.a_cool_std.append(False)
                self.b_cool_mean.append(False)
            else:
                self.a_cool_mean.append(np.mean(np.array(self.a_cool[i])[true_indices_cool,0]))
                self.a_cool_std.append(np.std(np.array(self.a_cool[i])[true_indices_cool,0]))
                self.b_cool_mean.append(np.mean(np.array(self.a_cool[i])[true_indices_cool,1]))
                    
            true_indices_hot = np.where(np.array(self.a_hot[i]) != False)[0]
            if float(len(true_indices_hot))/len(self.a_hot[i]) < 0.9:
                self.a_hot_mean.append(False)
                self.a_hot_std.append(False)
                self.b_hot_mean.append(False)
            else:
                self.a_hot_mean.append(np.mean(np.array(self.a_hot[i])[true_indices_hot,0]))
                self.a_hot_std.append(np.std(np.array(self.a_hot[i])[true_indices_hot,0]))
                self.b_hot_mean.append(np.mean(np.array(self.a_hot[i])[true_indices_hot,1]))
    
        ##############################

#class DEMAnalyzer(object):
#    
#    def __init__(self,root_dir,species,alpha,loop_length,tpulse,solver,**kwargs):
#        #set object variables
#        self.root_dir = root_dir
#        self.species = species
#        self.alpha = alpha
#        self.loop_length = loop_length
#        self.tpulse = tpulse
#        self.solver = solver
#        #set up paths
#        child_path = self.root_dir+self.species+'_heating_runs/alpha'+str(self.alpha)+'/data/'
#        self.file_path = 'ebtel_L'+str(self.loop_length)+'_tn%d_tpulse'+str(self.tpulse)+'_'+self.solver
#        self.root_path = child_path + self.file_path
#        #configure keyword arguments
#        if 'Tn' in kwargs:
#            self.Tn = kwargs['Tn']
#        else:
#            self.Tn = np.arange(250,5250,250)
#        if 'slope_limits' in kwargs:
#            self.slope_limits = kwargs['slope_limits']
#        else:
#            self.slope_limits = {}
#        if 'verbose' in kwargs:
#            self.verbose = kwargs['verbose']
#        else:
#            self.verbose = True
#        #set static variables
#        self.em_cutoff = 26.0
#        self.em_max_eps_percent = 0.999
#        #define variables to be used later
#        self.em,self.em_max,self.em_mean = [],[],[]
#        self.temp_em,self.temp_max,self.temp_mean = [],[],[]
#        self.a_cool,self.a_cool_mean,self.a_cool_std = [],[],[]
#        self.a_hot,self.a_hot_mean,self.a_hot_std = [],[],[]
#        self.b_cool_mean,self.b_hot_mean = [],[]
#            
#    def process_raw(self,**kwargs):
#        for i in range(len(self.Tn)):
#            tn_path = self.root_path%self.Tn[i]
#            #initialize lists
#            em = []
#            temp_em = []
#            #initialize flag and counter
#            eol_flag=False
#            counter=0
#            while eol_flag is False:
#                try:
#                    #load data
#                    temp = np.loadtxt(tn_path+'/'+self.file_path%self.Tn[i]+'_'+str(counter)+'_dem.txt')
#                    #check for nan
#                    temp[np.where(np.isnan(temp))] = -np.inf
#                    #append temperature and EM
#                    temp_em.append(temp[:,0])
#                    em.append(temp[:,4])
#                    #increment counter
#                    counter += 1
#                except:
#                    if self.verbose:
#                        print("Unable to process file for Tn = "+str(self.Tn[i])+", run = "+str(counter))
#                        print("Reached end of list or there was an error reading the file.")
#                    
#                    eol_flag=True
#                    pass
#            self.temp_em.append(temp_em)
#            self.em.append(em)            
#            
#    def em_statistics(self,**kwargs):
#        if not self.temp_em or not self.em:
#            raise ValueError("Before computing EM statistics, run self.process_raw() to process EM,T data.")
#            
#        for i in range(len(self.em)):
#            if len(np.shape(np.array(self.em[i]))) > 1:
#                temporary_mean_em = np.array(np.mean(self.inf_filter(self.em[i]),axis=0))
#                temporary_mean_em[np.where(temporary_mean_em==0.0)]=-np.float('Inf')
#                self.em_mean.append(temporary_mean_em)
#                self.temp_mean.append(np.mean(self.temp_em[i],axis=0))
#            else:
#                self.em_mean.append(np.array(self.em[i]))
#                self.temp_mean.append(np.array(self.temp_em[i]))
#                
#                
#    def find_em_max(self,**kwargs):
#        if not self.temp_em or not self.em:
#            raise ValueError("Before computing EM statistics, run self.process_raw() to process EM,T data.")
#            
#        for i in range(len(self.Tn)):
#            temp_max_temp = []
#            em_max_temp = []
#            for j in range(len(self.em[i])):
#                i_max = np.argmax(self.em[i][j])
#                indices_em_max = np.where(np.array(self.em[i][j]) > self.em_max_eps_percent*self.em[i][j][i_max])[0]
#                if len(indices_em_max) <= 1:
#                    temp_max_temp.append(self.temp_em[i][j][i_max])
#                    em_max_temp.append(self.em[i][j][i_max])
#                else:
#                    em_interp = np.linspace(self.em[i][j][indices_em_max[0]],self.em[i][j][indices_em_max[-1]],100)
#                    temp_interp = np.interp(em_interp,np.array(self.em[i][j][indices_em_max[0]:indices_em_max[-1]]),np.array(self.temp_em[i][j][indices_em_max[0]:indices_em_max[-1]]))
#                    temp_max_temp.append(np.mean(temp_interp))
#                    em_max_temp.append(np.mean(em_interp))
#            self.temp_max.append(temp_max_temp)
#            self.em_max.append(em_max_temp)
#                
#                
#    def many_slopes(self,**kwargs):
#        for i in range(len(self.Tn)):
#            acl = []
#            ahl = []
#            #redefine slope limits if kwargs
#            if 'slope_limits' in kwargs:
#                self.slope_limits['cool_lower'] = kwargs['slope_limits']['cool_lower'][i]
#                self.slope_limits['cool_upper'] = kwargs['slope_limits']['cool_upper'][i]
#                self.slope_limits['hot_lower'] = kwargs['slope_limits']['hot_lower'][i]
#                self.slope_limits['hot_upper'] = kwargs['slope_limits']['hot_upper'][i]
#                
#            for j in range(len(self.temp_em[i])):
#                ac,bc,ah,bh = self.slope(self.temp_em[i][j],self.em[i][j])
#                acl.append([ac,bc]),ahl.append([ah,bh])
#                
#            self.a_cool.append(acl),self.a_hot.append(ahl)
#
#
#    def slope(self,temp,dem,**kwargs):
#        #Calculate bounds
#        bound_arrays = self.bounds(temp,dem)
#        
#        #Function for linear fit
#        def linear_fit(x,a,b):
#            return a*x + b
#            
#        #Check if inside interpolated array and calculate slope
#        #cool
#        if bound_arrays['temp_cool'] is False:
#            a_coolward = False
#            b_coolward = False
#        else:
#            pars_cool,covar = curve_fit(linear_fit,bound_arrays['temp_cool'],bound_arrays['dem_cool'])
#            a_coolward,b_coolward = pars_cool[0],pars_cool[1]
#            
#        #hot
#        if bound_arrays['temp_hot'] is False:
#            a_hotward = False
#            b_hotward = False
#        else:
#            pars_hot,covar = curve_fit(linear_fit,bound_arrays['temp_hot'],bound_arrays['dem_hot'])
#            a_hotward,b_hotward = pars_hot[0],pars_hot[1]
#            
#        return a_coolward,b_coolward,a_hotward,b_hotward
#        
#        
#    def slope_statistics(self,**kwargs):
#        if not self.a_cool or not self.a_hot:
#            raise ValueError("Before computing statistics of slopes, first calculate slopes using self.many_slopes()")
#        
#        #compute mean and standard deviation for fit parameters for each T_n value
#        for i in range(len(self.a_cool)):
#            true_indices_cool = np.where(np.array(self.a_cool[i]) != False)[0]
#            if float(len(true_indices_cool))/len(self.a_cool[i]) < 0.9:
#                self.a_cool_mean.append(False)
#                self.a_cool_std.append(False)
#                self.b_cool_mean.append(False)
#            else:
#                self.a_cool_mean.append(np.mean(np.array(self.a_cool[i])[true_indices_cool,0]))
#                self.a_cool_std.append(np.std(np.array(self.a_cool[i])[true_indices_cool,0]))
#                self.b_cool_mean.append(np.mean(np.array(self.a_cool[i])[true_indices_cool,1]))
#                
#            true_indices_hot = np.where(np.array(self.a_hot[i]) != False)[0]
#            if float(len(true_indices_hot))/len(self.a_hot[i]) < 0.9:
#                self.a_hot_mean.append(False)
#                self.a_hot_std.append(False)
#                self.b_hot_mean.append(False)
#            else:
#                self.a_hot_mean.append(np.mean(np.array(self.a_hot[i])[true_indices_hot,0]))
#                self.a_hot_std.append(np.std(np.array(self.a_hot[i])[true_indices_hot,0]))
#                self.b_hot_mean.append(np.mean(np.array(self.a_hot[i])[true_indices_hot,1]))
#        
#        
#    def bounds(self,temp,dem,**kwargs):
#        #Filter inf and unrealistically low values
#        #Find the dem index where dem->inf (or less than the cutoff)
#        inf_index = np.where(dem > self.em_cutoff)
#        #Interpolate DEM and temperature arrays
#        temp_new = np.linspace(temp[inf_index[0][0]],temp[inf_index[0][-1]],2000)
#        dem_new = np.interp(temp_new,temp[inf_index[0][0]:inf_index[0][-1]],dem[inf_index[0][0]:inf_index[0][-1]])
#        #Select hot and cool upper and lower bounds
#        if not self.slope_limits:
#            self.slope_limits['cool_upper'],self.slope_limits['hot_lower'] = temp[np.argmax(dem)],temp[np.argmax(dem)]
#            self.slope_limits['cool_lower'] = self.slope_limits['cool_upper'] - 0.6
#            self.slope_limits['hot_upper'] = self.slope_limits['hot_lower'] + 0.4
#        
#        #Construct hot and cool dem and temp arrays for given bounds
#        i_cool_lower = np.where(temp_new<self.slope_limits['cool_lower'])
#        i_cool_upper = np.where(temp_new>self.slope_limits['cool_upper'])
#        if len(i_cool_lower[0]) > 0 and len(i_cool_upper[0]) > 0 and temp_new[i_cool_upper[0][0] - 1] <= self.slope_limits['hot_lower']:
#            temp_new_cool = temp_new[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
#            dem_new_cool = dem_new[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
#        else:
#            if self.verbose:
#                print("Cool bound out of range, T = %.2f > T_limit = %.2f"%(temp_new[0],self.slope_limits['cool_lower']))
#                print("or T_upper_limit = %.2f > T_max = %.2f"%(self.slope_limits['cool_upper'],self.slope_limits['hot_lower']))
#            temp_new_cool = False
#            dem_new_cool = False
#
#        i_hot_lower = np.where(temp_new<self.slope_limits['hot_lower'])
#        i_hot_upper = np.where(temp_new>self.slope_limits['hot_upper'])
#        if len(i_hot_lower[0]) > 0 and len(i_hot_upper[0]) > 0 and temp_new[i_hot_lower[0][-1] + 1] >= self.slope_limits['cool_upper']:
#            temp_new_hot = temp_new[(i_hot_lower[0][-1] + 1):(i_hot_upper[0][0] - 1)]
#            dem_new_hot = dem_new[(i_hot_lower[0][-1] + 1):(i_hot_upper[0][0] - 1)]
#        else:
#            if self.verbose:
#                print("Hot bound out of range, T = %.2f < T_limit = %.2f"%(temp_new[-1],self.slope_limits['hot_upper']))
#                print("or T_lower_limit = %.2f < T_max = %.2f"%(self.slope_limits['hot_lower'],self.slope_limits['cool_upper']))
#            temp_new_hot = False
#            dem_new_hot = False
#        
#        #Return interpolated arrays and indices
#        return {'temp_cool':temp_new_cool,'dem_cool':dem_new_cool,'temp_hot':temp_new_hot,'dem_hot':dem_new_hot}
#        
#        
#    def inf_filter(self,nested_list,**kwargs):
#        #preallocate space
#        filtered_list = []
#        #filter out infs in list and set to zero for averaging
#        for i in nested_list:
#            temp_array = np.array(i)
#            temp_array[np.where(np.isinf(temp_array)==True)]=0.0
#            filtered_list.append(temp_array)
#        return filtered_list
#        
#        
#    #DEPRECATED
#    def integrate(self,temp,dem,**kwargs):
#        #Find the corresponding temperature bounds
#        dict_bounds = self.bounds(temp,dem)
#
#        #First check if the bounds are inside of our interpolated array
#        if np.size(dict_bounds['bound_cool']) == 0 or np.size(dict_bounds['bound_hot']) == 0:
#            print("Cool and/or hot bound(s) out of range. Skipping integration for these bounds.")
#            hot_shoulder_strength = False
#        else:
#            #Refine the arrays we will integrate over
#            #Temprature
#            temp_hot = dict_bounds['temp_hot'][0:(dict_bounds['bound_hot'][0][0] - 1)]
#            temp_cool = dict_bounds['temp_cool'][(dict_bounds['bound_cool'][0][-1] + 1):-1]
#            #DEM (EM)
#            dem_hot = dict_bounds['dem_hot'][0:(dict_bounds['bound_hot'][0][0] - 1)]
#            dem_cool = dict_bounds['dem_cool'][(dict_bounds['bound_cool'][0][-1] + 1):-1]
#            #Do the integration
#            #Hot shoulder
#            hot_shoulder = np.trapz(dem_hot,x=temp_hot)
#            #Total
#            total_shoulder = np.trapz(np.concatenate([dem_cool[0:-1],dem_hot]),x=np.concatenate([temp_cool[0:-1],temp_hot]))
#            #Compute the ratio
#            hot_shoulder_strength = hot_shoulder/total_shoulder
#
#        return hot_shoulder_strength
#        
#        