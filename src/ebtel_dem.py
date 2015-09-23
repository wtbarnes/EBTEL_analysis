#ebtel_dem.py

#Will Barnes
#13 May 2015

#Import needed modules
import numpy as np
from scipy.optimize import curve_fit

class DEMProcess(object):
    
    em_max_eps_percent = 0.999
    
    def __init__(self,root_dir,species,alpha,loop_length,tpulse,solver,**kwargs):
        #check for wait time scaling option
        if 't_wait_q_scaling' in kwargs and kwargs['t_wait_q_scaling'] is True:
            scaling_suffix = kwargs['t_wait_q_scaling']
        else:
            scaling_suffix = ''
            
        #DEBUG
        print("Scaling suffix is %s"%scaling_suffix)
            
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
                    #DEBUG
                    print("Reading file: %s"%())
                    #
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
        self.temp= temp
        self.em_mean = em_mean
        self.temp_mean = temp_mean
        self.sigma = sigma
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
        if 'max_percent_drop' in kwargs:
            self.max_percent_drop = kwargs['max_percent_drop']
        else:
            self.max_percent_drop = 0.95
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
            sigma_new = np.interp(temp_new,self.temp_mean[i][inf_index[0][0]:inf_index[0][-1]],self.sigma[i][inf_index[0][0]:inf_index[0][-1]])
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
                cool_temp =[]
                hot_temp =[]
                for j in range(len(self.em[i])):
                    bound_arrays = self.bounds(self.temp[i][j], self.em[i][j], np.ones(len(self.em[i][j])))
                    fits = self.branch_fit(bound_arrays['temp_cool'], bound_arrays['dem_cool'], bound_arrays['temp_hot'], bound_arrays['dem_hot'])
                    cool_temp.append([fits['a_c'],fits['b_c']]), hot_temp.append([fits['a_h'],fits['b_h']])
                #calculate standard deviation and mean and store values
                #cool
                true_cool = np.where(np.array(cool_temp)[:,0] != False)[0]
                a,b,sigma_a,sigma_b = False,False,False,False
                if len(true_cool)/len(np.array(cool_temp)[:,0]) >= 0.9:
                    a = np.mean(np.array(cool_temp)[true_cool,0],axis=0)
                    b = np.mean(np.array(cool_temp)[true_cool,1],axis=0)
                    sigma_a = np.std(np.array(cool_temp)[true_cool,0],axis=0)
                    sigma_b = np.std(np.array(cool_temp)[true_cool,1],axis=0)
                self.cool_fits.append([a,b,[sigma_a,sigma_b]])
                #hot
                true_hot = np.where(np.array(hot_temp)[:,0] != False)[0]
                a,b,sigma_a,sigma_b = False,False,False,False
                if len(true_hot)/len(hot_temp) >= 0.9:
                    a = np.mean(np.array(hot_temp)[true_hot,0],axis=0)
                    b = np.mean(np.array(hot_temp)[true_hot,1],axis=0)
                    sigma_a = np.std(np.array(hot_temp)[true_hot,0],axis=0)
                    sigma_b = np.std(np.array(hot_temp)[true_hot,1],axis=0)
                self.hot_fits.append([a,b,[sigma_a,sigma_b]])
                
            elif self.fit_method is 'fit_plus_minus':
                #mean + sigma
                bound_arrays = self.bounds(self.temp_mean[i], self.em_mean[i]+self.sigma[i], self.sigma[i])
                fits_plus = self.branch_fit(bound_arrays['temp_cool'],bound_arrays['dem_cool'],bound_arrays['temp_hot'],bound_arrays['dem_hot'])
                #mean - sigma
                bound_arrays = self.bounds(self.temp_mean[i], self.em_mean[i]-self.sigma[i], self.sigma[i])
                fits_minus = self.branch_fit(bound_arrays['temp_cool'],bound_arrays['dem_cool'],bound_arrays['temp_hot'],bound_arrays['dem_hot'])
                #mean
                bound_arrays = self.bounds(self.temp_mean[i], self.em_mean[i], self.sigma[i])
                fits = self.branch_fit(bound_arrays['temp_cool'],bound_arrays['dem_cool'],bound_arrays['temp_hot'],bound_arrays['dem_hot'])
                #calculate sigma and store values
                sac,sbc,sah,sbh = False,False,False,False
                if fits['a_c'] and fits_minus['a_c'] and fits_plus['a_c']:
                    sac = max(np.fabs(fits['a_c']-fits_minus['a_c']),np.fabs(fits['a_c']-fits_plus['a_c']))
                if fits['b_c'] and fits_minus['b_c'] and fits_plus['b_c']:
                    sbc = max(np.fabs(fits['b_c']-fits_minus['b_c']),np.fabs(fits['b_c']-fits_plus['b_c']))
                if fits['a_h'] and fits_minus['a_h'] and fits_plus['a_h']:
                    sah = max(np.fabs(fits['a_h']-fits_minus['a_h']),np.fabs(fits['a_h']-fits_plus['a_h']))
                if fits['b_h'] and fits_minus['b_h'] and fits_plus['b_h']:
                    sbh = max(np.fabs(fits['b_h']-fits_minus['b_h']),np.fabs(fits['b_h']-fits_plus['b_h']))
                self.cool_fits.append([fits['a_c'],fits['b_c'],[sac,sbc]]),self.hot_fits.append([fits['a_h'],fits['b_h'],[sah,sbh]])
                
            elif self.fit_method is 'fit_mean_weighted':
                bound_arrays = self.bounds(self.temp_mean[i],self.em_mean[i],self.sigma[i])
                fits = self.branch_fit(bound_arrays['temp_cool'], bound_arrays['dem_cool'], bound_arrays['temp_hot'], bound_arrays['dem_hot'], sigma_cool=bound_arrays['sigma_cool'], sigma_hot=bound_arrays['sigma_hot'])
                self.cool_fits.append([fits['a_c'],fits['b_c'],fits['s_c']]),self.hot_fits.append([fits['a_h'],fits['b_h'],fits['s_h']])
                
            else:
                raise ValueError("Unrecognized fit method option.")
            
            
    def bounds(self,temp,dem,sigma,**kwargs):
        """Create bounded hot and cool branches from given hot and cool branch limits (or default values); interpolation over EM curves should be done before this step."""
        
        #find limits over which temperature will be calculated
        slope_limits = self.fit_limits(temp,dem)
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
            try:
                sigma_cool = np.ones(len(temp_cool))
            except TypeError:
                pass
            absolute_sigma_cool = False
            
        if 'sigma_hot' in kwargs:
            sigma_hot = kwargs['sigma_hot']
            absolute_sigma_hot = True
        else:
            try:
                sigma_hot = np.ones(len(temp_hot))
            except TypeError:
                pass
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
        """Calculate limits over which fit is calculated depending on input option."""
        
        if self.lim_method is 'dynamic':
            #isolate hot branch
            i_hot = np.where(temp > temp[np.argmax(em)])[0]
            em_hot = em[i_hot]
            temp_hot = temp[i_hot]
            #set upper and lower temperature bounds for 80% drop in EM from peak
            th_upper = temp[i_hot[np.where(em_hot>self.max_percent_drop*np.max(em))[0]][-1]]
            th_lower = th_upper - self.delta_t
            tc_lower = self.slope_limits['cool_lower']
            tc_upper = self.slope_limits['cool_upper']
        elif self.lim_method is 'peak':
            tc_upper,th_lower = temp[np.argmax(dem)],temp[np.argmax(dem)]
            tc_lower = slope_limits['cool_upper'] + self.cool_diff
            th_upper = slope_limits['hot_lower'] + self.hot_diff
        elif self.lim_method is 'static':
            th_upper = self.slope_limits['hot_upper']
            th_lower = self.slope_limits['hot_lower']
            tc_lower = self.slope_limits['cool_lower']
            tc_upper = self.slope_limits['cool_upper']
        else:
            raise ValueError("Unrecognized limit method calculation.")
        
        if self.verbose and self.lim_method is not 'static':
            print("T_cool_upper = "+str(tc_upper)+" K")
            print("T_cool_lower = "+str(tc_lower)+" K")
            print("T_hot_upper = "+str(th_upper)+" K")
            print("T_hot_lower = "+str(th_lower)+" K")
        
        return {'cool_upper':tc_upper,'cool_lower':tc_lower,'hot_upper':th_upper,'hot_lower':th_lower}
          