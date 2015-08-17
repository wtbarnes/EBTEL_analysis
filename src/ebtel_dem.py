#ebtel_dem.py

#Will Barnes
#13 May 2015

#Import needed modules
import numpy as np
from scipy.optimize import curve_fit

class DEMProcess(object):
    
    em_cutoff = 23.0
    em_max_eps_percent = 0.999
    
    def __init__(self,root_dir,species,alpha,loop_length,tpulse,solver,**kwargs):
        #set up paths
        child_path = root_dir+species+'_heating_runs/alpha'+str(alpha)+'/data/'
        self.file_path = 'ebtel_L'+str(loop_length)+'_tn%d_tpulse'+str(tpulse)+'_'+solver
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
            eol_flag=False
            counter=0
            #build wait-time specific path
            tn_path = self.root_path%Tn[i]
            while eol_flag is False:
                try:
                    temp = np.loadtxt(tn_path+'/'+self.file_path%Tn[i]+'_'+str(counter)+'_dem.txt')
                    temp[np.where(np.isnan(temp))] = -np.inf
                    temp_em.append(temp[:,0])
                    em.append(temp[:,4])
                    counter += 1
                except:
                    if self.verbose:
                        print("Unable to process file for Tn = "+str(Tn[i])+", run = "+str(counter))
                        print("Reached end of list or there was an error reading the file.")
                    
                    eol_flag=True
                    pass
                    
            self.temp_em.append(temp_em)
            self.em.append(em)
            
            
    def interp_and_filter(self,**kwargs):
        """Interpolate and filter mean and standard deviation for EM, T arrays; this step is mandatory for later slope calculations."""
        
        if not self.em_mean or not self.em_std:
            raise ValueError("Mean and standard deviation must be calculated before they can be interpolated.")
        
        for i in range(len(self.em_mean)):
            #find cutoff index
            inf_index = np.where(self.em_mean[i] > self.em_cutoff)
            #reshape temperature and interpolate emission measure
            temp_new = np.linspace(self.temp_mean[i][inf_index[0][0]],self.temp_mean[i][inf_index[0][-1]],2000)
            dem_new = np.interp(temp_new,self.temp_mean[i][inf_index[0][0]:inf_index[0][-1]],self.em_mean[i][inf_index[0][0]:inf_index[0][-1]])
            sigma_new = np.interp(temp_new,self.temp_mean[i][inf_index[0][0]:inf_index[0][-1]],self.em_std[i][inf_index[0][0]:inf_index[0][-1]])
            #reassign to nested list
            self.temp_mean[i] = temp_new
            self.em_mean[i] = dem_new
            self.em_std[i] = sigma_new
                
                
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
    
    def __init__(self,em,temp,sigma,**kwargs):
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
        #define variables to be used later
        self.cool_fits = []
        self.hot_fits = []
        
        
    def many_fits(self,**kwargs):
        """Calculate fits to hot and cool branches for all EM and T data sets"""
        
        for i in range(len(self.em)):
            acl = []
            ahl = []
            #redefine slope limits if kwargs
            if self.slope_limits:
                if len(self.slope_limits) > 4:
                    slope_limits = self.slope_limits[i]
                else:
                    slope_limits = self.slope_limits
                    
            else:
                slope_limits = []
                
            bound_arrays = self.bounds(self.temp_em[i],self.em[i],self.sigma_em[i],slope_limits)
            ac,bc,sc,ah,bh,sh = self.branch_fit(self.temp_em[i],self.em[i],bound_arrays)
            self.cool_fits.append([ac,bc,sc]),self.hot_fits.append([ah,bh,sh])
                            
            
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
        
        
    def branch_fit(self,temp,dem,bound_arrays,**kwargs):
        """Linear fit to hot and cool branches of EM curve using hot and cool branches constructed according to hot and cool limits."""
        
        #Function for linear fit
        def linear_fit(x,a,b):
            return a*x + b
            
        #Check if inside interpolated array and calculate slope
        #cool
        if bound_arrays['temp_cool'] is False:
            a_coolward = False
            b_coolward = False
            sigma_coolward = False
        else:
            pars_cool,covar_cool = curve_fit(linear_fit,bound_arrays['temp_cool'],bound_arrays['dem_cool'],sigma=bound_arrays['sigma_cool'])
            a_coolward,b_coolward = pars_cool[0],pars_cool[1]
            sigma_coolward = np.sqrt(np.diag(covar_cool))
            
        #hot
        if bound_arrays['temp_hot'] is False:
            a_hotward = False
            b_hotward = False
            sigma_hotward = False
        else:
            pars_hot,covar_hot = curve_fit(linear_fit,bound_arrays['temp_hot'],bound_arrays['dem_hot'],sigma=bound_arrays['sigma_hot'])
            a_hotward,b_hotward = pars_hot[0],pars_hot[1]
            sigma_hotward = np.sqrt(np.diag(covar_hot))
            
        return a_coolward,b_coolward,sigma_coolward,a_hotward,b_hotward,sigma_hotward
        
        
    def branch_fit_statistics(self,**kwargs):
        """Compute mean and standard deviation for fit parameters to hot and cool branches"""
        
        if not self.a_cool or not self.a_hot:
            raise ValueError("Before computing statistics of slopes, first calculate slopes using self.many_slopes()")
            
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