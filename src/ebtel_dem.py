#ebtel_dem.py

#Will Barnes
#13 May 2015

#Import needed modules
import os
import pickle
import numpy as np
import logging
import itertools
import em_binner as emb
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


class DEMProcess(object):

    def __init__(self, root_dir, species, alpha, loop_length, tpulse, solver, scaling_suffix='', aspect_ratio_factor=1.0, em_cutoff=1e+25, em_peak_falloff=0.99, **kwargs):
        """Constructor for process class"""
        #set up paths
        child_path = os.path.join(root_dir, species+'_heating_runs', 'alpha'+str(alpha), 'data')
        self.file_path = 'ebtel_L'+str(loop_length)+'_tn%d'+scaling_suffix+'_tpulse'+str(tpulse)+'_'+solver
        self.root_path = os.path.join(child_path, self.file_path)
        #configure logger
        self.logger = logging.getLogger(type(self).__name__)
        #configure keyword arguments
        self.tpulse = tpulse
        self.aspect_ratio_factor = aspect_ratio_factor
        self.em_cutoff = em_cutoff
        self.em_peak_falloff = em_peak_falloff
        #instantiate binner class
        self.binner = emb.EM_Binner(2.*loop_length*1.e+8)


    def import_raw(self,Tn,save_to_file=None,**kwargs):
        """Import all runs for given Tn waiting time values; calculate EM distributions from t,T,n."""

        self.em = []
        
        for i in range(len(Tn)):
            #initialize lists
            tmp = []
            #initialize counter and flag
            counter=0
            #build wait-time specific path
            tn_path = self.root_path%Tn[i]
            for pfile in os.listdir(tn_path):
                if 'heat_amp' not in pfile and 'dem' not in pfile:
                    self.logger.debug("Reading %s"%pfile)
                    data = np.loadtxt(os.path.join(tn_path,pfile))
                    n_index = 2
                    if 'electron' in tn_path: n_index += 1
                    t,T,n = data[:,0],data[:,1],data[:,n_index]
                    #calculate emission measure distribution
                    self.binner.set_data(t,T,n)
                    self.binner.build_em_dist()
                    #save data
                    tmp.append({'T':self.binner.T_em_flat, 'em':self.binner.em_flat/self.aspect_ratio_factor, 'bins':self.binner.T_em_histo_bins})
                    #increment counter
                    counter += 1
                else:
                    continue
                    
            #Estimate percentage of files read
            self.logger.info("Tn = %d s finished, Estimated total # of events simulated: %.2f %%"%(Tn[i], counter*int(np.ceil(t[-1]/(self.tpulse+Tn[i])))))
            
            #append to top level list
            self.em.append(tmp)
            
        if save_to_file is not None:
            with open(save_to_file,'wb') as f:
                pickle.dump(self.em,f)
                
                
    def import_from_file(self,pickled_file):
        """Import level 1 pickled results"""
        with open(pickled_file,'rb') as f:
            self.em = pickle.load(f)


    def calc_stats(self,**kwargs):
        """Calculate mean, standard deviation and max for EM and T."""

        if not hasattr(self,'em'):
            raise AttributeError("Before computing EM statistics, run self.import_raw() to calculate EM data.")

        self.em_stats, self.em_binned = [],[]
        
        for em in self.em:
            #initialize list
            tmp = []
            #preallocate memory in matrix
            tmp_mat = np.zeros([len(em),len(em[0]['bins'])-1])
            #loop over all mc runs
            i = 0
            for i in range(len(em)):
                h,_ = np.histogram(em[i]['T'],bins=em[i]['bins'],weights=em[i]['em'])
                tmp_mat[i,:] = h
                tmp.append({'hist':h,'bin_centers':np.diff(em[i]['bins'])/2. + em[i]['bins'][0:-1]})
            #compute statistics and bin centers (i.e. the temperature array)
            mean_tmp,std_tmp = np.mean(tmp_mat,axis=0),np.std(tmp_mat,axis=0)
            bin_centers = np.diff(em[0]['bins'])/2. + em[0]['bins'][0:-1]
            #save stats
            self.em_stats.append({'em_mean':mean_tmp, 'em_std':std_tmp, 'em_max':np.max(mean_tmp), 'T_max':bin_centers[np.argmax(mean_tmp)], 'T_mean':bin_centers})
            #save binned em
            self.em_binned.append(tmp)
            
    
    def fit_em(self,cool_limits=None,hot_limits=None):
        """Fit binned emission measure histograms on hot and cool sides"""
        
        if not hasattr(self,'em_binned'):
            raise AttributeError("EM histograms not yet binned. Run self.calc_stats() before fitting EMs.")
    
        self.fits = []
        
        for upper in self.em_binned:
            tmp = []
            for lower in upper:
                #split the curve
                t_cool,em_cool,t_hot,em_hot = self._split_branch(lower['bin_centers'],lower['hist'])
                #calculate limits if necessary
                cool_lims,hot_lims = cool_limits,hot_limits
                cool_lims = self._find_fit_limits(t_cool,em_cool,cool_lims)
                hot_lims = self._find_fit_limits(t_hot,em_hot,hot_lims)
                #compute fit values
                dc = self._fit_em_branch(t_cool,em_cool,cool_lims)
                dh = self._fit_em_branch(t_hot,em_hot,hot_lims)
                #store
                tmp.append({'cool':dc,'hot':dh})
            self.fits.append(tmp)
            
            
    def calc_fit_stats(self,mc_threshold=0.9,**kwargs):
        """Calculate fit statistics"""
        
        if not hasattr(self,'fits'):
            raise AttributeError("Before computing fit statistics, run self.fit_em() to calculate fits to EM distributions.")
                    
        self.fits_stats = []
        
        for upper in self.fits:
            tmp_cool,tmp_hot,tmp_lim_cool,tmp_lim_hot = [],[],[],[]
            for lower in upper:
                if lower['cool'] is not None:
                    tmp_cool.append([lower['cool']['a'],lower['cool']['b']])
                    tmp_lim_cool.append(lower['cool']['limits'])
                if lower['hot'] is not None:
                    tmp_hot.append([lower['hot']['a'],lower['hot']['b']])
                    tmp_lim_hot.append(lower['hot']['limits'])
                    
            #reject small number statistics
            if float(len(tmp_cool))/float(len(upper)) >= mc_threshold:
                dc = {'a':np.mean(np.array(tmp_cool),axis=0)[0], 'b':np.mean(np.array(tmp_cool),axis=0)[1], 'sigma_a':np.std(np.array(tmp_cool),axis=0)[0],'limits':np.mean(np.array(tmp_lim_cool),axis=0)}
            else:
                dc = None
            if float(len(tmp_hot))/float(len(upper)) >= mc_threshold:
                dh = {'a':np.mean(np.array(tmp_hot),axis=0)[0], 'b':np.mean(np.array(tmp_hot),axis=0)[1], 'sigma_a':np.std(np.array(tmp_hot),axis=0)[0],'limits':np.mean(np.array(tmp_lim_hot),axis=0)}
            else:
                dh = None
                
            self.fits_stats.append({'cool':dc,'hot':dh})
            
    
    def _find_fit_limits(self,t,em,limits):
        """Trim temperature and EM for fitting"""
        
        if limits is None:
            #just off the peak temperature
            t1 = self.em_peak_falloff*t[np.argmax(em)]
            #get the last entry for the cool side and the first entry on the hot side
            dEmdT_mp = np.gradient(em[em>self.em_cutoff],np.gradient(t[em>self.em_cutoff]))[int(len(em[em>self.em_cutoff])/2)]
            hc_var = int(dEmdT_mp/np.fabs(dEmdT_mp))
            indices = np.where(em < np.max(em)/(1e+2))[0]
            t2 = t[indices[-int((hc_var+1)/2)]+hc_var]
            limits = sorted([t1,t2])

        return self._check_fit_limits(t,em,limits)
        
        
    def _check_fit_limits(self,t,em,limits):
        """Check validity of fit limits"""
        
        if limits[0] < t[0] or limits[1] > t[-1]:
            self.logger.warning("Fit limits outside of temperature range. %.2f < %.2f MK or %.2f > %.2f MK. Returning None."%(limits[0]/1e+6,t[0]/1e+6,limits[1]/1e+6,t[-1]/1e+6))
            return None
            
        f = interp1d(t,em,kind='cubic')
        emnew = f(np.array([limits[0],limits[1]]))
        if emnew[0] < self.em_cutoff or emnew[1] < self.em_cutoff:
            self.logger.warning("Fit limits below EM threshold. logEM0,logEM1 = (%.2f,%.2f). Returning None."%(np.log10(emnew[0],np.log10(emnew[1]))))
            return None
        
        return limits
    
    
    def _split_branch(self,t,em):
        """Split EM into hot and cool branch for fitting"""
        
        max_t = t[np.argmax(em)]
        return t[t<=max_t],em[t<=max_t],t[t>max_t],em[t>max_t]
    
    
    def _fit_em_branch(self,t,em,limits):
        """Fit EM branch to power-law function for given bounds"""
        
        #Check for invalid fitting limits
        if not limits:
            self.logger.debug("Returning None fit parameters for None fitting limits.")
            return None
        
        #Clip temperature and emission measure
        t_fit = t[(t>=limits[0]) & (t<=limits[1])]
        em_fit = em[(t>=limits[0]) & (t<=limits[1])]
        #Fitting
        pars,covar = curve_fit(self._fit_function,np.log10(t_fit),np.log10(em_fit))
        
        return {'a':pars[0],'b':10.**pars[1],'sigma_a':np.sqrt(np.diag(covar))[0],'limits':limits}
    
    
    def _fit_function(self,x,a,b):
        """Function to use for fitting branches of emission measure distribution"""
        return b + a*x



#TODO:combine analyze and process classes, slim down fitting routines
class DEMAnalyze(object):

    cool_diff = -0.6
    hot_diff = 0.4

    def __init__(self,em,temp,em_mean,temp_mean,sigma,**kwargs):
        #get nested lists with EM and T values
        self.em = list(em)
        self.temp= list(temp)
        self.em_mean = list(em_mean)
        self.temp_mean = list(temp_mean)
        self.sigma = list(sigma)
        #configure logger
        self.logger = logging.getLogger(type(self).__name__)
        #keyword arguments
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
        #define lists to store all list values
        self.cool_fits_all = []
        self.hot_fits_all = []


    def interp_and_filter(self,**kwargs):
        """Interpolate and filter mean and standard deviation for EM, T arrays; this step is mandatory for later slope calculations.
        """

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
                #store all values
                self.cool_fits_all.append([s[0] for s in cool_temp])
                self.hot_fits_all.append([s[0] for s in hot_temp])

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
        if len(i_cool_lower[0]) > 0 and len(i_cool_upper[0]) > 0:
            temp_cool = temp[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
            dem_cool = dem[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
            sigma_cool = sigma[(i_cool_lower[0][-1] + 1):(i_cool_upper[0][0] - 1)]
        else:
            self.logger.debug("Cool bound out of range, T = %.2f > T_limit = %.2f"%(temp[0],slope_limits['cool_lower']))
            self.logger.debug("or T_upper_limit = %.2f > T_max = %.2f"%(slope_limits['cool_upper'],slope_limits['hot_lower']))
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
            self.logger.debug("Hot bound out of range, T = %.2f < T_limit = %.2f"%(temp[-1],slope_limits['hot_upper']))
            self.logger.debug("or T_lower_limit = %.2f < T_max = %.2f"%(slope_limits['hot_lower'],slope_limits['cool_upper']))
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
            th_lower = temp[i_hot[np.where(em_hot>0.99*np.max(em))[0]][-1]]#th_upper - self.delta_t
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
