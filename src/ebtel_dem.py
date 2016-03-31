#ebtel_dem.py

#Will Barnes
#13 May 2015

#Import needed modules
import os
import pickle
import numpy as np
import logging
import em_binner as emb
from scipy.optimize import curve_fit


class DEMProcess(object):
    """Class for processing and importing  EBTEL data"""

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


    def import_raw(self,Tn,save_to_file=None,read_teff=False,**kwargs):
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
                    t_index = 1
                    if 'electron' in tn_path or 'ion' in tn_path: n_index += 1
                    if read_teff:
                        n_index = 3
                        t_index = 2
                    t,T,n = data[:,0],data[:,t_index],data[:,n_index]
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
                h,b = np.histogram(em[i]['T'],bins=em[i]['bins'],weights=em[i]['em'])
                tmp_mat[i,:] = h
                tmp.append({'hist':h,'bin_centers':np.diff(b)/2. + b[0:-1]})
            #NOTE:Assuming all temperature arrays are the same!
            bin_centers = np.diff(em[0]['bins'])/2. + em[0]['bins'][0:-1]
            #calculate and save stats
            self.em_stats.append({'T_mean':bin_centers, 'em_mean':np.mean(tmp_mat,axis=0), 'em_std':np.std(tmp_mat,axis=0), 'em_max_mean':np.mean(np.max(tmp_mat,axis=1)), 'em_max_std':np.std(np.max(tmp_mat,axis=1)), 'T_max_mean':np.mean(bin_centers[np.argmax(tmp_mat,axis=1)]), 'T_max_std': np.std(bin_centers[np.argmax(tmp_mat,axis=1)]) })
            #save binned em
            self.em_binned.append(tmp)


    def diagnose_em(self,cool_limits=None,hot_limits=None,t_ratio_cool=[10**6.],t_ratio_hot=[10**7.],calc_slope=True,calc_ratio=True):
        """Fit binned emission measure histograms on hot and cool sides"""

        if not hasattr(self,'em_binned'):
            raise AttributeError("EM histograms not yet binned. Run self.calc_stats() before calculating EM diagnostics.")
            
        if len(t_ratio_cool) != len(t_ratio_hot):
            raise ValueError("Temperature ratio lists must have same length.")
            
        #zip cool and hot temperatures together and make public
        self.em_ratio_tpairs=[[tc,th] for tc,th in zip(t_ratio_cool,t_ratio_hot)]

        self.diagnostics=[]

        for upper in self.em_binned:
            tmp = []
            for lower in upper:
                if calc_slope:
                    #split the curve
                    t_cool,em_cool,t_hot,em_hot = self._split_branch(lower['bin_centers'],lower['hist'])
                    #calculate limits if necessary
                    cool_lims,hot_lims = cool_limits,hot_limits
                    cool_lims = self._find_fit_limits(t_cool,em_cool,cool_lims,temp_opt='cool')
                    hot_lims = self._find_fit_limits(t_hot,em_hot,hot_lims)
                    #compute fit values
                    dc = self._fit_em_branch(t_cool,em_cool,cool_lims)
                    dh = self._fit_em_branch(t_hot,em_hot,hot_lims)
                else:
                    dc,dh=None,None
                if calc_ratio:
                    #compute em ratio
                    tmp_ratio=[self._calc_em_ratio(tp[0],tp[1],lower['bin_centers'],lower['hist']) for tp in self.em_ratio_tpairs]
                else:
                    tmp_ratio=[None for _ in self.em_ratio_tpairs]
                #compute em ratio and store
                tmp.append({'cool':dc,'hot':dh,'ratio':tmp_ratio})
            self.diagnostics.append(tmp)
            

    def calc_diagnostic_stats(self,mc_threshold=0.9,**kwargs):
        """Calculate fit statistics"""

        if not hasattr(self,'diagnostics'):
            raise AttributeError("Before computing diagnostic statistics, run self.diagnose_em() to calculate fits,ratio diagnostics.")

        self.diagnostics_stats = []
        
        for upper in self.diagnostics:
            tmp_cool,tmp_hot,tmp_lim_cool,tmp_lim_hot,tmp_ratio = [],[],[],[],[]
            for lower in upper:
                tmp_ratio.append(lower['ratio'])
                if lower['cool'] is not None:
                    tmp_cool.append([lower['cool']['a'],lower['cool']['b']])
                    tmp_lim_cool.append(lower['cool']['limits'])
                if lower['hot'] is not None:
                    tmp_hot.append([lower['hot']['a'],lower['hot']['b']])
                    tmp_lim_hot.append(lower['hot']['limits'])

            #reject small number statistics
            #slopes
            if float(len(tmp_cool))/float(len(upper)) >= mc_threshold:
                dc = {'a':np.mean(np.array(tmp_cool),axis=0)[0], 'b':np.mean(np.array(tmp_cool),axis=0)[1], 'sigma_a':np.std(np.array(tmp_cool),axis=0)[0],'limits':np.mean(np.array(tmp_lim_cool),axis=0)}
            else:
                dc = None
            if float(len(tmp_hot))/float(len(upper)) >= mc_threshold:
                dh = {'a':np.mean(np.array(tmp_hot),axis=0)[0], 'b':np.mean(np.array(tmp_hot),axis=0)[1], 'sigma_a':np.std(np.array(tmp_hot),axis=0)[0],'limits':np.mean(np.array(tmp_lim_hot),axis=0)}
            else:
                dh = None
            #ratio
            tmp_ratio_stats={'mean':[],'sigma':[],'tpairs':self.em_ratio_tpairs}
            for i in range(len(self.em_ratio_tpairs)):
                filtered_ratios=np.array([r for r in np.array(tmp_ratio)[:,i] if r is not None])
                if float(len(filtered_ratios))/float(len(upper)) >= mc_threshold:
                    tmp_ratio_stats['mean'].append(np.mean(filtered_ratios))
                    tmp_ratio_stats['sigma'].append(np.std(filtered_ratios))
                else:
                    tmp_ratio_stats['mean'].append(None)
                    tmp_ratio_stats['sigma'].append(None)
            
            self.diagnostics_stats.append({'cool':dc, 'hot':dh, 'ratio':tmp_ratio_stats})
            
            
    def _calc_em_ratio(self,t_cool,t_hot,t,em):
        """Calculate EM ratio from EM(t_cool) and EM(t_hot)"""
        
        #check on temperature
        if t_cool > t_hot:
            self.logger.warning("t_cool > t_hot; Reassigning such that t_cool < t_hot.")
            tmp=t_cool
            t_cool=t_hot
            t_hot=tmp
        if t_hot > t[-1] or t_cool < t[0]:
            raise IndexError("Hot and/or cool temperature out of range. Select hot/cool T values in (%.3e,%.3e)"%(t[0],t[-1]))
            
        #interpolate
        em_interp=np.interp([t_cool,t_hot],t,em)
        if em_interp[0] < self.em_cutoff or em_interp[1] < self.em_cutoff:
            self.logger.debug("Interpolated EM below EM threshold. Returning None.")
            return None
        else:
            return em_interp[1]/em_interp[0]


    def _find_fit_limits(self,t,em,limits,temp_opt='hot'):
        """Calculate and check temperature limits for fitting"""

        if limits is None:
            #dynamic index choice
            #f,tnew = interp1d(t[em>self.em_cutoff],em[em>self.em_cutoff]),np.linspace(t[em>self.em_cutoff][0],t[em>self.em_cutoff][-1],10)
            #dEmdT_mp = np.gradient(f(tnew),np.gradient(tnew))[int(len(tnew)/2)]
            #hc_var = int(dEmdT_mp/np.fabs(dEmdT_mp))
            hc_var = -1.0
            if temp_opt == 'cool':
                hc_var = 1
            #get temperature corresponding to just above cutoff
            indices = np.where(em < self.em_cutoff)[0]
            t2 = t[indices[-int((hc_var+1)/2)]+hc_var]
            #get temperature corresponding to just below peak EM
            indices = np.where(em<self.em_peak_falloff*np.max(em))[0]
            t1 = t[indices[-int((hc_var+1)/2)]]
            limits = sorted([t1,t2])
            self.logger.debug("Calculated fit limits: (logT1,logT2)=(%.2f,%.2f)"%(np.log10(limits[0]),np.log10(limits[1])))

        return self._check_fit_limits(t,em,limits)


    def _check_fit_limits(self,t,em,limits):
        """Check validity of fit limits"""

        if limits[0] < t[0] or limits[1] > t[-1]:
            self.logger.warning("Fit limits outside of temperature range. %.2f < %.2f MK or %.2f > %.2f MK. Returning None."%(limits[0]/1e+6,t[0]/1e+6,limits[1]/1e+6,t[-1]/1e+6))
            return None

        emnew = np.interp(np.array([limits[0],limits[1]]),t,em)
        if emnew[0] < self.em_cutoff or emnew[1] < self.em_cutoff:
            self.logger.warning("Fit limits below EM threshold, logEM_thresh=%.2f. logEM0,logEM1 = (%.2f,%.2f). Returning None."%(np.log10(self.em_cutoff),np.log10(emnew[0]),np.log10(emnew[1])))
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

        return {'a':pars[0],'b':pars[1],'sigma_a':np.sqrt(np.diag(covar))[0],'limits':limits}


    def _fit_function(self,x,a,b):
        """Function to use for fitting branches of emission measure distribution"""
        return b + a*x
