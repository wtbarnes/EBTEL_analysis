#em_binner.py
#Will Barnes
#14 October 2015

#Class to construct emission measure (EM) distributions from temperature and density profiles. Incorporate effective temperature profiles due to non-equilibrium ionization.

#Import needed modules
import logging
import numpy as np
import matplotlib.pyplot as plt


class EM_Binner(object):

    def __init__(self,loop_length,time=None,temp=None,density=None,read_abundances=False,**kwargs):
        #set loop length as member variable
        self.loop_length = loop_length
        #set up logger
        self.logger = logging.getLogger(type(self).__name__)
        #Get abundances
        if(read_abundances):
            self.logger.warning("TODO: import abundances")
            pass
        #set data
        if time is not None and temp is not None and density is not None:
            self.set_data(time,temp,density)
        else:
            self.logger.warning("Parameters not yet set. Run self.set_data(t,T,n)")
            
    def set_data(self,time,temp,density):
        """Add parameters as class attributes"""
        self.time = time
        self.temp = temp
        self.density = density

    def make_T_bins(self,logT_a=4.0,logT_b=8.5,delta_logT=0.01):
        """Build temperature bins in log_10 for creating EM distribution"""

        self.delta_logT = delta_logT
        self.T_em = np.logspace(logT_a,logT_b,int((logT_b - logT_a)/self.delta_logT))
        #Add right edge for histogram bins
        self.T_em_histo_bins = np.append(self.T_em,self.T_em[-1]*10.**self.delta_logT)

    def _emission_measure_calc(self,n):
        """Calculate emission measure distribution"""

        return self.loop_length*n**2
        
    def _differential_emission_measure_calc(self,n,tmin,tmax):
        """Calculate differential emission measure distribution"""
        
        em = self.emission_measure_calc(n)
        #EBTEL method for calculating coronal DEM
        tmax = np.log10(tmax)
        tmin = np.log10(tmin)
        jmax = (tmax - 4.0)*100.0
        jmin = (tmin - 4.0)*100.0
        delta_t = 1.0e+4*(10.0**((jmax + 0.5)/100.0) - 10.0**((jmin - 0.5)/100.0));
        return em/delta_t

    def _coronal_limits(self,T):
        """Find limits of corona in log temperature space."""

        #Use EBTEL method for calculating coronal temperature bounds
        #logT_C_a = np.log10(2.0/3.0*T)
        T_C_a = 8.0/9.0*T
        T_C_b = 10.0/9.0*T
        return T_C_a,T_C_b

    def build_em_dist(self,build_mat=False):
        """Create EM distribution from temperature arrays. Build for both T and T_eff"""

        try:
            self.T_em
        except AttributeError:
            self.logger.info("Temperature bins not yet created. Building now with default values.")
            self.make_T_bins()
            
        #allocate space for DEM, EM matrices
        if build_mat:
            self.em_mat = np.zeros([len(self.time),len(self.T_em)])
            self.dem_mat = np.zeros([len(self.time),len(self.T_em)])

        #Flattened EM and temp lists for easily building histograms
        self.em_flat = []
        self.T_em_flat = []

        #Calculate time weights
        w_tau = np.gradient(self.time)/np.sum(np.gradient(self.time))

        #Loop over time
        for i in range(len(self.time)):
            #calculate coronal temperature bounds at time i
            Ta,Tb = self._coronal_limits(self.temp[i])
            #find coronal indices in logT
            iC = np.where((self.T_em >= Ta) & (self.T_em <= Tb))
            if len(iC) > 0:
                if build_mat:
                    #add entries to dem,em matrices
                    self.dem_mat[i,iC[0]] = self._differential_emission_measure_calc(self.density[i],Ta,Tb)
                    self.em_mat[i,iC[0]] = self._emission_measure_calc(self.density[i])
                #append coronal temperatures to temperature list
                self.T_em_flat.extend(self.T_em[iC[0]])
                #append emission measure weighted by timestep to emission measure list
                self.em_flat.extend(len(iC[0])*[w_tau[i]*self._emission_measure_calc(self.density[i])])
                
        #cast as numpy arrays for convenience
        self.T_em_flat = np.array(self.T_em_flat)
        self.em_flat = np.array(self.em_flat)
