#em_binner.py
#Will Barnes
#14 October 2015

#Class to construct emission measure (EM) distributions from temperature and density profiles. Incorporate effective temperature profiles due to non-equilibrium ionization.

#Import needed modules
import logging
import numpy as np
import matplotlib.pyplot as plt


class EM_Binner(object):

    def __init__(self,time,temp,density,loop_length,read_abundances=False,**kwargs):
        #set loop length as member variable
        self.loop_length = loop_length
        #Slice array and set member arrays
        self.time = time
        self.temp = temp
        self.density = density
        #set up logger
        self.logger = logging.getLogger(type(self).__name__)
        #Get abundances
        if(read_abundances):
            self.logger.warning("TODO: import abundances")
            pass

    def logT_bins(self,logT_a=4.0,logT_b=8.5,delta_logT=0.01):
        """Build temperature bins in log_10 for creating EM distribution"""

        self.delta_logT = delta_logT
        self.logT_EM = np.arange(logT_a,logT_b,delta_logT)
        #Add right edge for histogram bins
        self.logT_EM_histo_bins = np.append(self.logT_EM,self.logT_EM[-1]+self.delta_logT)

    def emission_measure_calc(self,n):
        """Calculate emission measure distribution"""

        return self.loop_length*n**2
        
    def differential_emission_measure_calc(self,n,tmin,tmax):
        """Calculate differential emission measure distribution"""
        
        em = self.emission_measure_calc(n)
        #EBTEL method for calculating coronal DEM
        jmax = (tmax - 4.0)*100.0
        jmin = (tmin - 4.0)*100.0
        delta_t = 1.0e+4*(10.0**((jmax + 0.5)/100.0) - 10.0**((jmin - 0.5)/100.0));
        return em/delta_t

    def coronal_limits(self,T):
        """Find limits of corona in log temperature space."""

        #Use EBTEL method for calculating coronal temperature bounds
        #logT_C_a = np.log10(2.0/3.0*T)
        logT_C_a = np.log10(8.0/9.0*T)
        logT_C_b = np.log10(10.0/9.0*T)
        return logT_C_a,logT_C_b

    def build_em_dist(self):
        """Create EM distribution from temperature arrays. Build for both T and T_eff"""

        try:
            self.logT_EM
        except AttributeError:
            self.logger.warning("Temperature bins not yet created. Building now with default values.")
            self.logT_bins()
            
        #allocate space for DEM, EM matrices
        self.em_mat = np.zeros(len(self.time),len(self.logT_EM))
        self.dem_mat = np.zeros(len(self.time),len(self.logT_EM))

        #Flattened EM and temp lists for easily building histograms
        self.em_flat = []
        self.logT_em_flat = []

        #Calculate time weights
        w_tau = np.gradient(self.time)/np.sum(np.gradient(self.time))

        #Loop over time
        for i in range(len(self.time)):
            #calculate coronal temperature bounds at time i
            logTa,logTb = self.coronal_limits(self.temp[i])
            #find coronal indices in logT
            iC = np.where((self.logT_EM >= logTa) & (self.logT_EM <= logTb))
            if len(iC) > 0:
                #add entries to dem,em matrices
                self.dem_mat[i,ic[0]] = self.differential_emission_measure_calc(self.density[i],logTa,logTb)
                self.em_mat[i,ic[0]] = self.emission_measure_calc(self.density[i])
                #append coronal temperatures to temperature list
                self.logT_em_flat.extend(self.logT_EM[iC[0]])
                #append emission measure weighted by timestep to emission measure list
                self.em_flat.extend(len(iC[0])*[w_tau[i]*self.emission_measure_calc(self.density[i])])
