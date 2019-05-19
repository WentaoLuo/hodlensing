import numpy as np
import camb
from camb import model, initialpower
from scipy import integrate
import scipy.interpolate as interp

pi = np.pi
#-----------------------------------------------
class correlation(object):
   def __init__(self,logMh=None,Zred=None,fcen=None,Ombh2=None,Omch2=None,Hnull=None):
      rho_crit= 9.47e-27
      ckm     = 3.24078e-20
      ckg     = 5.0e-31
      rhoc    = rho_crit*ckg*10e+9/ckm/ckm/ckm
      self.rhom    = rhoc*omega_m

      self.logMh = logMh
      self.Zred  = Zred
      self.Ombh2 = Ombh2
      self.Omch2 = Omch2
      self.Hnull = Hnull
      self.fcen  = fcen
      self.fsat  = 1.0-self.fcen
      self.nps   = 0.965
      self.sigma8= 0.815
      self.h     = 1.0
      self.alphas= -0.04

      self.con   = con
      self.r200 = (10.0**self.logMh*3.0/200./self.rhom/pi)**(1./3.)
      self.rs   = self.r200/self.con
      self.delta= (200./3.0)*(self.con**3)\
                 /(np.log(1.0+self.con)-self.con/(1.0+self.con))
      self.amp  = 2.0*self.rs*self.delta*rhoc*1e-13

   def galaxybias(self):
      omega_m = self.Ombh2/0.672/0.672
      sigma8  = 0.815
      h       = 1.0
      Mnl = 8.73*10e+12
      Mh  = 10.0**self.logMh
      xx  = Mh/Mnl
      b0  = 0.53+0.39*xx**0.45+(0.13/(40.0*xx+1.0))\
           +5.0*0.0004*xx**1.5
      bias= b0+\
           np.log10(xx)*(0.4*(omega_m-0.3+self.nps-1)+\
           0.3*(self.sigma8-0.9+self.h-0.7)+0.8*self.alphas)

      return bias

   def twohalo_corr(self):
     pars  = camb.CAMBparams()
     pars.set_cosmology(H0=self.Hnull,ombh2=self.Ombh2,omch2=self.Omch2)
     pars.set_dark_energy()
     pars.InitPower.set_params(ns=0.965)
     pars.set_matter_power(redshifts=[0.0,self.Zred],kmax=100.0)
     # Linear Spectra----------------------------------------
     pars.NonLinear=model.NonLinear_none
     results       =camb.get_results(pars)
     kh,znon,pk    =results.get_matter_power_spectrum(minkh=1e-4,maxkh=100.0,npoints=1024)
     kmax   = np.max(kh)
     kmin   = np.min(kh)
     nsteps = 10000
     kn     = np.linspace(kmin,kmax,nsteps)
     pknew  = interp.interp1d(kh,pk,kind='linear')
     pkn    = pknew(kn)
     step   = (kmax-kmin)/nsteps
     rmax   = 200.0  # Mpc/h
     rmin   = 0.01   # Mpc/h
     r      = np.linspace(rmin,rmax,nsteps)
     corr   = np.zeros(nsteps) 
     for i in range(nsteps):
	 corr[i] = step*(kn*kn*pkn*np.sin(r[i]*kn)/kn/r[i]).sum()
     return {'rcov':r,'corr':corr}

   def onehalo_corr(self):
     return 0

   def sumcorr(self):
     return 0
