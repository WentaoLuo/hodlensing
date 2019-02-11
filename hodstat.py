
import halotools
import numpy as np
from halotools.empirical_models.occupation_models import ZuMandelbaum15Cens,ZuMandelbaum15Sats
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from subgen import *
from scipy.special import erf
import mpmath
cosmology.setCosmology('WMAP9')
pi = np.pi
# HOD framework-----------------------------------------------------------------
class HODSTAT(object):
  def __init__(self,Zred=None,Msmin=None,Msmax=None):
    self.Zred = Zred
    self.Msmin= Msmin
    self.Msmax= Msmax
  def paramtable(self):
    """
    from Yang et al ApJ 2009 695:900-916 table p11
    logMh ranges from 12.16 to 14.58.
    """
    logMh  = np.array([12.16,12.45,12.75,13.05,13.35,13.64,13.94,14.23,14.58]) 
    phiall = np.array([0.31,0.61,1.10,1.96,3.61,6.15,9.72,16.49,24.91])
    phired = np.array([0.18,0.36,0.71,1.37,2.71,4.86,8.3,14.4,22.64])
    phiblu = np.array([0.15,0.26,0.43,0.68,2.71,4.86,8.3,14.4,22.64])
    phis   = np.array([phiall,phired,phiblu])
    alpall = np.array([-1.24,-1.19,-1.19,-1.15,-1.16,-1.22,-1.34,-1.4,-1.59])
    alpred = np.array([-1.17,-1.14,-1.16,-1.11,-1.10,-1.18,-1.30,-1.39,-1.54])
    alpblu = np.array([-1.25,-1.21,-1.13,-1.12,-1.23,-1.26,-1.41,-1.48,-1.71])
    alpha  = np.array([alpall,alpred,alpblu])
    lgMall = np.array([10.485,10.66,10.803,10.926,11.026,11.122,11.209,11.277,11.364]) 
    lgMred = np.array([10.539,10.713,10.844,10.958,11.047,11.131,11.216,11.278,11.365]) 
    lgMblu = np.array([10.414,10.601,10.746,10.872,10.978,11.082,11.152,11.277,11.364]) 
    lgMs   = np.array([lgMall,lgMred,lgMblu])
    sigall = np.array([0.145,0.140,0.142,0.168,0.179,0.180,0.182,0.164,0.159]) 
    sigred = np.array([0.100,0.112,0.127,0.161,1.047,11.131,11.216,11.278,11.365]) 
    sigblu = np.array([0.122,0.127,0.746,0.872,10.978,11.082,11.152,11.277,11.364]) 
    sigs   = np.array([sigall,sigred,sigblu])

    params = {'logMh':logMh,'phi':phis,'alpha':alpha,'lgMc':lgMs,'sigs':sigs}
    return params 
  def csmfunc(self,theta):
    """
    theta is the input for csm
    color=0,all; color=1,red; color=2,blue
    censat=0,central; censat=1,satellite
    """
    logMh,color = theta
    # get the parameter table and interpolate
    params = self.paramtable()
    Mhost  = params['logMh']
    phi    = params['phi']
    alpha  = params['alpha']
    lgMc   = params['lgMc']
    sigs   = params['sigs']
    Msmin  = self.Msmin
    Msmax  = self.Msmax
    Msx    = np.linspace(Msmin,Msmax,500)
    if color ==0:
      phi  = np.interp(logMh,Mhost,phi[0,:]) 
      alpha= np.interp(logMh,Mhost,alpha[0,:])
      lgMc = np.interp(logMh,Mhost,lgMc[0,:]) 
      sigs = np.interp(logMh,Mhost,sigs[0,:])
    if color ==1:
      phi  = np.interp(logMh,Mhost,phi[1,:])
      alpha= np.interp(logMh,Mhost,alpha[1,:]) 
      lgMc = np.interp(logMh,Mhost,lgMc[1,:]) 
      sigs = np.interp(logMh,Mhost,sigs[1,:]) 
    if color ==2:
      phi  = np.interp(logMh,Mhost,phi[2,:]) 
      alpha= np.interp(logMh,Mhost,alpha[2,:]) 
      lgMc = np.interp(logMh,Mhost,lgMc[2,:]) 
      sigs = np.interp(logMh,Mhost,sigs[2,:]) 

    csmfcen = (1.0/np.sqrt(2.0*pi)/sigs)*np.exp(-0.5*(Msx-lgMc)**2.0/sigs/sigs) 
    csmfsat = phi*((10.0**Msx/10.0**(lgMc-0.25))**(alpha+1.0))*np.exp(-((10.0**Msx/10.0**(lgMc-0.25)))**2.0)
    return {'Mrange':Msx,'Central':csmfcen,'Satellite':csmfsat}
#----------------------------------------------------------------------------------
  def analyticaloccnum(self,theta):
    """
    This is the analytical form of HOD number from either CSMFunction or CLFunction
    based on Cacciato et al 2009 MNRAS 394.
    """
    Msmin,Msmax,color = theta
    params = self.paramtable()
    Mhost  = params['logMh']
    phis   = params['phi']
    alphas = params['alpha']
    lgMcs  = params['lgMc']
    sigss  = params['sigs']

    nbins    = 300
    ncen     = np.zeros(nbins) 
    nsat     = np.zeros(nbins) 
    mbins    = np.linspace(12.17,14.5,nbins)
  
    for i in range(nbins):
      logMh  = mbins[i]
      if color ==0:
        phi  = np.interp(logMh,Mhost,phis[0,:]) 
        alpha= np.interp(logMh,Mhost,alphas[0,:])
        lgMc = np.interp(logMh,Mhost,lgMcs[0,:]) 
        sigs = np.interp(logMh,Mhost,sigss[0,:])
      if color ==1:
        phi  = np.interp(logMh,Mhost,phis[1,:])
        alpha= np.interp(logMh,Mhost,alphas[1,:]) 
        lgMc = np.interp(logMh,Mhost,lgMcs[1,:]) 
        sigs = np.interp(logMh,Mhost,sigss[1,:]) 
      if color ==2:
        phi  = np.interp(logMh,Mhost,phis[2,:]) 
        alpha= np.interp(logMh,Mhost,alphas[2,:]) 
        lgMc = np.interp(logMh,Mhost,lgMcs[2,:]) 
        sigs = np.interp(logMh,Mhost,sigss[2,:]) 

      ncen[i]= 0.5*(erf((Msmax-lgMc))-erf(Msmin-lgMc))
      tmp1   = 0.5*alpha+0.5
      tmp2   = (10.0**(Msmin-(lgMc-0.25)))**2.0
      tmp3   = (10.0**(Msmax-(lgMc-0.25)))**2.0
      gamma1 = mpmath.gammainc(tmp1,tmp2)
      gamma2 = mpmath.gammainc(tmp1,tmp3)
      nsat[i]=0.5*phi*(gamma1-gamma2)

    return {'Mhrange':mbins,'Occen':ncen,'Ocsat':nsat}
  def occupationnum(self,theta):
    Msmin,Msmax,color = theta
    nbins    = 300
    ncen     = np.zeros(nbins) 
    nsat     = np.zeros(nbins) 
    mbins    = np.linspace(12.16,14.58,nbins)

    for i in range(nbins):
      params   = [mbins[i],color]
      csmfuncs = self.csmfunc(params) 
      msrange  = csmfuncs['Mrange']
      csmcen   = csmfuncs['Central']
      csmsat   = csmfuncs['Satellite']
      step     = (np.max(msrange)-np.min(msrange))/len(msrange)
      ncen[i]  = step*csmcen.sum()
      nsat[i]  = step*csmsat.sum()/10.0**step

    res  = {'Mhrange':mbins,'Occen':ncen,'Ocsat':nsat}
    return res
  def hodstats(self,z):

    modelcen = ZuMandelbaum15Cens()
    modelsat = ZuMandelbaum15Sats()
    nbins    = 100
    ncen     = np.zeros(nbins) 
    nsat     = np.zeros(nbins) 
    mbins    = np.zeros(nbins)
    for i in range(nbins):
      mbins[i]    = 10**(11+i*0.05) 
      ncen[i]     = modelcen.mean_occupation(prim_haloprop=mbins[i])
      nsat[i]     = modelsat.mean_occupation(prim_haloprop=mbins[i])

    return {'mbins':mbins,'ncen':ncen,'nsat':nsat}
#-----------------------------------------------
  def hmfunc(self,z):

    nbins    = 100
    mbins    = np.zeros(nbins)
    mfunc_so = np.zeros(nbins)
    for i in range(nbins):
      mbins[i]    = 10**(8+i*0.1) 
      mfunc_so[i] =mass_function.massFunction(mbins[i],z,q_in='M',q_out='dndlnM',mdef='vir',model='tinker08')
  
    return {'mbins':mbins,'hmf':mfunc_so}

  def PofMass(self,theta,z):
    occnmm = self.occupationnum(theta)  
    mrange = occnmm['Mhrange']
    ncens  = occnmm['Occen']
    nsats  = occnmm['Ocsat']
    #ncens = hod['ncen']
    hmf    = self.hmfunc(z)
    mhbins = np.log10(hmf['mbins'])
    dndlnM = hmf['hmf']
    fnofm  = interp1d(mhbins,dndlnM,kind='linear')
    nofm   = fnofm(mrange) 

    Pofcen= (ncens)*nofm/((ncens)*nofm).sum()
    Pofsat= (nsats)*nofm/((nsats)*nofm).sum()
    return {'mbins':mrange,'Pcen':Pofcen,'Psat':Pofsat}
# END OF HODSTAT class####################################################
