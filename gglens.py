import numpy as np

omega_m = 0.28
rho_crit= 9.47e-27
ckm     = 3.24078e-20
ckg     = 5.0e-31
rhoc    = rho_crit*ckg*10e+9/ckm/ckm/ckm
rhom    = rhoc*omega_m

pi = np.pi
# NFW to ESD analytical model---------------------------------------------------
class ESD(object):
  def __init__(self,logMh=None,con=None):
    self.logMh = logMh
    self.con   = con

    self.r200 = (10.0**self.logMh*3.0/200./rhom/pi)**(1./3.)
    self.rs   = self.r200/self.con
    self.delta= (200./3.0)*(self.con**3)\
                /(np.log(1.0+self.con)-self.con/(1.0+self.con))
    self.amp  = 2.0*self.rs*self.delta*rhoc*1e-13

  def funcs(self,xx):
    x   = xx/self.rs
    x1  = x*x-1.0
    x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
    x3  = np.sqrt(np.abs(1.0-x*x))
    x4  = np.log((1.0+x3)/(x))
    s1  = xx*0.0
    s2  = xx*0.0

    ixa = x>0.
    ixb = x<1.0
    ix1 = ixa&ixb
    s1[ix1] = 1.0/x1[ix1]*(1.0-x2[ix1]*x4[ix1])
    s2[ix1] = 2.0/(x1[ix1]+1.0)*(np.log(0.5*x[ix1])\
               +x2[ix1]*x4[ix1])

    ix2 = x==1.0
    s1[ix2] = 1.0/3.0
    s2[ix2] = 2.0+2.0*np.log(0.5)

    ix3 = x>1.0
    s1[ix3] = 1.0/x1[ix3]*(1.0-x2[ix3]*np.arctan(x3[ix3]))
    s2[ix3] = 2.0/(x1[ix3]+1.0)*(np.log(0.5*x[ix3])+\
               x2[ix3]*np.arctan(x3[ix3]))

    res = {'gfunc':s2,'ffunc':s1}
    return res

  def NFWcen(self,Rp):
    functions = self.funcs(Rp)
    funcf     = functions['gfunc']-functions['ffunc']
    res       = self.amp*(funcf)
    return res

  def NFWRoff(self,Rp):
    nstp = 1000
    Rf   = np.linspace(0.01,0.9,nstp)
    the  = np.linspace(0.0,2.0*pi,nstp)
    sthe = 2.0*pi/nstp
    nxx  = len(Rp)
    tmp  = np.zeros((nstp,nxx))
    tmpsm= np.zeros((nstp,nxx))
    dsig = np.zeros((nstp,nxx))
    for i in range(nstp):
      tmp1 = np.zeros((nstp,nxx)) 
      for j in range(nstp):
        cosa   = np.cos(the[j])
        xx     = np.sqrt(Rp*Rp+Rf[i]*Rf[i]+2.0*Rp*Rf[i]*cosa)
	funcf  = self.funcs(xx)
        tmp1[j,:]= sthe*self.amp*(funcf['gfunc']-funcf['ffunc'])
      
      tmp[i,:] = 0.5*(tmp1[:,:]).sum(axis=0)/pi
      tmp2     = np.zeros((nstp,nxx))
      for iy in range(nxx):
        Rsub     = np.linspace(0.0,Rp[iy],nstp)
        srp      = Rp[iy]/nstp
        interpesd= np.interp(Rsub,Rp,tmp[i,:])
        for ix in range(nstp):
	   tmp2[ix,iy] = srp*interpesd[ix]*Rsub[ix]
        tmpsm[i,iy] = 2.0*(tmp2[:,iy].sum(axis=0))/Rp[iy]/Rp[iy]
      dsig[i,:]  = tmpsm[i,:]-tmp[i,:]

    return {'Roff':Rf,'subesd':dsig}

  def nfw2ddens(self):
    nn  = 100
    Rpj = np.linspace(0.01,0.9,nn)
    func= self.funcs(Rpj)
    pdf = func['ffunc']/func['ffunc'].sum()
    return {'Rpj':Rpj,'pdf':pdf}
#-----CLASS FINISHED--------------------------------------------------
