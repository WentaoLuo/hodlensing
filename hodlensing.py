#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
#from subgen import *

pi = np.pi
#------------------------------------------------------------------------------

def main():
  #import hodstat as hod
  #hodstr  = hod.HODSTAT(0.1,8.5,10.5)
  #pofcen  = hodstr.PofMass([8.5,10.5,0.0],0.0)
  #mbins  = pofcen['mbins']
  #pcen   = pofcen['Pcen']
  #plt.plot(mbins,pcen)
  #plt.yscale('log')
  #plt.show()
  import gglens
  esdstr = gglens.ESD(13.0,7.0)
  Rp     = np.linspace(0.01,2,1000) 
  esd    = esdstr.NFWRoff(Rp,0.3)
  plt.plot(Rp,esd,'b-.',linewidth=3)
  plt.xlim(0.01,1.5)
  plt.ylim(-5.0,20.5)
  plt.xscale('log')
  plt.xlabel(r'Rp',fontsize=15)
  plt.show()
if __name__=='__main__':
  main()

