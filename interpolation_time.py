#!/usr/bin/env python3

import numpy as np

def GetEffectiveactionIndex(x0,y0,z0,xant=0,yant=0,zant=0,ns=325,kr=-0.1218,stepsize = 20000):

        rearth=6371007.0
        R02=x0*x0+y0*y0  #notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
        h0=(np.sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3 #altitude of emission, in km

        if(h0>100):
         return 1

        rh0 = ns*np.exp(kr*h0) #refractivity at emission
        n_h0=1.E0+1.E-6*rh0 #n at emission
        #print("n_h0",n_h0,ns,kr,x0,y0,z0,h0,rh0)

        hd=(zant)/1.E3 #detector altitude

#       Vector from detector to average point on track. Making the integral in this way better guaranties the continuity
#       since the choping of the path will be always the same as you go farther away. If you start at your starting point, for a given geometry,
#       the choping points change with each starting position.

        ux = x0-xant
        uy = y0-yant         #the antenna position, considered to be at the core
        uz = z0-zant

        Rd=np.sqrt(ux*ux + uy*uy)
        kx=ux/Rd
        ky=uy/Rd #!k is a vector from the antenna to the track, that when multiplied by Rd will end in the track and sumed to antenna position will be equal to the track positon
        kz=uz/Rd

#       integral starts at antenna
        nint=0
        sum=0.E0

        currpx=0+xant
        currpy=0+yant    #!current point (1st antenna position)
        currpz=zant
        currh=hd

        while(Rd > stepsize): #if distance projected on the xy plane is more than 10km
          nint=nint+1
          nextpx=currpx+kx*stepsize
          nextpy=currpy+ky*stepsize           #this is the "next" point
          nextpz=currpz+kz*stepsize

          nextR2=nextpx*nextpx + nextpy*nextpy #!se usa el cuadrado, se puede ahorrar la raiz cuadrada
          nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1.E3

          if(np.absolute(nexth-currh) > 1.E-10  ):   #check that we are not going at constant height, if so, the refraction index is constant
              sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
          else:
              sum=sum+np.exp(kr*currh)

          currpx=nextpx
          currpy=nextpy
          currpz=nextpz  #Set new "current" point
          currh=nexth

          Rd=Rd-stepsize #reduce the remaining lenght
        #enddo

        #when we arrive here, we know that we are left with the last part of the integral, the one closer to the track (and maybe the only one)

        nexth=h0

        if(np.absolute(nexth-currh) > 1.E-10 ): #check that we are not going at constant height, if so, the refraction index is constant
          sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
        else:
          sum=sum+np.exp(kr*currh)

        nint=nint+1
        avn=ns*sum/nint
        n_eff=1.E0+1.E-6*avn #average (effective) n
        return n_eff



def GetZHSEffectiveactionIndex(x0,y0,z0,xant=0,yant=0,zant=0,ns=325,kr=-0.1218,stepsize = 20000):
      #rearth=6371007.0 #new aires
      rearth=6370949.0 #19.4.0
#     Variable n integral calculation ///////////////////
      R02=x0*x0+y0*y0  #!notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada (entro con injz-zXmax -> injz-z0=zXmax
      h0=(np.sqrt( (z0+rearth)*(z0+rearth) + R02 ) - rearth)/1E3    #!altitude of emission

      rh0=ns*np.exp(kr*h0) #!refractivity at emission (this
      n_h0=1+1E-6*rh0 #!n at emission
#        write(*,*) "n_h0",n_h0,ns,kr,x0,y0,injz-z0,h0,rh0
      modr=np.sqrt(R02)

      if(modr > 1000): #! if inclined shower and point more than 20km from core. Using the core as reference distance is dangerous, its invalid in upgoing showers

#         Vector from average point of track to observer.
          ux = xant-x0
          uy = yant-y0
          uz = zant-z0

#         divided in nint pieces shorter than 10km
          nint=int((modr/stepsize)+1)
          kx=ux/nint
          ky=uy/nint       #k is vector from one point to the next
          kz=uz/nint
#
          currpx=x0
          currpy=y0        #current point (1st is emission point)
          currpz=z0
          currh=h0
#
          sum=0
          for iii in range(0,nint):
            nextpx=currpx+kx
            nextpy=currpy+ky #!this is the "next" point
            nextpz=currpz+kz
            nextR2=nextpx*nextpx + nextpy*nextpy
            nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1E3
#c
            if(np.abs(nexth-currh) > 1E-10  ):
              sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
              sum=sum+np.exp(kr*currh)
#            endif
#c
            currpx=nextpx
            currpy=nextpy
            currpz=nextpz  #!Set new "current" point
            currh=nexth
#c
          avn=ns*sum/nint
#          print*,"avn:",avn
          n_eff=1+1E-6*avn #!average (effective) n
#c
      else:
#c         withouth integral
          hd=zant/1E3 #!detector altitude
#
          if(np.abs(hd-h0) > 1E-10):
            avn=(ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
#            print*,"avn2:",avn
          else:
            avn=ns*np.exp(kr*h0)
#           print*,"avn3:",avn
#            print *,"Effective n: h0=hd"
#          endif
          n_eff=1+1E-6*avn #!average (effective) n
#        endif
#c     ///////////////////////////////////////////////////
      return n_eff
#      end


print('\t ComputeTimeAntennas')
def ComputeTimeAntennas(Nant, xmaxdistance, xmaxposition, antennapos):
    """
    Created on Wed Mar 30 10:37:14 2022
    @author: chiche
    """
    
    # Nant: number of antennas
    # xmaxdistance: Distance between Xmax and the shower core in meters
    # xmaxposition: numpy array  with Xmax coordinates [xXmax, yXmax, zXmax] 
    # in meters
    # antennapos: numpy array [xpos, ypos, zpos] that gives the target 
    # antennas coordinates in meters
    

    c = 299792458.0
    time2core = (xmaxdistance/c)*1e9 # time from xmax to the core for n = 1 
    #if(realXmax): time2core = ((XmaxDistance/(299792458.0))*1e9)
    positions = antennapos
    xpoints, ypoints , zpoints  = positions[:,0], positions[:,1], positions[:,2]
    t0all= np.zeros(len(xpoints))
    k =0
    
    for i in range(len(xpoints)):
        
        nant = i
        nref = GetZHSEffectiveactionIndex(xmaxposition[0],xmaxposition[1],xmaxposition[2],\
        xant=xpoints[nant],yant=ypoints[nant],zant=zpoints[nant],ns=325,kr=-0.1218,stepsize = 20000)
      
        dant = np.sqrt((xmaxposition[0]-xpoints[nant])**2+ (xmaxposition[1]-ypoints[nant])**2 +\
        (xmaxposition[2]-zpoints[nant])**2)
            
        t0 = (dant*nref/c)*1e9 - time2core 
        t0all[k] = t0
        k = k +1
    
    desiredtime = t0all 
     
    return desiredtime

