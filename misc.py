#!/usr/bin/env python3

import numpy as np

print('\t get_in_shower_plane')
#Real antennae positions to airshower plane positions
def get_in_shower_plane(pos, k, core, inclination, declination):
    pos= (pos - core[:,np.newaxis])
    B  = np.array([np.sin(inclination)*np.cos(declination),
                   np.sin(inclination)*np.sin(declination),
                   np.cos(inclination)])
    kxB = np.cross(k,B)
    kxB /= np.linalg.norm(kxB)
    kxkxB = np.cross(k,kxB)
    kxkxB /= np.linalg.norm(kxkxB)
    #print("k_", k_, "_kxB = ", _kxB, "_kxkxB = ", _kxkxB)

    return np.array([np.dot(kxB, pos), np.dot(kxkxB, pos), np.dot(k, pos)])

print('\t peak_amplitude_shower_plane')
def peak_amplitude_shower_plane(hitx, hity, hitz, zen, azi, bInc, bDec):
    # Collect information of an event required to make plots in event-display.
    # This part of code came from Valentin Decoene.
    # 'data' is kept here only for syntax reason and is not used. Remove this in future.
    k_shower   = np.array([np.sin(np.deg2rad(zen))*np.cos(np.deg2rad(azi)), 
                           np.sin(np.deg2rad(zen))*np.sin(np.deg2rad(azi)), 
                           np.cos(np.deg2rad(zen))])
    # Position of antannae in shower coordinate system. Note that hitx and hitX are different. hitX is time ordered hitx.
    xsp, ysp, zsp = get_in_shower_plane(np.array([hitx, hity, hitz]), 
                                        k_shower, 
                                        np.array([0,0,np.mean(hitz)]), # z-value is not on the ground.
                                        np.deg2rad(bInc), np.deg2rad(bDec))
    return xsp, ysp, zsp

print('\t SRTCleaning')
def SRTCleaning(hitx, hity, hitt, p2p):
    # S is for signal, R is for radius, and T is for time.
    # Collect indices of hits already tested for SRT and hits that forms a cluster around the testedHits.
    testedHits = []    # index of hits that have already been tested for SRT.
    inCluster  = []    # index of hits that are within SRT of the tested hit.

    def SRTCheck(indx, R=2100., T=7000.):
        # 5000ns required to travel 1500m for light.
        # ~7,000ns required to travel 2100m for light.
        # Use max p2p value as a seed to form a cluster.
        #indx    = np.where(p2p==S)
        S = p2p[indx]
        if indx not in testedHits:
            # No need to check again if it has already been checked.
            testedHits.append(indx)

            # Test for R and T. Make sure hits are close to the maximum p2p.
            r = np.sqrt((hitx-hitx[indx])**2 + (hity-hity[indx])**2)
            t = np.abs(hitt - hitt[indx])
            indxRT = np.where(np.logical_and(r<=R, r!=0, t<=T))[0]
            indxRT = np.where(np.logical_and(np.logical_and(r<R, r!=0), t<T))[0]

            # Test for S. Make sure signal on nearby hits has at least half of the tested signal strength.
            # indxS is wrt indxRT array and not wrt to initial arrays like hitx.
            # To include nearby antenna in cluster, signal strength on them must be greater than 10% 
            #    of test hit and must be more than 1% of max p2p.
            indxS = np.where(np.logical_and(np.logical_and(p2p[indxRT]>=0.1*S, p2p[indxRT]>=.01*p2p_max), 
                                            p2p[indxRT]>=800))[0]

            # Append indicies of hits near the hits with maximum signal strength.
            for item in indxS:
                if indxRT[item] not in inCluster:
                    inCluster.append(indxRT[item])
                    
        if indx in inCluster:
            # SRTCheck is run based on hits mentioned on inCluster.
            # Remove hits from inCluster if the check has already been performed.
            inCluster.remove(indx)
    
    # Provide hit with maximum p2p as the seed to form a cluster of hits. 
    argsort = np.argsort(p2p)  # returns indices for p2p in ascending order. 
    indx_max= argsort[-1]      # index of max p2p is at the end.
    p2p_max = p2p[indx_max]
    
    # Use max p2p value as a seed to form hits cluster.
    SRTCheck(indx_max)
        
    # If max p2p fails as a seed, use second maximum p2p as the seed to form a hits cluster.
    if len(inCluster)==0:
        p2p_2ndmax = p2p[argsort[-2]]
        if p2p_2ndmax>=0.5*p2p_max:
            SRTCheck(argsort[-2])
    
    # After the seed has been created around maximum p2p, loop over all hits that had satisfied SRT conditions.
    while len(inCluster)!=0:
        for index in inCluster:
            SRTCheck(index)    # Perform SRTCheck for hits located at position 'index'.
            
    return np.asarray(testedHits)    
    






