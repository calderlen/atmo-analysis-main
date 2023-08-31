#function horus,infoin=struc,mode=mode,convol=convol,koi13=koi13,gravd=gravd,ressize=res,resnum=resnum,image=image,onespec=onespec,diffrot=diffrot,macroturb=macroturb,emode=emode,amode=amode,path=path
#this function holds the routine for numerically calculating the time
#series line profile residuals or the photometric lightcurve for a
#transiting planet. It should be called by all other routines needing
#these models.
  
#The resolution may either be specified by mode--low (25^2 pix,
#default), medium (100^2 pix), or high (300^2 pix)--through the
#keyword RESSIZE, or directly through keyword RESNUM

import numpy as np
import math
import sys
import matplotlib.pyplot as pl
import uncertainties
from uncertainties import unumpy as unp

def model(struc, mode='spec', convol='y', gravd='n', res='low', resnum=0, image='n', onespec='n', diffrot='n', macroturb='n', emode='simple', amode='a', path='n', lineshifts='n', starspot=False):
    ckms=2.9979e5

    if resnum == 0:
        if res == 'low' : rstar=25
        if res == 'high' : rstar=300
        if res ==  'medium' : rstar=100
    else:
        rstar=round(resnum)

    rstar=int(rstar)

    gamma1=struc['gamma1']
    gamma2=struc['gamma2']
    vsini=struc['vsini']
    sysname=struc['sysname']
    obs=struc['obs']
    width=struc['width']

    if mode == 'spec':
        vabsfine=struc['vabsfine']
        lineabs1=vabsfine

    if gravd == 'y':
        beta=struc['beta'] #gravity darkening parameter, =0.25 for fully radiative star
        bigOmega=struc['Omega'] #in radians s-1
        gpole=10.0**struc['logg'] #assuming logg=g at pole, do something better later...
        psi=struc['psi']*np.pi/180.0 #stellar obliquity, input in degrees
        fb=struc['f']
        Reqcm=struc['rstar']*6.963e10 #Rsun->cm
    elif diffrot == 'y':
        inc=struc['psi']*np.pi/180.0
        psi=inc
        v0=np.absolute(vsini/np.sin(inc))
        v1=(-1.)*v0*struc['alpha']
        v2=v1
        fb=0.0
    else:
        fb=0.0
        psi=0.0

    if macroturb == 'y':
        zeta=struc['zeta'] #macroturbulent velocity dispersion

    feff=1.-np.sqrt((1.0-fb)**2*(np.cos(psi))**2+(np.sin(psi))**2)


    if onespec != 'y':
        Pd=struc['Pd']
        llambda=struc['lambda']*np.pi/180.0
        b=struc['b']
        if not starspot:
            rplanet=np.array([struc['rplanet']*rstar])
        else:
            rplanet=np.array(struc['rplanet'])
            rplanet*=rstar
        t=struc['t'] #make sure this is a numpy array!!!!
        times=struc['times']
        ecc=struc['e']
        periarg=struc['periarg']
        if amode == 'dur':
            dur=struc['dur']
            a=Pd/dur*np.sqrt((1.+struc['rplanet'])**2-b**2)/np.pi 
        else:
            a=struc['a']

    if starspot == True:
        #transparency=struc['transparency']
        t0=struc['t0']
        Prot=struc['Prot']*24.*60 #days -> minutes
        nspots=len(t0)
        nobj=nspots+1
    else:
        nobj=1

        #don't actually need dur anywhere later, but can calculate from
        #Dawson & Johnson '12: dur=P/pi*(1-e^2)^(3/2)/(1+esinomega)^2*asin(sqrt((1+RpRs)^2-b^2)/(a^2*(1-e^2)^2/(1+esinomega)^2-b^2))

    if lineshifts == 'y':
        lineshifts=struc['lineshifts']
    elif onespec == 'n':
        lineshifts=np.zeros(len(t))
    else:
        lineshifts=np.array([0.],dtype=float)

    if mode == 'spec' :

        lineabs=lineabs1
        npix=lineabs.size

    
#create the stellar array and line profiles

    if onespec != 'y':
        nexp=t.size
        rps=rplanet/rstar
        P=Pd*24.0*60.0 #orbital period in minutes
        #omega=np.zeros(nobj)
        #thetat, thetat2, thetamid = np.zeros((nexp,nobj)), np.zeros((nexp,nobj)), np.zeros((nexp,nobj))
        omega=2.0*np.pi/P*(1.0-ecc**2)**(-3.0/2.0)*(1.0+ecc*np.sin(periarg))**2 
        #this assumes angular velocity is constant over the course of the transit
        thetat=omega*t+np.pi/2.0-periarg
        thetat2=omega*np.add(t,times)+np.pi/2.0-periarg
        thetamid=omega*np.add(t,times/2.0)+np.pi/2.-periarg #middle of exposure, used later.
        z1, z2 = np.zeros((nexp,nobj)), np.zeros((nexp,nobj))
        z1[:,0]=a*(1.0-ecc**2)/(1.0+ecc*unp.cos(thetat))*(np.sin(periarg)*unp.sin(thetat)-np.cos(periarg)*unp.cos(thetat))
        z2[:,0]=a*(1.0-ecc**2)/(1.0+ecc*np.cos(thetat2))*(np.sin(periarg)*unp.sin(thetat2)-np.cos(periarg)*unp.cos(thetat2))
        
        while llambda > np.pi : llambda=llambda-2.0*np.pi
        while llambda < (-1.0)*np.pi : llambda=llambda+2.0*np.pi
        if starspot:
            for spot in range (1,nspots+1):
                #omega[spot]=2.0*np.pi/Prot
                #thetat[:,spot]=omega*t
                #thetat2[:,spot]=omega*(np.add(t,times))
                #thetamid[:,spot]=omega*(np.add(t,times/2.0))
                z1[:,spot]=np.sin((t0[spot-1]-t)*2.0*np.pi/Prot)*(-1.)
                z2[:,spot]=np.sin((t0[spot-1]-(t+times))*2.0*np.pi/Prot)*(-1.)

        #set up array of profiles in line across star, the limb darkening, and
        #the baseline profile

    limbarr=np.zeros((int(rstar)*2+1,int(rstar)*2+1))
    if image == 'each' or image == 'vels' : imarr=np.zeros((rstar*2+1,rstar*2+1,nexp))
    if path == 'y' : patharr=np.zeros((nexp,2,nobj))
    rarr=np.zeros((int(rstar)*2+1,int(rstar)*2+1))
    rprimearr=rarr
    if diffrot == 'y' :
        allvarr=rarr
        philat=rarr

    if mode == 'spec' :
        if diffrot != 'y' and macroturb != 'y': 
            linearr=np.zeros((int(rstar)*2+1,lineabs.size))
        else: 
            linearr=np.zeros((int(rstar)*2+1,int(rstar)*2+1,lineabs.size))
        if macroturb == 'y' : Bigtheta=np.zeros((int(rstar)*2+1,int(rstar)*2+1,lineabs.size))
        varr=lineabs
        vx=np.zeros(int(rstar)*2+1)
        baseline=np.zeros(npix)
        if onespec != 'y' :
            profarr=np.zeros((nexp,npix))
            basearr=np.zeros((nexp,npix))
        if image == 'vels' : velsarr=limbarr-99999.0

    if mode == 'phot' :
        flux=0.0
        timeflux=np.zeros(nexp)

    xstar=np.arange(2*rstar+1, dtype=float)-rstar
    xstar1=xstar
    ystar1=np.sqrt(rstar**2-xstar**2)
    ystar2=(-1.0)*ystar1

#now loop over the surface elements on the star
    count=0
    for x in range (-1*np.int64(rstar), 1*np.int64(rstar)+1) :
        vx=vsini*x/rstar

        if mode == 'spec' and diffrot != 'y' : 
            if isinstance(vx, uncertainties.core.AffineScalarFunc):
                linearr[count, : ]=unp.exp((-1.0)*((vx.nominal_value-varr)/width)**2)
            else:
                linearr[count, : ]= np.exp((-1.0)*((vx-varr)/width)**2)

        ycount=0
        for y in range (-1*np.int64(rstar), 1*np.int64(rstar)+1) :

            ybprime=y/(1.-feff)

            rarr[count,ycount]=np.sqrt(float(x)**2+float(y)**2)
            rprimearr[count,ycount]=np.sqrt(float(x)**2+ybprime**2)
            if rprimearr[count,ycount] <= rstar : 
                theta=np.arcsin(rarr[count,ycount]/rstar)
                mu=np.cos(theta)


                if gravd == 'y' :
                    db=4.0*y**2*(1.0-(1.0-fb**2))**2*(np.sin(psi))**2*(np.cos(psi))**2-4.0*(((np.cos(psi))**2*(1.0-fb)**2+(np.sin(psi))**2)*((y**2*(np.sin(psi))**2-rstar**2+x**2)*(1.0-fb**2)+y**2*(np.cos(psi))**2)) #rstar in this equation is reall Req

                    zb=(-2.0*y*(1.0-(1.0-fb)**2)*np.sin(psi)*np.cos(psi)+np.sqrt(np.absolute(db)))/(2.0*((1.0-fb)**2*(np.cos(psi))**2+(np.sin(psi))**2))
                    xb0=x
                    yb0=y*np.cos(psi)+zb*np.sin(psi)
                    zb0=(-1.0)*y*np.sin(psi)+zb*np.cos(psi)

                    Rb=np.sqrt(xb0**2+yb0**2+zb0**2)
                    Rvec=np.array([xb0/Rb,yb0/Rb,zb0/Rb])
                    Rperp=np.sqrt(xb0**2+zb0**2)
                    Rperpvec=np.array([xb0/Rperp,0.0,zb0/Rperp])

                    ruvec=Rvec/np.sqrt(np.sum(Rvec**2))
                    rperpuvec=Rperpvec/np.sqrt(np.sum(Rperpvec**2))

                    gvec=-gpole*(1.0-fb)**2*(xb0**2+yb0**2/(1.0-fb)**2+zb0**2)/(xb0**2+yb0**2+zb0**2)*ruvec+bigOmega**2*Reqcm*(np.sqrt(xb0**2+zb0**2)/np.sqrt(xb0**2+yb0**2/(1.0-fb)**2+zb0**2))*rperpuvec
                    glocal=np.sqrt(np.sum(gvec**2))


                    limbarr[count,ycount]=(1.0-gamma1*(1.0-mu)-gamma2*(1.0-mu)**2)*(glocal/gpole)**(4.0*beta)

                    if not np.isfinite(limbarr[count,ycount]) : return -np.inf
                #no gravity darkening
                else :
                    limbarr[count,ycount]=1.0-gamma1*(1.-mu)-gamma2*(1.-mu)**2
                    #take care of differential rotation
                    if diffrot == 'y' :
                        philat[count,ycount]=np.arcsin(np.sqrt(1.0-(float(x)/float(rstar))**2-(float(y)/float(rstar))**2)*np.cos(inc)+(float(y)/float(rstar))*np.sin(inc))
                        if not np.isfinite(philat[count,ycount]) : philat[count,ycount]=0.0
                        omegaxyr=(v0+v1*(np.sin(philat[count,ycount]))**2)
                        allvarr[count,ycount]=float(x)/float(rstar)*omegaxyr*np.absolute(np.sin(inc))
                        linearr[count,ycount, : ]=np.exp((-1.)*((allvarr[count,ycount]-varr)/width)**2)


                #calculate the macroturbulent profile at that point on the disk
                if macroturb == 'y' :
                    if np.cos(theta) != 0. and np.sin(theta) != 0.:
                        Bigtheta[count,ycount, : ]=np.exp((-1.0)*((varr)/(zeta*np.cos(theta)))**2)/(2.0*np.pi**(1.0/2.0)*zeta*np.cos(theta))+np.exp((-1.0)*((varr)/(zeta*np.sin(theta)))**2)/(2.0*np.pi**(1.0/2.0)*zeta*np.sin(theta))
                    elif np.cos(theta) == 0.:
                        Bigtheta[count,ycount, : ]=np.exp((-1.0)*((varr)/(zeta*np.sin(theta)))**2)/(2.0*np.pi**(1.0/2.0)*zeta*np.sin(theta))
                    elif np.sin(theta) == 0.:
                        Bigtheta[count,ycount, : ]=np.exp((-1.0)*((varr)/(zeta*np.cos(theta)))**2)/(2.0*np.pi**(1.0/2.0)*zeta*np.cos(theta))
                    #normalize so that large areas under the curve near the limb
                    #don't cause problems
                    Bigtheta[count,ycount, : ]=Bigtheta[count,ycount, : ]/np.sum(Bigtheta[count,ycount, : ])
                    temp=np.convolve(linearr[count,ycount,:],Bigtheta[count,ycount, : ],mode='same')
                    linearr[count,ycount, : ]=temp


                if image == 'vels' and diffrot != 'y' : velsarr[count,ycount]=vx.nominal_value
                if image == 'vels' and diffrot == 'y' : velsarr[count,ycount]=allvarr[count,ycount] #check this

#the numerical integration (summation) is done here.
                if mode == 'spec' and diffrot != 'y' and macroturb != 'y': baseline+=linearr[count, : ]*limbarr[count,ycount]
                if mode == 'spec' and (diffrot == 'y' or macroturb == 'y'): baseline+=linearr[count,ycount, : ]*limbarr[count,ycount]

                #if mode == 'spec' and macroturb == 'y' :
                    #if np.sin(theta) != 0.0 : baseline+=*limbarr[count,ycount]
                    #Bigtheta[count,ycount, : ]*limbarr[count,ycount]*np.cos(theta) #not clear why I was using theta...

                if mode == 'phot' : flux+=limbarr[count,ycount]


            ycount+=1
        
        count+=1




#OK, so now we need to go through the time series and find what areas
#are obscured, i.e., the meat of the whole problem.

    if image == 'all' : imarr=limbarr

    if onespec != 'y' :

        for count in range (0, nexp):
            fcount=0
            if mode == 'spec' : profarr[count, : ]=baseline
            if mode == 'phot' : timeflux[count]=flux
            if image == 'each' : imarr[ : , : ,count]=limbarr
            if image == 'vels' : imarr[ : , : ,count]=velsarr
            #find the location of the center of the planet at the start and the
            #end of the exposure

            #####I *think* I have the right sign on rstar-CORRECTED
            for obj in range (0,nobj):
                if obj == 0: #planet
                    center1=np.array([z1[count,0]*rstar*np.cos(llambda)+b*rstar*np.sin(llambda),b*rstar*np.cos(llambda)-z1[count,0]*rstar*np.sin(llambda)])
                    center2=np.array([z2[count,0]*rstar*np.cos(llambda)+b*rstar*np.sin(llambda),b*rstar*np.cos(llambda)-z2[count,0]*rstar*np.sin(llambda)])
                else: #spots
                    center1=np.array([z1[count,obj]*rstar,0.]) #assume spots move along equator for now
                    center2=np.array([z2[count,obj]*rstar,0.])
           

                if path == 'y' :
                    patharr[count,0,obj]=np.mean([center1[0],center2[0]])
                    patharr[count,1,obj]=np.mean([center1[1],center2[1]])

                xstar=np.arange(-100, 201, dtype=float)
                yp11=np.sqrt(rplanet[obj]**2-(xstar-center1[0])**2)
                yp12=yp11*(-1.0)

                yp11+=center1[1]
                yp12+=center1[1]

                yp21=np.sqrt((rplanet[obj])**2-(xstar-center2[0])**2)
                yp22=yp21*(-1.0)

                yp21+=center2[1]
                yp22+=center2[1]

                if np.sqrt(center1[0]**2+center1[1]**2) <= rstar+rplanet[obj] and np.sqrt(center2[0]**2+center2[1]**2) <= rstar+rplanet[obj] :

                    if np.absolute(llambda) != 90.0 :
                        mline=(center1[1]-center2[1])/(center1[0]-center2[0])
                        bline=center1[1]-mline*center1[0]

                        b1=center1[1]+center1[0]/mline
                        b2=center2[1]+center2[0]/mline

                    #now find points inside boxes around the planets
                    box1x=np.array([np.floor(center1[0]-rplanet[obj]),np.floor(center1[0]+rplanet[obj])+1],dtype=float)
                    box1y=np.array([np.floor(center1[1]-rplanet[obj]),np.floor(center1[1]+rplanet[obj])+1],dtype=float)

                    box2x=np.array([np.floor(center2[0]-rplanet[obj]),np.floor(center2[0]+rplanet[obj])+1],dtype=float)
                    box2y=np.array([np.floor(center2[1]-rplanet[obj]),np.floor(center2[1]+rplanet[obj])+1],dtype=float)

                    mbox=(box2y[0]-box1y[0])/(box2x[0]-box1x[0])

                    minx=np.amin([box1x[0],box2x[0]])
                    miny=np.amin([box1y[0],box2y[0]])

                    maxx=np.amax([box1x[1],box2x[1]])
                    maxy=np.amax([box1y[1],box2y[1]])

                    #cases
                    #+x+y
                    if box2x[0] > box1x[0] and box2y[0] > box1y[0] :
                        xu0=box2x[0]
                        yu0=box2y[1]

                        xl0=box2x[1]
                        yl0=box2y[0]

                        flat=0

                    #+x-y
                    if box2x[0] > box1x[0] and box2y[0] < box1y[0] :
                        xu0=box2x[1]
                        yu0=box2y[1]

                        xl0=box2x[0]
                        yl0=box2y[0]

                        flat=0

                    #-x-y
                    if box2x[0] < box1x[0] and box2y[0] < box1y[0] :
                        xu0=box2x[0]
                        yu0=box2y[1]

                        xl0=box2x[1]
                        yl0=box2y[0]

                        flat=0

                    #-x+y
                    if box2x[0] < box1x[0] and box2y[0] > box1y[0] :
                        xu0=box2x[1]
                        yu0=box2y[1]

                        xl0=box2x[0]
                        yl0=box2y[0]

                        flat=0

                    #x0
                    if box2x[0] == box1x[0] :
                        flat=1

                    #y0
                    if box2y[0] == box1y[0] :
                        flat=1
  

                    for x in range (np.int64(minx), np.int64(maxx)+1) :

                        for y in range (np.int64(miny), np.int64(maxy)+1) :

                            if np.absolute(x) <= rstar and np.absolute(y) <= rstar :

                                if flat == 0 :

                                    if y <= mline*(x-xu0)+yu0 and y >= mline*(x-xl0)+yl0 and rarr[x+rstar,y+rstar] <= rstar :
                                        condition=1
                                    else:
                                        condition=0

                                else:
                                    condition=1

                                if condition == 1 :

                                    rpli1=np.sqrt((float(x)-center1[0])**2+(float(y)-center1[1])**2)
                                    rpli2=np.sqrt((float(x)-center2[0])**2+(float(y)-center2[1])**2)

                                    if mline == 0 :
                                        xc=float(x)
                                        yc=center1[1]

                                        d1=center1[0]-float(x)
                                        d2=float(x)-center2[0]

                                    elif np.absolute(llambda) == 90.0 :
                                        d1=center1[1]-float(y)
                                        d2=float(y)-center2[1]

                                        xc=center1[0]
                                        yc=float(x)

                                    else:
                                        xc=(float(y)-center1[1]+mline*center1[0]+1./mline*float(x))/(mline+1.0/mline)
                                        yc=mline*xc+bline

                                        y1=(-1.0)/mline*float(x)+b1
                                        y2=(-1.0)/mline*float(x)+b2

                                        d1=y1-float(y)
                                        d2=float(y)-y2

                                    yprime=np.sqrt((float(x)-xc)**2+(float(y)-yc)**2)/rstar

                                    if yprime <= rps[obj] :

                                        if rpli1 <= rplanet[obj] and rpli2 <= rplanet[obj] :
                                            #covered by planet for the entire exposure
                                            factor=0.


                                        if bool(rpli1 <= rplanet[obj]) ^ (rpli2 <= rplanet[obj]) : #this syntax necessary b/c Python doesn't have logical xor operator
                                            if rpli1 <= rplanet[obj] : 
                                                ri=rpli1/rstar 
                                            else: 
                                                ri=rpli2/rstar

                                            if d1*d2 > 0.0 :
                                                #covered at the start or the end, by inner half of planet

                                                cotime=np.absolute((np.sqrt(rps[obj]**2-yprime**2)+np.sqrt(ri**2-yprime**2))/(a*omega*(1.0-ecc**2)/(1.0+ecc*np.cos(thetamid[count]))**2*(-ecc*np.sin(periarg)+2.0*ecc*np.sin(periarg)*(np.sin(thetamid[count]))**2+np.cos(periarg)*np.sin(thetamid[count])-np.sin(periarg)*np.cos(thetamid[count]))))
                                                #made abs b/c just care about the magnitude, not the sign, of the velocity
                                                if cotime > np.amax(times) : cotime=np.amax(times)

                                            else:
                                                #covered at start or end, by outer half of planet
                                                cotime=np.absolute((np.sqrt(rps[obj]**2-yprime**2)-np.sqrt(ri**2-yprime**2))/(a*omega*(1.0-ecc**2)/(1.0+ecc*np.cos(thetamid[count]))**2*(-ecc*np.sin(periarg)+2.0*ecc*np.sin(periarg)*(np.sin(thetamid[count]))**2+np.cos(periarg)*np.sin(thetamid[count])-np.sin(periarg)*np.cos(thetamid[count]))))
                                                if cotime > np.amax(times) : cotime=np.amax(times)

                                            factor=1.0-cotime/times[count]



                                        if rpli1 > rplanet[obj] and rpli2 > rplanet[obj]:
                                            if d1*d2 > 0.0 :
                                                #covered for part of the exposure but not at the beginning or end
                                                cotime=np.absolute(2.0*np.sqrt(rps[obj]**2-yprime**2)/(a*omega*(1.0-ecc**2)/(1.0+ecc*np.cos(thetamid[count]))**2*(-ecc*np.sin(periarg)-2.0*ecc*np.sin(periarg)*(np.sin(thetamid[count]))**2-np.cos(periarg)*np.sin(thetamid[count])-np.sin(periarg)*np.cos(thetamid[count]))))#modify to make this be the t instintaneously AT this time!!!
                                                if cotime > np.amax(times) : cotime=np.amax(times)
                                                factor=1.0-cotime/times[count]

                                            else:
                                                #along the planet's path but not currently covered
                                                factor=1.




                                                #linearr[count,*]+=exp((-1.)*((vx-lineabs)/5.)**2)*stararr

                                    else:
                                        #this is the case for outside of the planetary disk
                                        factor=1.


                                    if factor != 1.0 :
                                        #if starspot: factor*=transparency
                                        if mode == 'spec' : 
                                            if isinstance(linearr[x+rstar, :], uncertainties.core.AffineScalarFunc):
                                                line_value = linearr[x+rstar, :].nominal_value
                                            else:
                                                line_value = linearr[x+rstar, :]
                                            
                                            if isinstance(limbarr[x+rstar,y+rstar], uncertainties.core.AffineScalarFunc):
                                                limb_value = limbarr[x+rstar,y+rstar].nominal_value
                                            else:
                                                limb_value = limbarr[x+rstar,y+rstar]
                                                
                                            if isinstance((1.0-factor), uncertainties.core.AffineScalarFunc):
                                                factor_value = (1.0-factor).nominal_value
                                            else:
                                                factor_value = (1.0-factor)

                                            if isinstance(profarr[count, :], uncertainties.core.AffineScalarFunc):
                                                prof_value = profarr[count, :].nominal_value
                                            else:
                                                prof_value = profarr[count, :]

                                            profarr[count, : ] = prof_value - line_value * limb_value * factor_value
                                        
                                        if mode == 'phot' : timeflux[count]-=limbarr[x+rstar,y+rstar]*(1.0-factor)
                                        if image == 'each':
                                            if isinstance(factor, uncertainties.core.AffineScalarFunc):
                                                factor_value = factor.nominal_value
                                            else:
                                                factor_value = factor
                                            imarr[x+rstar,y+rstar,count] *= factor_value

                                        if image == 'vels' : imarr[x+rstar,y+rstar,count]=-9999.0
                                        if image == 'all' : imarr[x+rstar,y+rstar,count]*=factor

                                        #THIS ISN'T RIGHT, FIX IT
                                        #???
                fcount+=1
            if mode == 'spec' :
                avgprof=baseline
                profarr[count, : ]/=np.amax(baseline)
                basearr[count, : ]=baseline/np.amax(baseline) 

                if convol == 'y' :
                    if obs == 'keck' : Resolve=50000.0
                    if obs == 'hjst' : Resolve=60000.0
                    if obs == 'het' : Resolve=30000.0
                    if obs == 'keck-lsi' : Resolve=20000.0
                    if obs == 'subaru' : Resolve=80000.0
                    if obs == 'aat' : Resolve=70000.0
                    if obs == 'not' : Resolve=47000.0
                    if obs == 'tres' : Resolve=44000.0
                    if obs == 'harpsn'  or obs == 'harps': Resolve=120000.0
                    if obs == 'lbt' or obs== 'pepsi' : Resolve=120000.0
                    if obs == 'geminin1' : Resolve=67500.0
                    if obs == 'geminin2' : Resolve=40000.0
                    if obs == 'igrins': Resolve=40000.0
                    if obs == 'nres': Resolve=48000.0
    
                    sigma=ckms/Resolve
                    
                    vi=vabsfine[np.where(np.absolute(vabsfine) < 25.0)]
                    if vi.size == 1.0 :
                        vi=np.arange(-10, 21, dtype=float)

                    if obs == 'het' or obs == 'hjst' : 
                        instp=makeinstp(obs,vi-lineshifts[count]) 
                    else: 
                        instp=np.exp(-1.0/2.0*((vi-lineshifts[count])/sigma)**2)

                    profarrtemp=np.zeros((vabsfine.size))
                    profarrtemp=profarr[count, : ]
                    basearrtemp=np.zeros((vabsfine.size))
                    basearrtemp=basearr[count, : ]
                    if len(instp) == 0: import pdb; pdb.set_trace()
                    profarr[count, : ]=np.convolve(profarrtemp,instp,mode='same')
                    basearr[count, : ]=np.convolve(basearrtemp,instp,mode='same')

                profarr[count, : ]/=np.amax(basearr[count, : ])
                basearr[count, : ]/=np.amax(basearr[count, : ])

            if mode == 'phot' : timeflux[count]/=flux


    if convol == 'y' and onespec == 'y' :
        if obs == 'keck' : Resolve=50000.0
        if obs == 'hjst' : Resolve=60000.0
        if obs == 'het' : Resolve=30000.0
        if obs == 'keck-lsi' : Resolve=20000.0
        if obs == 'subaru' : Resolve=80000.0
        if obs == 'not' : Resolve=47000.0
        if obs == 'aat' : Resolve=70000.0
        if obs == 'tres' : Resolve=44000.0
        if obs == 'harpsn' : Resolve=120000.0
        if obs == 'lbt' : Resolve=120000.0
        if obs == 'geminin' : Resolve=40000.0
        if obs == 'igrins': Resolve=40000.0
        if obs == 'nres': Resolve=48000.0

        sigma=ckms/Resolve

        vi=vabsfine[np.where(np.absolute(vabsfine) < 25.0)]
        if vi.size == 1.0 : vi=np.arange(-10, 21, dtype=float)

        if obs == 'het' or obs == 'hjst' : 
            instp=makeinstp(obs,vi) 
        else: 
            instp=np.exp(-1.0/2.0*(vi/sigma)**2)

        baselinetemp=np.zeros(npix)
        baselinetemp=baseline

        baseline=np.convolve(baselinetemp,instp,mode='same') 

    if image == 'comb' : imarr=limbarr

    if mode == 'spec' :
        baseline/=np.amax(baseline)
        if onespec != 'y' :
            if image == 'n' :
                if path == 'n' : 
                    outstruc={'profarr': profarr, 'basearr': basearr, 'baseline':baseline, 'z1': z1, 'z2': z2} 
                else:
                    outstruc={'profarr': profarr, 'basearr': basearr, 'baseline': baseline, 'z1': z1, 'z2': z2, 'patharr': patharr, 'staraxis':xstar1}

            else: 
                if path == 'n' : 
                    outstruc={'profarr': profarr, 'basearr': basearr, 'baseline': baseline, 'imarr': imarr, 'z1': z1, 'z2': z2} 
                else: 
                    outstruc={'profarr': profarr, 'basearr': basearr, 'baseline': baseline, 'imarr': imarr,'z1': z1, 'z2': z2, 'patharr': patharr, 'staraxis': xstar1}

        else:
            outstruc={'baseline': baseline}

    if mode == 'phot' :
        if onespec != 'y':
            outstruc={'timeflux': timeflux, 'z1': z1, 'z2' :z2}
        else:
            outstruc={'imarr': imarr}

   # breakpoint()
    
    return outstruc


#this function makes an instrumental profile
def makeinstp(obs, vin):
    
#create oversample abscissa
    npix1=vin.size
    npix2=npix1*5
    vfine=np.zeros(npix2)
    deltav=vin[1]-vin[0]
    pixnumbers=vfine
#pixnumbers=(dindgen(npix2)/double(npix2)-0.5)*double(npix1)

    for i in range (0,npix2):
        vfine[i]=vin[i/5]+deltav*float(np.mod(i,5))/5.0
        pixnumbers[i]=float(np.round(vin[i/5]/deltav))+float(np.mod(i, 5))/5.0

#make the gaussian and the tophat
    if obs == 'hjst' : sigma=0.9
    if obs == 'het' : sigma=1.1#/2. #due to binning


    gaussian=np.exp(-1.0/2.0*(pixnumbers/sigma)**2)

    if obs == 'hjst' : hats=2.0/2.0 #width->halfwidth
    if obs == 'het' : hats=4.3/(2.0)#*2.) #width->halfwidth,binning

    tophat=np.zeros(npix2)
    goods=np.where(np.absolute(pixnumbers) <= hats)
    tophat[goods]=1.0/(hats*2.0)

    gaussian1=gaussian
    tophat1=tophat
    psfout1=np.convolve(tophat1,gaussian1,mode='same')

#and let's try it analytically
    dx=pixnumbers #center at zero
    profile=np.absolute(1.0/(np.sqrt(2.0*np.pi*sigma)))*np.exp(-dx*dx*0.5/(sigma*sigma))
    denom=sigma*np.sqrt(2.0)
    erfc1=1.0-erf((dx-hats)/denom)
    erfc2=1.0-erf((dx+hats)/denom)
    profile=profile+erfc1-erfc2
    profile/=2.0*(2.0*hats)
    profile/=np.amax(profile)


#OK, now we need to rebin

    psfout=np.zeros(npix1)

    for i in range (1,npix1-1): psfout[i]=np.mean(profile[5*i-2:5*i+2+1]) 
    #in python, seems to return the range [x:y-1] when enter [x:y]

    psfout[0]=np.mean(profile[0:2+1])
    psfout[npix1-1]=np.mean(profile[npix2-2:npix2-1+1])

    psfout/=np.max(psfout)*sigma*np.sqrt(2.0*np.pi)

    return psfout


def fourierfilt(profarr, mask):
    
    fft1=np.fft.fft2(profarr)
    fft2=mask*fft1
    fft3=np.fft.ifft2(fft2)
    return np.real(fft3)


def erf(x): #b/c can't get the native Python erf to work.
#code from http://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
#modified to work on arrays

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = np.ones(x.size)
    for i in range (0, x.size):
        if x[i] < 0:
            sign[i] = -1
    x = abs(x)

    # A & S 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)

    return sign*y

def mkskyline(vabsfine,sdepth,svel,obs):
    if obs == 'keck' : Resolve=50000.0
    if obs == 'hjst' : Resolve=60000.0
    if obs == 'het' : Resolve=30000.0
    if obs == 'keck-lsi' : Resolve=20000.0
    if obs == 'subaru' : Resolve=80000.0
    if obs == 'aat' : Resolve=70000.0
    if obs == 'not' : Resolve=47000.0
    if obs == 'tres' : Resolve=44000.0
    if obs == 'harpsn' : Resolve=120000.0
    if obs == 'lbt' : Resolve=120000.0
    if obs == 'igrins': Resolve=40000.0
    if obs == 'nres': Resolve=48000.0

    ckms=2.9979e5

    sigma=ckms/Resolve

    if obs == 'het' or obs == 'hjst' : 
        instp=makeinstp(obs,vabsfine-svel) 
    else: 
        instp=np.exp(-1.0/2.0*((vabsfine-svel)/sigma)**2)

    instp/=np.max(instp)
    instp*=sdepth

    return instp

