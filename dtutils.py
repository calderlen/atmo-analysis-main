import numpy as np
import matplotlib.pyplot as pl

def psarr(inarr,xaxis,yaxis,xname,yname,zname,filename=' ',contour=False,clevels=10,encapsulated=False,ctable='viridis',flat=False,dtlines=False,dtstruc={},lslines=False,ctrev=False,fsize=1,carr=np.zeros(1),points=False,xypointss=0,xyerr=0,xysyms='o',levels=0,alines=False,apoints=0,acolor='',stack=0,textstr=' ',textloc=0,textcolor='white',colorbar=True,fileformat='pdf'):
    
    """

        Generates a color-mapped 2D plot of a given data array (inarr) against x and y axes. Can include contour lines, a colorbar, 
        and text labels. It can also save the plot to a file.

           - Sets the figure size based on provided parameters.
           - Creates a color map of the data (inarr) against the provided x and y axes.
           - Depending on the options selected, it can add white dotted lines to the plot.
           - Adds a color bar (optional).
           - Can overlay contour lines (optional).
           - Can overlay text on the plot (optional).
           - Saves the plot to a file (optional).

        Inputs:
            inarr - 2D array of values to be plotted
            xaxis - 2D array of xaxis values
            yaxis - 2D array of yaxis values
            xname - x-axis label
            yname - y-axis label
            zname - z-axis (colorbar) label
            filename - name of output file
            contour - T/F; if true adds contour lines; default = False
            clevels - determines number of contour levels; default = 10
            encapsulated - T/F; unused; default = False
            ctable - color table; default = 'viridis'
            flat - T/F; if true go to stack
            dtlines - T/F; if true adds dotted lines
            dtstruc - dictionary of dtlines parameters
            lslines - T/F; unused; default = False
            ctrev - T/F; unused; default = False
            fsize - font size; unused; default = 1
            carr - unused; default = np.zeros(1)
            points - unused; default = False
            xypoints - unused; default = 0
            xyerr - unused; default = 0 
            xysyms - unused; default = 'o'
            levels - unused; default = 0
            alines - T/F;  ; default = False
            apoints - Adds a dotted line parallel to the y-axis to note a feature of one's choosing; default = 0
            acolor - color of the dotted line; default = ''
            stack - determines amount of plots stacked into one file; determines individual figure size; default = 0
            textstr - text overlayed on plot; default = ' '
            textloc - location of text; default = 0
            textcolor - color of text; default = 'white'
            colorbar - T/F; if true adds colorbar
            fileformat - format of output file; default = 'pdf'

        Outputs:
            

    """


    if flat:
        if stack == 0: pl.figure(figsize=(7,2.5))
        if stack == 1: pl.figure(figsize=(7,1.85))
        if stack == 2: pl.figure(figsize=(6.375,2.5))
        if stack == 3: pl.figure(figsize=(6.375,1.85))
    else:
        pl.figure(figsize=(7,7))

    pl.pcolor(xaxis,yaxis,inarr, cmap=ctable, edgecolors='none',rasterized=True)
    pl.axis([np.min(xaxis), np.max(xaxis), np.min(yaxis), np.max(yaxis)])
    ax=pl.gca()
    if stack == 0 or stack == 2: pl.xlabel(xname)
    if stack == 0 or stack == 1: pl.ylabel(yname)
    if stack == 1 or stack == 3: ax.axes.get_xaxis().set_visible(False)
    if stack == 2 or stack == 3: ax.axes.get_yaxis().set_visible(False)

    if dtlines:
        for i in range(0,4): pl.plot(0,dtstruc['crosses'][i],'+',color='white')
        pl.plot([dtstruc['vsini']*(-1.),dtstruc['vsini']*(-1.)],[np.min(yaxis),np.max(yaxis)],':',color='white')
        pl.plot([dtstruc['vsini'],dtstruc['vsini']],[np.min(yaxis),np.max(yaxis)],':',color='white')
        pl.plot([0.0,0.0],[np.min(yaxis),np.max(yaxis)],':',color='white')
        pl.plot([np.min(xaxis),np.max(xaxis)],[dtstruc['middle'],dtstruc['middle']],':',color='white')


    cbar=pl.colorbar(aspect=10,pad=0.01)
    cbar.ax.set_ylabel(zname)

    if contour:
        pl.contour(xaxis,yaxis,inarr,clevels,colors='white')

    if textstr != ' ':
        pl.text(textloc[0],textloc[1],textstr,color=textcolor,fontsize='x-large')

    if alines and len(apoints) > 1:
        if len(apoints) == 2:
            pl.plot([apoints[0], apoints[0]], [np.min(yaxis), np.max(yaxis)], ':', color=acolor)
            pl.plot([np.min(xaxis), np.max(xaxis)], [apoints[1], apoints[1]], ':', color=acolor)
    
    pl.tight_layout()
    if filename != 'none': 
        pl.savefig(filename, format=fileformat)
        pl.clf()


def mktslprplot(profarr,profarrerr,vabsfine,phase,phase2,vsini,bpar,RpRs,filename=' ', maxrange=[0,0], dophase=True, stack=0, zrange=[0,0], weighted=True, usetime=False, dur=0.):
    """
    Inputs:

        Processes profile data and then calls the psarr function to generate a plot. Plot represents changes in a stellar line profile 
        over the phase of a transit.\
        
        - Calculates a parameter called rmsprof that seems to be the standard deviation of the profiles outside a certain velocity range.
        - Re-bins the profile data in phase or time (using a new grid of phases or times).
        - Calculates some transit-related parameters like the time at different transit phases.
        - Finally, it calls the psarr function to plot the processed data

        profarr - 2D array of profiles
        profarrerr - 2D array of profile errors; unused
        vabsfine - 1D array of velocities
        phase - 1D array of phases
        phase2 - 1D array of phases
        vsini - projected stellar rotation velocity
        bpar - transit impact parameter????
        RpRs - planet to star radius ratio
        filename - name of output file
        maxrange - unused; default = [0,0]
        dophase - T/F; if true plots phase; default = True
        stack - determines amount of plots stacked into one file; determines individual figure size; default = 0
        zrange - unused; default = [0,0]
        weighted - T/F; if true, weights the profiles by their rms; default = True
        usetime - T/F; if true, plots time instead of phase; default = False
        dur - duration of transit in days; default = 0.
    """
    nexps=len(phase)
    nvabsfine=len(vabsfine)
    rmsprof=np.zeros(nexps)
    outsides=np.where(np.abs(vabsfine) > vsini*1.1)

    for i in range (0,nexps):
        rmsprof[i]=np.std(profarr[i,outsides[0]])

    #profarr*=(-1.)

    phaserange=[np.min(phase),np.max(phase2)]
    tspan=(phaserange[1]-phaserange[0])
    gaps=np.zeros(len(phase)-1)
    for i in range(0,len(gaps)): gaps[i]=phase[i+1]-phase2[i]
    res=np.abs(np.max(np.array([np.min(np.abs(gaps)),0.01])))
    npix=np.int(np.abs(np.ceil(tspan/res)))

    

    phaseprof=np.zeros((npix,nvabsfine))
    yphase=np.zeros(npix)
    time=phaserange[0]
    temparr1=np.zeros(nexps)
    temparr2=np.zeros(nexps)

    for i in range (0,npix):
        temparr1=(-1.)*phase+time
        temparr2=phase2-time
        place=np.where(temparr1*temparr2 >= 0.0)
        if len(place[0] > 1):
            for j in range (0,nvabsfine):
                if weighted: 
                    phaseprof[i,j]=np.sum(profarr[place[0],j]/rmsprof[place[0]]**2)/np.sum(1./rmsprof[place[0]]**2) 
                else: 
                    phaseprof[i,j]=np.mean(profarr[place[0],j])
        elif place[0] != -1:
            phaseprof[i,:]=profarr[place[0],:]

        yphase[i]=time
        time+=res


    t1=[0.,0.]
    t4=[1.,1.]
    vv=[0.,0.]

    x1=np.sqrt((1.+RpRs)**2-bpar**2)
    x2=np.sqrt((1.-RpRs)**2-bpar**2)

    x12=(x1-x2)/(2.*x1)
    t2=[x12,x12]
    t3=[1.-x12,1.-x12]

    if usetime:
        crosses = [(-1.)*dur/2.*24., (-1.+x12*2.)*dur/2.*24.,(1.-x12*2.)*dur/2.*24.,dur/2.*24.]
        middle = 0.
    else:
        crosses=[0.,x12,1.-x12,1.]
        middle = 0.5

    dtstruc={'crosses':crosses,'vsini':vsini,'middle':middle}

    if zrange[1] != 0.:
        bads=np.where(phaseprof > zrange[1])
        phaseprof[bads[0]]=zrange[1]
    if zrange[0] != 0.:
        bads=np.where(phaseprof < zrange[0])
        phaseprof[bads[0]]=zrange[0]

    if usetime:
        ylabel = 'hours from center of transit'
    else:
        ylabel = 'phase'

    
        


    psarr(phaseprof*(-100.),vabsfine,yphase,'velocity (km s$^{-1}$)',ylabel,'percent deviation',filename=filename,flat=True,dtlines=True,dtstruc=dtstruc,fsize=1.5,stack=stack,ctable='inferno')
