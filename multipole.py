#!/usr/bin/env python
"""
### Written by Matteo Guzzo ###
### A.D. MMXIV (2014)       ###
Multipole fit module. It is callable as a stand-alone script. 
"""
from __future__ import print_function
import numpy as np


def getdata_file(infilename,wantedcol=1):
    """
    This function opens a file with a given filename
    and puts (as default) the first two 
    columns into two numpy arrays
    that are returned. 
    With an optional keyword (wantedcol) one 
    can choose to wrap the n-th column of the file.
    """
    import numpy as np
    infile = open(infilename)
    preen = []
    predata = []
    ncol = wantedcol 
    for lines in infile : 
        #print(lines[0])
        if lines[0] != "#" :
            line = map(float,lines.split())
            #print(line)
            preen.append(line[0])
            predata.append(line[ncol])
    infile.close()
    preen = np.array(preen)
    predata = np.array(predata)
    return preen, predata

def first_inverse_moment(preen,predata):
    """
    Calculates and returns the first inverse moment of
    a dataset.
    """
    import numpy as np
    nbin = np.size(preen)
    fxonx = np.zeros(nbin)
    tmpen = preen
    tmpdata = predata
    dx = abs((tmpen[-1] - tmpen[0])/nbin) # or nbin + 1 ???
    for i in xrange(nbin) : 
        if tmpen[i] == 0. : 
            tmpen[i] += dx/1000
            #izero = i
            #tmpdata[i] = 0
            #print(" Avoiding zero:", preen[i], predata[i] )
            break
    #tmpen[tmpen == 0] += dx/1000
    fxonx = tmpdata / tmpen
    #if izero is not None: preen[izero] = 0.
    return fxonx

def resize_en(preen, nbin) :
    """
    This is a quite primitive solution. 
    I am not sure it will work in any situation.
    TODO: think of a smarter grid analyser.
    """
    import numpy as np
    nbin = int(nbin)
    if np.size(preen) < float( 2 * nbin ) :
        #print(" X-axis grid is too coarse for so many poles.")
        print(" Refining grid...")
        nx = 2*nbin+1
        print(" Old dx = %g, new dx = %g." % (abs(preen[-1]-preen[0])/(np.size(preen)-1),abs(preen[-1]-preen[0])/nx))
        en = np.linspace(preen[0], preen[-1], nx)
    else :
        en = preen
    return en

def fit_multipole(preen,predata,nbin, method='const2', ifilewrite=0):
    """
    Wrapper function to speed-up selection of fit method.
    _const is with uniformly-spaced binning method (newer).
    _fast is with equal-area binning method (legacy).
    """
    #method = 'fast'
    if method == 'const':
        omegai, gi, deltai = fit_multipole_const(preen,predata,nbin, ifilewrite)
       #print(omegai, gi, deltai)
    elif method == 'const2':
        omegai, gi, deltai = fit_multipole_const2(preen,predata,nbin, ifilewrite)
       #print(omegai, gi, deltai)
    elif method == 'fast':
        omegai, gi, deltai = fit_multipole_fast(preen,predata,nbin)
    else:
        omegai, gi, deltai = fit_multipole_const2(preen,predata,nbin)
    return omegai, gi, deltai

def fit_multipole_const2(preen,predata,nbin, ifilewrite=0):
    """
    VERSION WITH EQUIDISTANT Delta_i!
    Hopefully improved version.
    This function fits a curve given by some dataset (preen,predata) 
    with a given number of poles (nbin).
    It returns omegai, lambai, deltai.
    """
    import numpy as np
    import sys
    from scipy.interpolate import interp1d
    import matplotlib.pylab as plt
    print("fit_multipole_const2 :: ")
    nbin = int(nbin)
    dx = (preen[-1]-preen[0])/nbin
    x = np.linspace(preen[0],preen[-1],nbin+1)
    interp_data = interp1d(preen, predata, kind = 'linear', axis = -1)
    y = interp_data(x)
    totalint = np.trapz(predata,preen)
    totdeltax = abs( preen[-1] - preen[0] )
    print(" Totdeltax, np.size(preen), dx:", totdeltax, np.size(preen), ( preen[-1] - preen[0] ) / float( np.size(preen) - 1 ))
    print(" Number of poles (nbin):", nbin)
    print(" Total integral:", totalint )
    print(" Total integral / nbin:", totalint / float(nbin))
    lambdai = []
    omegai = []
    deltai = [dx for i in range(x.size-1)]
    for i in np.arange(x.size-1):
        oi = (x[i+1]+x[i])/2
        omegai.append(oi)
        tmpint = (y[i+1]+y[i])*dx/2
        tmpgi = tmpint 
        lambdai.append(tmpgi)
    if nbin == 1:
        print(" WARNING: with 1 bin, the value of lambdai is simply the integral under the curve.")
        print(" WARNING: with 1 bin, the value of omegai is simply the max value of the curve.")
        lambdai = totalint
        argmax = np.argmax(predata)
        omegai = preen[argmax]
        max_val = np.amax(predata)
        #Check for multiple max values
        many_argmax = [omegai]
        j = 0
       #print(argmax,j,predata.size)
       #plt.plot(preen,predata);plt.show();sys.exit()
        while np.amax(predata[argmax+j+1:]) == max_val:
            j += 1
            max_val = np.amax(predata[argmax+1+j:])
            argmax = np.argmax(predata[argmax+1+j:])
            many_argmax.append(preen[argmax+j+1])
            if argmax+j+2 == predata.size:
                break
        many_argmax = np.array(many_argmax)
        omegai = many_argmax.mean()
    lambdai = np.array(lambdai)
    omegai  = np.array(omegai)
    deltai  = np.array(deltai)
    print(" Size(omegai, lambdai, deltai): ",omegai.shape,lambdai.shape,deltai.shape)
    sum_li = np.sum(lambdai)
    print(" Sum of lambdai:", sum_li)
    print(" Error on total integral:", (totalint-sum_li)/totalint)
    return omegai, lambdai, deltai
    

def fit_multipole_const(preen,predata,nbin, ifilewrite=0):
    """
    VERSION WITH EQUIDISTANT Delta_i!
    This function fits a curve given by some dataset (preen,predata) 
    with a given number of poles (nbin).
    preen is supposed to be all positive and with increasing
    numbers starting from its first element (not decreasing numbers). 
    It returns omegai, gi, deltai.
    TODO: uniform the notation with the documentation.
    """
    #import matplotlib.pylab as plt
    import numpy as np
    import sys
    from scipy.interpolate import interp1d
    print("fit_multipole_const :: ")
    nbin = int(nbin)
    eta = 0.005 # This is the Lorentzian broadening that would be used???
   #safe_shift =  -10. # This is a little trick to avoid x=zero which introduces errors. 
   #preen = preen + safe_shift
    totalint = np.trapz(predata,preen)
    totdeltax = abs( preen[-1] - preen[0] )
    print(" Totdeltax, np.size(preen), dx:", totdeltax, np.size(preen), ( preen[-1] - preen[0] ) / float( np.size(preen) - 1 ))
    print(" Number of poles (nbin):", nbin)
    print(" Total integral:", totalint )
    print(" Total integral / nbin:", totalint / float(nbin))
    # This is the supposed single interval width
    delta = totdeltax / float(nbin)
    # Test plot
   #plt.plot(preen,predata,label="data")
   #plt.plot(preen,xfx,'-x',label="f(x)*x")
   #plt.plot(preen,(predata*preen),label="f(x)*x")
   #plt.plot(preen,fxonx,label="f(x)/x")
   #plt.plot(preen,(predata/preen),'-x',label="f(x)/x")
    ### =========================================== ###
    # Here we calculate the bins' bounds
    # First we want the x-axis grid to be finer than the density of poles
    en = preen
    data = predata
    if en.size < nbin:
        print("WARNING: Interpolating the data! (should be harmless)")
        interp_data = interp1d(preen, predata, kind = 'linear', axis = -1)
        en = np.linspace(preen[0],preen[-1],num=nbin)
        data = interp_data(en)
    print("en,preen",en,preen)
    print("data,predata",data,predata)
   #data[0:200] = 0
   #print(preen)
   #print("Removing at the beginning "+str(data[5])+" eV.")
   ## Use a finer grid if a lot of poles are required
   #if preen.size < nbin/2:
   #    print("WARNING: the multipole fit requires a finer energy grid.")
   #    print("It is going to be a bit slower.")
   #    newdx = delta/4.
   #    en = np.arange(preen[0],preen[-1],newdx)
   #    interp_data = interp1d(preen, predata, kind = 'linear', axis = -1)
   #    data = interp_data(en)
    bdensity = 1
    bounds = []
    ibound = 0
    gi = []
    omegai = []
    istart = 0
    funcint = 0
    bounds.append(en[istart])
    x0 = en[istart]
    x1 = x0
    print(" Getting poles...",)
    # CASE NPOLES == 1
    if int(nbin) == 1:
        gi.append(totalint)
        maxes = en[np.argwhere(data == np.amax(data))]
       #print(np.amax(data),maxes)
       #sys.exit()
        if maxes.size > 1:
            omegai.append(np.mean(maxes))
            print(" WARNING: Multiple max points. Taking the mean value.")
        else: 
            omegai.append(en[data.argmax()])
        bounds.append( en[-1] )
        ibound += 1
    # CASE NPOLES > 1
    else:
        for i in xrange(1,np.size(en)) : 
            x2 = en[i]
            #i2 = i
            #x3 = ( x1 + x2 ) / 2
            tmpint = np.trapz(data[istart:i],en[istart:i])
           #print(tmpint, partint)
            if x2 > x1 + delta:
               #print(" Bound found. ibound, en, tmpint:", ibound, en[i], tmpint)
                bounds.append(en[i] )
                # Formula to calculate omega_i
                tmpomegai = ( x1 + x2 ) / 2
                # Formula to calculate g_i
                tmpgi = 2. / np.pi * tmpint / tmpomegai
                gi.append( tmpgi )
                omegai.append( tmpomegai )
                #print(" gi, omegai:", tmpgi, tmpomegai # Test print)
                # Reset variables
                istart = i - 1
                ibound += 1
                x1 = x2
        # There should always be a patch-up last bin 
        if ibound < nbin:
            bounds.append( en[-1] )
            tmpomegai = ( x1 + x2 ) / 2
            tmpgi = 2. / np.pi * tmpint / tmpomegai
            gi.append( tmpgi )
            omegai.append( tmpomegai )
            ibound += 1
        print("Done.")
        # Add last value as the upper bound
        print(" ibound       = %4i (should be %g) " % (ibound, nbin))
        print(" Size(bounds) = %4i (should be %g) " % (np.size(bounds), nbin+1))
        print(" Size(omegai) = %4i (should be %g) " % (np.size(omegai), nbin))
      # if ibound < nbin:
      #     print("WARNING: too few bins! Adding a fictitious one.")
      #     gi.append(0.0)
      #     omegai.append(en[-1])
      #     bounds.append( en[-1] )
      #     ibound += 1
        # Here we assign the value as f is a sum of delta with one coefficient only (no pi/2 or else)
        # AKA: This gi is actually the pure lambda coefficient of the delta function
        # in the multipole model, equal to gi*omegai
        gi = np.array(gi)
        gi = np.pi/2*gi*omegai
   #print("TEST GI:", gi)
    omegai = np.array(omegai)
    # Here we restore the correct x axis removing safe_shift
   #omegai = omegai - safe_shift
    deltai = []
    sumcheck = 0
    print(" Calculating deltai...")
    for i in xrange(1,np.size(bounds)) :
        deltai.append(bounds[i]-bounds[i-1])
        sumcheck += abs(bounds[i]-bounds[i-1])
    deltai = np.array(deltai)
    print(" Check if sum of deltai gives the original length: ", sumcheck)
    if abs((sumcheck - abs(en[-1] - en[0])) / sumcheck) > 1E-02: 
        print()
        print(en[-1] - en[0])
        print("WARNING: the difference is", abs((sumcheck - abs(en[-1] - en[0])) / sumcheck))
    else: print("(OK)")
    #intcheck = np.pi/2*np.sum(gi[:]*omegai[:])
    intcheck = np.sum(gi)
   #print("gi:", gi)
    print(" Check if sum of gi gives the original total integral (origint): ", intcheck, totalint)
    if abs((intcheck - totalint) / intcheck) > 1E-02: 
        print()
        print("WARNING: the difference is", abs((intcheck - totalint) / intcheck))
    else: print("(OK)")
    print(" ibound       = %4i (should be %g) " % (ibound, nbin))
    print(" Size(bounds) = %4i (should be %g) " % (np.size(bounds), nbin+1))
    print(" Size(omegai) = %4i (should be %g) " % (np.size(omegai), nbin))
    print(" Size(deltai) = %4i (should be %g) " % (np.size(deltai), nbin))
    if ifilewrite == 1:
        # Print a file like Josh output
        # omega_i  gamma_i  g_i    delta_i
        #    .        .       .       .
        #    .        .       .       .
        #    .        .       .       .
        outname = "poles."+str(nbin)+".dat"
        outfile = open(outname,'w')
        # print(2-line header)
        outfile.write("### number of poles: %g\n" % (nbin))
        outfile.write("### omega_i  gamma_i(meaningless ATM)  g_i    delta_i \n")
        for j in xrange(np.size(omegai)) :
            outfile.write("%12.8e %12.8e %12.8e %12.8e\n" % (omegai[j], eta, gi[j], deltai[j]))
        outfile.close()
        print(" Parameters written in file", outname)
    return omegai, gi, deltai

def fit_multipole_fast(preen,predata,nbin):
    """
    This function fits a curve given by some dataset (preen,predata) 
    with a given number of poles (nbin).
    preen is supposed to be all positive and with increasing
    numbers starting from its first element (not decreasing numbers). 
    It can write out a file with the calculated parameters if the 
    appropriate flag is equal to 1 (ifilewrite).
    It returns omegai, gi, deltai.
    NOTE that these are not the same gi as in the formula: 
    these gi are actually equal to \pi/2*gi*omegai, i.e. lambdai.
    This version aims at getting rid of the interpolation and 
    just use the bare data. 
    TODO: uniform the notation with the documentation.
    """
    #import matplotlib.pylab as plt
    import numpy as np
    import sys
    print("fit_multipole_fast :: ")
    nbin = int(nbin)
    eta = 0.005 # This is the Lorentzian broadening that would be used???
   #safe_shift = 10. # This is a little trick to avoid x=zero which introduces errors. 
   #preen = preen + safe_shift
    totalint = np.trapz(predata,preen)
    totdeltax = abs( preen[-1] - preen[0] )
    print(" Totdeltax, np.size(preen), dx:", totdeltax, np.size(preen), ( preen[-1] - preen[0] ) / float( np.size(preen) - 1 ))
    print(" Number of poles (nbin):", nbin)
    print(" Total integral:", totalint )
    print(" Total integral / nbin:", totalint / float(nbin))
    # This is the supposed integral within a single interval
    partint = totalint / float(nbin)
    # First moment
    xfx = np.zeros(np.size(preen))
    xfx = predata * preen
    # First inverse moment
    fxonx = first_inverse_moment(preen,predata)
    # Test plot
   #plt.plot(preen,predata,label="data")
   #plt.plot(preen,xfx,'-x',label="f(x)*x")
   #plt.plot(preen,(predata*preen),label="f(x)*x")
   #plt.plot(preen,fxonx,label="f(x)/x")
   #plt.plot(preen,(predata/preen),'-x',label="f(x)/x")
    ### =========================================== ###
    # Here we calculate the bins' bounds
    # First we want the x-axis grid to be finer than the density of poles
    en = preen
    bdensity = 1
    bounds = []
    ibound = 0
    gi = []
    omegai = []
    istart = 0
    funcint = 0
    bounds.append(en[istart])
    x0 = en[istart]
    x1 = x0
    # Number of gaussians used in the integration (40 is safe, lower values are not tested too well)
    ngaussint = 20
    print(" Getting poles...",)
    # CASE NPOLES == 1
    if int(nbin) == 1:
        gi.append(totalint)
        omegai.append(en[predata.argmax()])
        bounds.append( en[-1] )
        ibound += 1
    # CASE NPOLES > 1
    else:
        for i in xrange(1,np.size(en)) : 
            x2 = en[i]
            #i2 = i
            x3 = ( x1 + x2 ) / 2
           #(tmpint,dummy) = fixed_quad(interpdata, x0, x2,(), ngaussint)
            tmpint = np.trapz(predata[istart:i],en[istart:i])
           #print(tmpint, partint)
            if tmpint > partint :
               #print(" Bound found. ibound, en, tmpint:", ibound, en[i], tmpint)
                bounds.append(en[i] )
                # Formula to calculate g_i
                #gi.append(float( 2. / np.pi * tmpint2))
               #tmpint2, dummy = fixed_quad(interpfxonx, x0, x3,(), ngaussint)
                tmpint2 = np.trapz(fxonx[istart:i],en[istart:i])
                tmpgi = 2. / np.pi * tmpint2
                gi.append( tmpgi )
                # Formula to calculate omega_i
               #tmpint, dummy = fixed_quad(interpxfx, x0, x3,(), ngaussint)
                tmpint = np.trapz(xfx[istart:i],en[istart:i])
                tmpomegai = np.sqrt( 2./np.pi * tmpint / tmpgi ) 
                omegai.append( tmpomegai )
                #print(" gi, omegai:", tmpgi, tmpomegai # Test print)
                # Reset variables
                istart = i - 1
                ibound += 1
        # There should always be a patch-up last bin 
        tmpint2 = np.trapz(fxonx[istart:],en[istart:])
        bounds.append( en[-1] )
        tmpgi = 2. / np.pi * tmpint2
        gi.append( tmpgi )
        tmpint = np.trapz(xfx[istart:],en[istart:])
        tmpomegai = np.sqrt( 2/np.pi * tmpint / tmpgi ) 
        omegai.append( tmpomegai )
        ibound += 1
        print("Done.")
        # Add last value as the upper bound
        print(" ibound       = %4i (should be %g) " % (ibound, nbin))
        print(" Size(bounds) = %4i (should be %g) " % (np.size(bounds), nbin+1))
        print(" Size(omegai) = %4i (should be %g) " % (np.size(omegai), nbin))
        if ibound < nbin:
            print("WARNING: too few bins! Adding a fictitious one.")
            gi.append(0.0)
            omegai.append(en[-1])
            bounds.append( en[-1] )
            ibound += 1
        # Here we assign the value as f is a sum of delta with one coefficient only (no pi/2 or else)
        # AKA: This gi is actually the pure lambda coefficient of the delta function
        # in the multipole model, equal to gi*omegai
        gi = np.array(gi)
        gi = np.pi/2*gi*omegai
   #print("TEST GI:", gi)
    omegai = np.array(omegai)
   ## Here we restore the correct x axis removing safe_shift
   #omegai = omegai - safe_shift
    deltai = []
    sumcheck = 0
    print(" Calculating deltai...")
    for i in xrange(1,np.size(bounds)) :
        deltai.append(bounds[i]-bounds[i-1])
        sumcheck += abs(bounds[i]-bounds[i-1])
    deltai = np.array(deltai)
    print(" Check if sum of deltai gives the original length: ", sumcheck)
    if abs((sumcheck - abs(en[-1] - en[0])) / sumcheck) > 1E-02: 
        print()
        print(en[-1] - en[0])
        print("WARNING: the difference is", abs((sumcheck - abs(en[-1] - en[0])) / sumcheck))
    else: print("(OK)")
    #intcheck = np.pi/2*np.sum(gi[:]*omegai[:])
    intcheck = np.sum(gi)
   #print("gi:", gi)
    print(" Check if sum of gi gives the original total integral (origint): ", intcheck, totalint)
    if abs((intcheck - totalint) / intcheck) > 1E-02: 
        print()
        print("WARNING: the difference is", abs((intcheck - totalint) / intcheck))
    else: print("(OK)")
    print(" ibound       = %4i (should be %g) " % (ibound, nbin))
    print(" Size(bounds) = %4i (should be %g) " % (np.size(bounds), nbin+1))
    print(" Size(omegai) = %4i (should be %g) " % (np.size(omegai), nbin))
    print(" Size(deltai) = %4i (should be %g) " % (np.size(deltai), nbin))
    return omegai, gi, deltai

def fit_multipole_old(preen,predata,nbin,ifilewrite=0,binmode=0):
    """
    This function fits a curve given by some dataset (preen,predata) 
    with a given number of poles (nbin).
    preen is supposed to be all positive and with increasing
    numbers starting from its first element (not decreasing numbers). 
    It can write out a file with the calculated parameters if the 
    appropriate flag is equal to 1 (ifilewrite).
    It returns omegai, gi, deltai.
    NOTE that these are not the same gi as in the formula: 
    these gi are actually equal to \pi/2*gi*omegai.
    TODO: uniform the notation with the documentation.
    """
    #import matplotlib.pylab as plt
    import numpy as np
    import sys
    from scipy.interpolate import interp1d
    from scipy.integrate import fixed_quad
    print("fit_multipole :: ")
    nbin = int(nbin)
    eta = 0.005 # This is the Lorentzian broadening that would be used???
    safe_shift = 10. # This is a little trick to avoid x=zero which introduces errors. 
    preen = preen + safe_shift
    totalint = np.trapz(predata,preen)
    totdeltax = abs( preen[-1] - preen[0] )
    print(" Totdeltax, np.size(preen), dx:", totdeltax, np.size(preen), ( preen[-1] - preen[0] ) / float( np.size(preen) - 1 ))
    print(" Number of poles (nbin):", nbin)
    print(" Total integral:", totalint)
    print(" Total integral / nbin:", totalint / float(nbin))
    # This is the supposed integral within a single interval
    partint = totalint / float(nbin)
    interpdata = interp1d(preen, predata[:], kind = 'linear', axis = -1)
    # First moment
    xfx = np.zeros(np.size(preen))
    xfx = predata * preen
    interpxfx = interp1d(preen, xfx[:], kind = 'linear', axis = -1)
    # First inverse moment
    fxonx = first_inverse_moment(preen,predata)
    interpfxonx = interp1d(preen, fxonx[:], kind = 'linear', axis = -1)
    # Test plot
    #plt.plot(preen,interpdata(preen),label="data")
    #plt.plot(preen,interpxfx(preen),'-x',label="f(x)*x")
    #plt.plot(preen,(predata*preen),label="f(x)*x")
    #plt.plot(preen,interpfxonx(preen),label="f(x)/x")
    #plt.plot(preen,(predata/preen),'-x',label="f(x)/x")
    ### =========================================== ###
    # Here we calculate the bins' bounds
    # First we want the x-axis grid to be finer than the density of poles
    en = resize_en(preen, nbin)
    bdensity = 1
    iboundcheck = 1
    while iboundcheck == 1:
        #print(" ### ========================= ###"
        #print(" ###   Calculating Delta_i     ###"
        bounds = []
        ibound = 0
        gi = []
        omegai = []
        istart = 0
        funcint = 0
        bounds.append(en[istart])
        x0 = en[istart]
        x1 = x0
        # Number of gaussians used in the integration (40 is safe, lower values are not tested too well)
        ngaussint = 20
        print(" Getting poles...",end="")
        for i in xrange(1,np.size(en)) : 
            x2 = en[i]
            x3 = ( x1 + x2 ) / 2
            (tmpint,dummy) = fixed_quad(interpdata, x0, x2,(), ngaussint)
            if tmpint > partint :
                (tmpint,dummy) = fixed_quad(interpdata, x0, x3,(), ngaussint)
                while abs( ( tmpint - partint ) / partint ) > 1E-06 :
                    if tmpint < partint :
                        x1 = x3
                        x3 = ( x3 + x2 ) / 2
                    else :
                        x2 = x3
                        x3 = ( x1 + x3 ) / 2
                    (tmpint,dummy) = fixed_quad(interpdata, x0, x3,(), ngaussint)
                #print(" Bound found. ibound, en, tmpint:", ibound, x3, tmpint
                bounds.append( x3 )
                # Formula to calculate g_i
                tmpint2, dummy = fixed_quad(interpfxonx, x0, x3,(), ngaussint)
                #if np.isnan(tmpint2): sys.exit(1)
                #gi.append(float( 2. / np.pi * tmpint2))
                tmpgi = 2. / np.pi * tmpint2 
                gi.append( tmpgi )
                # Formula to calculate omega_i
                tmpint, dummy = fixed_quad(interpxfx, x0, x3,(), ngaussint)
                tmpomegai = np.sqrt( 2/np.pi * tmpint / tmpgi ) 
                omegai.append( tmpomegai )
                #print(" gi, omegai:", tmpgi, tmpomegai # Test print
                # Reset variables
                x0 = x3
                istart = i
                ibound += 1
        print("Done.")
        # Add last value as the upper bound
        print(" ibound       = %4i (should be %g) " % (ibound, nbin))
        print(" Size(bounds) = %4i (should be %g) " % (np.size(bounds), nbin+1))
        print(" Size(omegai) = %4i (should be %g) " % (np.size(omegai), nbin))
        if ibound == nbin : # i.e. if it is as it is supposed to be
            print(" ibound == nbin, Fixing last value")
            bounds[-1] = en[-1]
            bounds = np.array(bounds)
            iboundcheck = 0
        # Prevent approximate integration to miss g_i and omega_i for last interval
        elif ibound == nbin - 1 :  # i.e. supposedly there is one bound missing ( ibound == nbin -1 )
            print(" ibound == nbin - 1. Calculating parameters for last bin...")
            bounds.append(en[-1])
            bounds = np.array(bounds)
            x0 = bounds[-2]
            x3 = bounds[-1]
            # Formula to calculate g_i
            tmpint2, dummy = fixed_quad(interpfxonx, x0, x3,(), ngaussint)
            tmpgi = 2. / np.pi * tmpint2 
            gi.append( tmpgi )
            # Formula to calculate omega_i
            tmpint, dummy = fixed_quad(interpxfx, x0, x3,(), ngaussint)
            tmpomegai = np.sqrt( tmpint / tmpint2 ) 
            omegai.append( tmpomegai )
            print(" gi, omegai:", tmpgi, tmpomegai )
            ibound += 1
            iboundcheck = 0
        else :
            print(" ibound has a non-compliant value. ibound:", ibound)
            iboundcheck = 1
            bdensity*=2
            print(" Will retry with a higher bdensity:", bdensity)
            en = resize_en(preen, bdensity*nbin)
            #sys.exit(1)
    omegai = np.array(omegai)
    gi = np.array(gi)
    # Here we assign the value as f is a sum of delta with one coefficient only (no pi/2 or else)
    # AKA: This gi is actually the pure lambda coefficient of the delta function
    # in the multipole model, equal to gi*omegai
    gi = np.pi/2*gi*omegai
    # Here we restore the correct x axis removing safe_shift
    omegai = omegai - safe_shift
    # Uncomment to change weights to single delta function
    # gi = gi / abs(omegai)
    #print(" bounds[-5:]:", bounds[-5:])
    #print(" omegai[-5:]:", omegai[-5:])
    #print(" Bounds:", bounds)
    #print(" Omegai:", omegai)
    deltai = []
    sumcheck = 0
    print(" Calculating deltai...")
    for i in xrange(1,np.size(bounds)) :
        deltai.append(bounds[i]-bounds[i-1])
        sumcheck += abs(bounds[i]-bounds[i-1])
    deltai = np.array(deltai)
    print(" Check if sum of deltai gives the original length: ", sumcheck)
    if abs((sumcheck - abs(en[-1] - en[0])) / sumcheck) > 1E-02: 
        print()
        print(en[-1] - en[0])
        print("WARNING: the difference is", abs((sumcheck - abs(en[-1] - en[0])) / sumcheck))
    else: print("(OK)")
    #intcheck = np.pi/2*np.sum(gi[:]*omegai[:])
    intcheck = np.sum(gi)
    print(" Check if sum of gi gives the original total integral (origint): ", intcheck, totalint)
    if abs((intcheck - totalint) / intcheck) > 1E-02: 
        print
        print("WARNING: the difference is", abs((intcheck - totalint) / intcheck))
    else: print("(OK)")
    #if np.isnan(intcheck): sys.exit(1)
    #if np.isnan(omegai): sys.exit(1)
    print(" ibound       = %4i (should be %g) " % (ibound, nbin))
    print(" Size(bounds) = %4i (should be %g) " % (np.size(bounds), nbin+1))
    print(" Size(omegai) = %4i (should be %g) " % (np.size(omegai), nbin))
    print(" Size(deltai) = %4i (should be %g) " % (np.size(deltai), nbin))
    if ifilewrite == 1:
        # Print a file like Josh output
        # omega_i  gamma_i  g_i    delta_i
        #    .        .       .       .
        #    .        .       .       .
        #    .        .       .       .
        outname = "poles."+str(nbin)+".dat"
        outfile = open(outname,'w')
        # print(2-line header)
        outfile.write("### number of poles: %g\n" % (nbin))
        outfile.write("### omega_i  gamma_i  g_i    delta_i \n")
        for j in xrange(np.size(omegai)) :
            outfile.write("%12.8f %12.8f %12.8f %12.8f\n" % (omegai[j], eta, gi[j], deltai[j]))
        outfile.close()
        print(" Parameters written in file", outname)
    return omegai, gi, deltai

def fit_multipole2(x,y,nbin,ifilewrite=0,binmode=0):
    """
    Another version with constant bin separation.
    NOTE: check first multipole model for actual 
    quantities returned. 
    --- Formulas used:
    f(\omega) = \pi/2 \sum_i^N g_i \omega_i \delta(\omega - \omega_i))
    - General formula: 
    \int_{\Delta_i} f(\omega) \omega^n = \pi/2 g_i \omega_i^{n+1}
    - Case n = -1 (first inverse moment): 
    g_i = 2/\pi \int_{\Delta_i} f(\omega)/\omega
    - Case n = 1 (first moment): 
    \omega_i = \sqrt{2/\pi/g_i }  \int_{\Delta_i} \omega f(\omega)
    - Actual gi that is returned:
    gi = \pi/2 g_i \omega_i
    """
    import numpy as np
    import sys
    from scipy.interpolate import interp1d
    from scipy.integrate import fixed_quad,quad, simps
    for i in range(x.size):
        if x[i]<0:
            print("UONOBNOBOOBUOBUOB some energy <<< 0 in multipole ")
            sys.exit()
    nbin = int(nbin)
    eta = 0.005 # This is the Lorentzian broadening that would be used???
    totint = np.trapz(y,x)
    totdx = abs( x[-1] - x[0] )
    # First moment
    xfx = x * y
    # First inverse moment
    fxonx = first_inverse_moment(x,y)
    interpxfx = interp1d(x, xfx, kind = 'linear', axis = -1)
    interpfxonx = interp1d(x, fxonx, kind = 'linear', axis = -1)
    print(" Totdeltax, np.size(x), dx:", totdx, np.size(x), ( x[-1] - x[0] ) / float( np.size(x) - 1 ))
    #for n in range(nbin):
    #    gi[n] = np.trapz(y[n:n+1],x[n:n+1])
    gi = []
    omegai = []
    first_i = []
    myint = []
    mysum = []
    step = x.size/nbin
    myrange = range(0,x.size,x.size/nbin)
    #print("--- multipole 2 nbin, len(myrange):", nbin, len(myrange))
    if (len(myrange) > nbin):
        myrange = myrange[:nbin]
    #for i in range(0,x.size,x.size/nbin):
    if int(nbin) == 1:
        gi.append(totint)
        omegai.append(x[y.argmax()])
        myint.append(totint)
        lambdai = np.array(gi)
        omegai = np.array(omegai)
     #  print(omegai)
     #  sys.exit()
    else:
        for i in myrange:
            #pint = np.trapz(fxonx[i:i+x.size/nbin],x[i:i+x.size/nbin])
            inv = np.trapz(fxonx[i:i+x.size/nbin],x[i:i+x.size/nbin])
            #gi.append(np.trapz(y[i:i+x.size/nbin],x[i:i+x.size/nbin]))
            gi.append(inv)
            #pom =np.trapz(xfx[i:i+x.size/nbin],x[i:i+x.size/nbin])
            first = np.trapz(xfx[i:i+x.size/nbin],x[i:i+x.size/nbin])
            first_i.append(first)
            omegai.append(np.sqrt(first/inv))
            #omegai.append((x[i+(x.size/nbin-1)]+x[i])/2)
            myint.append(np.trapz(y[i:i+x.size/nbin],x[i:i+x.size/nbin]))
            #mysum.append(np.sum(y[i:i+x.size/nbin]) / (x[i+x.size/nbin-1] - x[i]))
        gi = np.array(gi)
        omegai = np.array(omegai)
        lambdai = np.sqrt(gi*first_i)
        dint = (myint - lambdai)
        #print("--- multipole 2 nbin, len(myrange):", nbin, len(myrange))
        print("--- multipole 2 np.size(gi):", gi.size)
        #if gi.size > nbin:
        #    print(" WARNING: One bin too much created! Correcting...")
        #    gi = gi[:-1]
        #    omegai = omegai[:-1]
        xrest =  x.size - int(x.size/nbin)*nbin
        if xrest == 0:
            intrest  = 0.
        else:
            intrest  = np.trapz(y[-xrest:],x[-xrest:])
        print("--- x.size, nbin:", x.size, nbin)
        print("--- Size of one bin:", x.size/nbin)
        print("--- Remaining unused bin size:", xrest )
        print("--- Integral of neglected part:", intrest )
        myint = np.array(myint)
        #mysum = np.array(mysum)
        print("--- multipole:: np.np.trapz(ims):", totint )
        print("--- multipole:: np.np.trapz(fxonx):", np.trapz(fxonx,x))
        print("--- multipole:: np.sum(lambdai):", np.sum(lambdai))
        print("--- multipole:: np.sum(lambdai+dint):", np.sum(lambdai+dint))
        print("--- multipole:: np.sum(myint):", np.sum(myint))
        #print("--- multipole:: np.sum(mysum):", np.sum(mysum))
        print("--- multipole:: np.trapz(myint)-np.sum(lambdai):", (np.sum(lambdai) - np.sum(myint))/np.sum(myint))
        lambdai = lambdai + dint
        print("--- multipole:: np.sum(sqrt(dint**2)/totint):", np.sum(np.sqrt(dint**2)/totint))
    deltai = np.ones(lambdai.size)*totdx/nbin
    return omegai, lambdai, deltai

def write_f_as_sum_of_poles(preen,omegai,gi,deltai, eta = 0.001, ifilewrite = 0):
    """
    An attempt to rewrite the original function given the set of poles and weights
    """
    import numpy as np
    def lorentz_delta(x, a, b):
        """
        This is the lorentzian representation of a delta function. 
        x is the independent variable, a is the pole and b is the damping. 
        This is exact for b going to 0. 
        This small function can work also with numpy arrays. 
        """
        from numpy import pi
        florentz = b / pi / ((x - a)**2 + b**2)
        return florentz
    #for eni in preen
    #interpdata = interp1d(preen, predata[:], kind = 'linear', axis = -1)
    # Mh... Here I used a rule-of-thumb procedure.
    # TODO: think of a consistent and robust way of defining the x grid.
    nbin = 8*np.size(omegai)/eta/100
    en = resize_en(preen, nbin)
    f = np.zeros(np.size(en))
    f1 = np.zeros(np.size(en))
    for i in xrange(np.size(omegai)):
    #for oi in omegai:
        #f[:] += 2/np.pi * gi[i] * omegai[i]**2 * lorentz(en[:]**2,omegai[i]**2,eta)
        #for j in xrange(np.size(en)):
        f1[:] += 2.* gi[i] / omegai[i] * lorentz_delta(en[:]**2, omegai[i]**2, eta)
        f[:] += gi[i] * lorentz_delta(en[:], omegai[i], eta)
    if ifilewrite == 1:
        oname = "mpfit."+str(np.size(gi))+".dat"
        ofile = open(oname,'w')
        # print(2-line header)
        ofile.write("### Function reconstructed following many-pole fit. \n")
        ofile.write("### x   f(x) = \sum_i^N | \omega_i | \eta / ( ( \omega - \omega_i )^2 + \eta^2 ) \n")
        for j in xrange(np.size(en)):
            ofile.write("%15.8f %15.8f %15.8f\n" % (en[j], f[j], f1[j]))
        ofile.close()
        print(" Sum of poles written in file", oname)
    return en, f

def plot_pdf(en3, im3, omegai, lambdai, deltai, npoles, ik=0, ib=0, tag = '',fname = None):
    """
    Produces a hard copy of the fit, using a lorentzian representation.
    """
    import matplotlib.pylab as plt
    import pylab
    plt.figure(2)
    eta = 0.1
    enlor, flor = write_f_as_sum_of_poles(en3, omegai, lambdai, deltai, eta)
    plt.plot(enlor, flor,"-",label="sum of poles, eta: "+str(eta))
    plt.plot(en3,im3,"-",label="ImS(e-w)")
    plt.plot(omegai,lambdai,"go", label = "omegai, lambdai")
   #print("deltai",deltai)
   #print("lambdai",lambdai)
   #print("lambdai/deltai",lambdai/deltai)
   #import sys
   #sys.exit()
    plt.plot(omegai,lambdai/deltai,"ro", label = "omegai, lambdai/deltai")
    plt.title("ik: "+str(ik)+", ib: "+str(ib)+", npoles: "+str(npoles))
    plt.legend()
    if fname is None:
        fname = 'imS_fit_np'+str(npoles)+'_ik'+str(ik)+'_ib'+str(ib)+str(tag)+'.pdf'
    pylab.savefig(fname)
    plt.close(2)

def check_fix_coarse(en3, im3, omegai, lambdai, deltai, npoles, ):
    """
    This function solves the issue of not having enough data points. 
    Does nothing in case there is no issue.
    """
    if npoles > omegai.size:
        print(" WARNING: npoles used ("+str(npoles)+") is larger"+\
                " than poles x data array can give ("+str(omegai.size)+").")
       #print "WARNING: Reduce npoles. You are wasting resources!!!" 
        print(" Im(Sigma) will be interpolated to obtain the desired number of poles." )
        current_size = omegai.size
        counter = 0
        while npoles > current_size:
            counter += 1
            print(" WARNING: Arrays are too coarse.")
            print(" npoles, omegai.size:", npoles, omegai.size)
            print(" Filling arrays with interpolated values...")
            en1 = array_doublefill(en3)
            im1 = array_doublefill(im3)
            en3 = en1
            im3 = im1
            omegai, lambdai, deltai = fit_multipole(en1,im1,npoles)
            current_size = omegai.size
            if counter > 4:
                print(60*"=")
                print(" WARNING: You are trying too hard with too few points.") 
                print(" The array has been interpolated more than 4 times.") 
                print(" Maybe use less poles or calculate more points for Sigma?") 
                print(60*"=")
    return omegai2, lambdai2, deltai2

if __name__ == '__main__':
    import sys
    import numpy as np
    usage = 'Usage: %s npoles infile method(const/fast, optional)' % sys.argv[0]
    method = 'const2'
    print(len(sys.argv))
    try:
        infilename = sys.argv[2]
        nbin = sys.argv[1]
        ifilewrite = 1
        if len(sys.argv) > 3:
            method = sys.argv[3]
    except:
        print(usage)
        sys.exit(1)
    preen, predata = getdata_file(infilename)
    omegai, lambdai, deltai = fit_multipole(preen,predata,nbin,method,ifilewrite)
    plot_pdf(preen, predata, omegai, lambdai, deltai, nbin, ik=0, ib=0)
    print(np.sum(gi/np.square(omegai)))
    write_f_as_sum_of_poles(preen,omegai,gi,deltai,1)
#
