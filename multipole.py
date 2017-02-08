#!/usr/bin/env python
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
