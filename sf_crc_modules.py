#!/usr/bin/env python

from __future__ import print_function
import numpy as np;
from multipole import *
import matplotlib.pylab as plt;
from scipy.interpolate import interp1d
import sys
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir
import csv

def calc_multipole(nkpt, nband,gwfermi,npoles, ims, kptrange, bdrange,bdgw_min, eqp, en, enmin, enmax):
    """
    Function that calculates frequencies and amplitudes
    of ImSigma using the multipole model. 
    """
    import numpy as np
    from multipole import fit_multipole_const #, write_f_as_sum_of_poles
    print(" ### ================== ###")
    print(" ###    Multipole fit   ###")
    print(" Number of poles:", npoles)
    newdx = 0.005
    if enmin < en[0] and enmax >= en[-1]:  
        newen = np.arange(en[0],en[-1],newdx)
    elif enmin < en[0]:  
        newen = np.arange(en[0],enmax,newdx)
    elif enmax >= en[-1] :  
        newen = np.arange(enmin,en[-1],newdx)
    else :  
        newen = np.arange(enmin,enmax,newdx)
    omegampole =  np.zeros((nkpt,nband,npoles))
    ampole =  np.zeros((nkpt,nband,npoles))
    for ik in kptrange:
        ikeff =  ik + 1
        for ib in bdrange:
            eqpkb = eqp[ik,ib]
            ibeff =  ib + bdgw_min
            if eqpkb > newen[-npoles]:
                omegampole[ik,ib] = omegampole[ik,ib-1]
                ampole[ik,ib] = ampole[ik,ib-1]
                print(" Eqp beyond available energy range. Values from lower band are taken.")
                continue
            else:
                print(" ik, ib", ikeff, ibeff)
                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis =
                                     -1)
                imskb = interpims(newen)
                # Here we take the curve starting from eqp and then we invert it
                # so as to have it defined on the positive x axis
                # and so that the positive direction is in the 
                # increasing direction of the array index
                if eqpkb <= gwfermi:
                    en3 = newen[newen<=eqpkb] # So as to avoid negative omegampole
                else:
                    en3 = newen[newen>eqpkb] # So as to avoid negative omegampole
                im3 = abs(interpims(en3)/np.pi)
                if en3.size == 0:
                    print()
                    print(" WARNING: QP energy is outside of given energy range!\n"+\
                            " This state will be skipped!\n"+\
                            "You might want to modify enmin/enmax.")
                    print(" eqp[ik,ib], newen[-1]", eqp[ik,ib] , newen[-1])
                    continue
                en3 = en3 - eqp[ik,ib]
                if eqp[ik,ib] <= gwfermi:
                    en3 = -en3[::-1] 
                    im3 = im3[::-1]
                #omegai, lambdai, deltai = fit_multipole_const(en3,im3,npoles)
                omegai, lambdai, deltai = fit_multipole(en3,im3,npoles,0) ## Matteo
                #used this funciton. TO DO: fit_multipole_const does not work

                # HERE WE MUST CHECK THAT THE NUMBER OF POLES 
                # IS NOT BIGGER THAN THE NUMBER OF POINTS THAT HAS TO BE FITTED
                if npoles > omegai.size:
                    omegampole[ik,ib][:omegai.size] = omegai 
                    ampole[ik,ib][:omegai.size] = np.true_divide(lambdai,(np.square(omegai)))
                    print()
                    print(" WARNING: npoles used ("+str(npoles)+") is larger"+\
                            " than x data array ("+str(omegai.size)+").")
                    print(" Reduce npoles. You are wasting resources!!!")
                else:
                    omegampole[ik,ib] = omegai 
                    ampole[ik,ib] = np.true_divide(lambdai,(np.square(omegai)))
                print(" Integral test. Compare \int\Sigma and \sum_j^N\lambda_j.")
                print(" 1/pi*\int\Sigma   =", np.trapz(im3,en3))
                print(" \sum_j^N\lambda_j =", np.sum(lambdai))
    # Writing out a_j e omega_j
    print(" ### Writing out a_j and omega_j...")
    outname = "a_j_np"+str(npoles)+".dat"
    outfile = open(outname,'w')
    outname = "omega_j_np"+str(npoles)+".dat"
    outfile2 = open(outname,'w')
    for ipole in xrange(npoles):
        for ik in xrange(nkpt):
            for ib in xrange(nband):
                outfile.write("%10.5f"  % (ampole[ik,ib,ipole]))
                outfile2.write("%10.5f" % (omegampole[ik,ib,ipole]))
            outfile.write("\n")
            outfile2.write("\n")
        outfile.write("\n")
        outfile2.write("\n")
    outfile.close()
    outfile2.close()
    return omegampole, ampole

def calc_crc(invar_eta, gwfermi, wtk, kptrange, bdrange, bdgw_min, omegampole,
                ampole, npoles, beta_greater, en_toc96, toc96_tot, imeqp, eqp):
    """
    Calculation of the unoccupied part of the CRC spectral function 
    """
    import numpy as np;
    from extmod_spf_mpole import f2py_calc_crc_mpole
    tol_fermi = 1e-3
    newdx = 0.005
    ftot_unocc = np.zeros((np.size(en_toc96)),order='Fortran')
    ftot = np.zeros((np.size(en_toc96)),order='Fortran')

    for ik in kptrange:
        ikeff = ik + 1
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib, ikeff, ibeff", ik, ib, ikeff, ibeff)
            #prefac=np.exp(-np.sum(ampole[ik,ib]))/np.pi*wtk[ik]*pdos[ib]*abs(imeqp[ik,ib])
            # Experimental fix for npoles dependence
            tmp = 1/np.pi*wtk[ik]*abs(imeqp[ik,ib])
            exponent = - np.sum(ampole[ik,ib]) - beta_greater[ik,ib] 
            prefac = np.exp(exponent)*tmp
            akb = ampole[ik,ib] # This is a numpy array (slice)
            bkb =  beta_greater[ik,ib]/npoles  # check here why we need the npoles!!
            omegakb = omegampole[ik,ib] # This is a numpy array (slice)
            eqpkb = eqp[ik,ib] - gwfermi
            imkb = imeqp[ik,ib]+invar_eta
            #tmpf1 = calc_spf_mpole(enexp,prefac,akb,omegakb,eqpkb,imkb,npoles)
            #print(nen, np.size(enexp))
            #tmpf = 0.0*tmpf
            if eqpkb < tol_fermi:
                #print("-omegakb", omegakb)
                tmpf = np.zeros((np.size(en_toc96)), order='Fortran')
                tmpf = f2py_calc_crc_mpole(tmpf,en_toc96,bkb,prefac,akb,omegakb,eqpkb,imkb)
                ftot_unocc = ftot_unocc + tmpf
                with open("CRC_unocc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
                     'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerows(zip (en_toc96, tmpf/wtk[ik]))

                #tmpf = calc_spf_mpole(enexp,prefac,akb,omegakb,eqpkb,imkb,npoles)
            #outnamekb = "spf_exp-k"+str("%02d"%(ikeff+1))+"-b"+str("%02d"%(ibeff+1))+"_np"+str(npoles)+"."+str(penergy)
            #outfilekb = open(outnamekb,'w')
            #for ien in xrange(nenexp):
            #    outfilekb.write("%8.4f %12.8f\n" % (enexp[ien], tmpf[ien]))
            #outfilekb.close()
    ftot = ftot_unocc + toc96_tot 
    return ftot


def A_model_crc(x,eqpkb,beta1, beta2,wp, eta_crc):
    import math
    G=0
    m=0
    while m<20:
        G += np.exp(-beta1-beta2)*(math.pow(beta1+beta2,
                                           m)-math.pow(beta1,m))/math.factorial(m)*(1./(x-eqpkb+m*(abs(wp)-1j*eta_crc) -1j*eta_crc))
        m += 1
    return (1.0/(math.pi))*abs(G.imag)

def calc_toc96 (gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence, imeqp,nkpt, nband):
    import numpy as np
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    import csv
    print("calc_toc96 : :")
    metal_valence = 1
    ddinter = 0.005
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc96_tot = np.zeros((np.size(newen_toc)))
    beta_greater = np.zeros((nkpt,nband))
    npoles_crc = 1
    #crc_tot = np.zeros((np.size(newen_toc)))
    tol_fermi = 1e-3
    #pdos = np.array(pdos)
    fftsize = FFTtsize
    outname = "occupation_toc96.dat"
    outfile = open(outname,'w')
    for ik in kptrange:
        ikeff = ik + 1
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib:",ikeff, ibeff)
            eqp_kb = eqp[ik, ib]
            imeqp_kb = imeqp[ik, ib]
            if scgw == 1:
                Elda_kb = eqp[ik, ib]
            else:
                Elda_kb = Elda[ik, ib]
            if scgw == 1:
                xfermi = gwfermi
            else:
                xfermi = lda_fermi
            if Elda_kb - xfermi  <= tol_fermi:
                Done = False
                Es2 = 0
                while not Done:
                    if -2*Eplasmon <= en[-1]:
                        NewEn_min = int(-2*Eplasmon + Es2)
                    else:
                        NewEn_min = int(-2*Eplasmon - Es2)
                    Es2 += 1
                    if NewEn_min > en[0] and NewEn_min + Elda_kb > en[0]:
                        Done = True
                Done_max = False
                Es = 0
                while not Done_max:
                    NewEn_max = -(Elda_kb-xfermi) - Es
                    Es += 1
                    if NewEn_max < en[-1] and NewEn_max+Elda_kb < en[-1]:
                        Done_max = True

                Done_max_crc = False
                Es_crc = 0
                while not Done_max_crc:
                    NewEn_max_crc = en[-1] - Es
                    Es_crc += 1
                    if NewEn_max_crc < en[-1] and NewEn_max_crc+Elda_kb < en[-1]:
                        Done_max_crc = True
                if metal_valence == 1:
                    NewEn_max = -Elda_kb #-0.005
                tfft_min = -2*np.pi/invar_den
                tfft_max = 0
                trange = np.linspace(tfft_min, tfft_max, fftsize)
                dtfft = abs(trange[-1]-trange[0])/fftsize
                print ("the time step is", dtfft)
                print("the size of fft is", fftsize)
                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis
                                         = -1)
                gt_list = []
                newdx = invar_den  # must be chosen carefully so that 0 is
                # included in NewEn. invar_den can be 0.1*0.5^n, or 0.2. 
                NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
                NewEn = [x for x in NewEn_0 if abs(x) > 1e-6]
                NewEn = np.asarray(NewEn)
                NewEn_greater = np.arange(NewEn_max+1, NewEn_max_crc, newdx)
                NewEn_crc =  np.arange(NewEn_min, NewEn_max_crc, newdx)
                NewEn_size = len(NewEn)
                if NewEn[-1]>=0 and NewEn_size == len(NewEn_0):
                    print("""Zero is not in the intergration of ImSigma(w),
                          please check invar_den""")

                    sys.exit(0)
                ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                ShiftIms = interpims(ShiftEn)
                ShiftIms_0 = interpims(NewEn_0+Elda_kb)
                ShiftIms_crc = interpims(NewEn_greater + Elda_kb )/np.pi
                #bcrc = 0
                #for i in xrange(len(NewEn_greater)):
                #    bcrc += abs(ShiftIms_crc[i])/((NewEn_greater[i])**2)
                #    
                #B_crc_new = bcrc
                #print("SKYDEBUG Bcrc",  B_crc_new)
                im_greater = abs(interpims(NewEn_greater))/np.pi
                en_greater = NewEn_greater - Elda_kb
                omega_greater, lambda_greater, delta_greater = fit_multipole_const(en_greater,im_greater,1)
                beta_greater[ik,ib] = map(abs,lambda_greater)/(omega_greater)**2

                #im_lesser =  abs(interpims(NewEn))/np.pi 
                #en_lesser = NewEn - Elda_kb
                #omega_lesser, lambda_lesser, delta_lesser = fit_multipole_const(en_lesser,im_lesser,150)
                #beta_lesser = map(abs,lambda_lesser)/(omega_lesser)**2

                #print("SKYDEBUG beta_greater", beta_greater,np.exp(-beta_greater))

                #print("SKYDEBUG beta_lesser", beta_lesser, np.exp(-beta_lesser))
                #with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                #    writer = csv.writer(f, delimiter = '\t')
                #    writer.writerows(zip (NewEn_0, ShiftIms_0))
                for t in trange:
                    tImag = t*1.j 
                    area_tmp1 = 1.0/np.pi*abs(ShiftIms)*(np.exp(-(NewEn)*tImag)-1.0)*(1.0/((NewEn)**2))
                    ct_tmp1 = np.trapz(area_tmp1, NewEn)

                    ct_tot = ct_tmp1 
                    gt_tmp = np.exp(ct_tot)
                    gt_list.append(gt_tmp)

                denfft = 2*np.pi/abs(trange[-1]-trange[0])
                print("the energy resolution after FFT is",denfft)
                fften_min = -2*np.pi/dtfft
                fften_max = 0
                enrange = np.arange(fften_min,NewEn[-1],denfft)
                print("IFFT of ")
                print("kpoint = %02d" % (ikeff))
                print("band=%02d" % (ibeff))

                fft_in = pyfftw.empty_aligned(fftsize, dtype='complex128')
                fft_out = pyfftw.empty_aligned(fftsize, dtype='complex128')
                ifft_object = pyfftw.FFTW(fft_in, fft_out,
                                  direction='FFTW_BACKWARD',threads
                                          = 1)
                cw=ifft_object(gt_list)*(fftsize*dtfft)

                freq = fftfreq(fftsize,dtfft)*2*np.pi
                s_freq = fftshift(freq)  
                s_go = fftshift(cw)
                eta = 1.j*invar_eta
                w_list = np.arange(enmin,newen_toc[-1]+denfft,denfft)
                gw_list = []
                for w in w_list:
                    Area2 = s_go/(w-eqp_kb-s_freq-eta) 
                    c = np.trapz(Area2, dx = denfft)
                    cwIm = 1./np.pi*c.imag
                    gw_list.append(0.5*wtk[ik]*np.exp(-beta_greater[ik,ib])/np.pi*cwIm)

                print("IFFT done .....")
                interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                interp_en = newen_toc

                spfkb = interp_toc(interp_en)
                toc96_tot += spfkb
                with open("TOC96-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerows(zip (interp_en-gwfermi, spfkb/wtk[ik]))
                #outnamekb = "TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
                #outfilekb = open(outnamekb,'w')
                #en_toc11 = []
                #for i in xrange(len(interp_en)):
                #    en_toc11.append(interp_en[i])
                #    outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
                #outfilekb.close()
                #Gw_unocc = wtk[ik]* A_model_crc(interp_en, eqp_kb ,beta_lesser[0],
                #                       beta_greater[0],omega_lesser,
                #                       imeqp_kb+eta) 
                #crc_tot += spfkb+Gw_unocc
                #with open("CRC_unocc"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                #    writer = csv.writer(f, delimiter = '\t')
                #    writer.writerows(zip(interp_en-gwfermi, Gw_unocc))
                print("Calculate occupation of TOC96 : :")
                norm = np.trapz(spfkb,interp_en)/(wtk[ik])
                print("The occupation for ik, ib, is", ikeff, ibeff, norm)
    
                outfile.write("%8.4f %12.8e \n" % (newdx, norm))
    outfile.close()
    return interp_en-gwfermi, toc96_tot, beta_greater #, crc_tot
