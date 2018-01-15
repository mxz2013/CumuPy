#!/usr/bin/env python
"""
Cumulant code that reads GW outputs and 
calculates the cumulant spectra functions.
TODO: mpi4py to run MPI
pyNFFT to realize the non-uniform FFT from 
gt to gw, this will make the convergence
faster.
"""
from __future__ import print_function
import numpy as np;
from multipole import *
import matplotlib.pylab as plt;
from scipy.interpolate import interp1d
import sys
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir

def calc_spf_gw(cs1,cs2,cs3,pjt1,pjt2,pjt3,bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res,
                ims, hartree, gwfermi, invar_eta):
    import numpy as np;
    import csv
    print("calc_spf_gw ::")
    newdx = 0.005
    #newen = np.arange(en[0], en[-1], newdx)
    if enmin < en[0] and enmax >= en[-1]:
        newen = np.arange(en[0],en[-1],newdx)
    elif enmin < en[0]:
        newen = np.arange(en[0],enmax,newdx)
    elif enmax >= en[-1] :
        newen = np.arange(enmin,en[-1],newdx)
    else :
        newen = np.arange(enmin,enmax,newdx)
    print (" ### Interpolation and calculation of A(\omega)_GW...  ")
    spftot = np.zeros((np.size(newen)));
    spftot_pjt1 = np.zeros((np.size(newen)));
    spftot_pjt2 = np.zeros((np.size(newen)));
    spftot_pjt3 = np.zeros((np.size(newen)));
    # Here we interpolate re and im sigma
    # for each band and k point
    for ik in kptrange:
        ikeff = ik + 1
        spf_sumb =  np.zeros((np.size(newen))) 
        spf_sumb1 =  np.zeros((np.size(newen))) 
        spf_sumb2 =  np.zeros((np.size(newen))) 
        spf_sumb3 =  np.zeros((np.size(newen))) 
        spftot_sumbp = np.zeros((np.size(newen)))  # sum over bands, projections,
                                                # k-resolved
        for ib in bdrange:
            ibeff = ib + bdgw_min
            interpres = interp1d(en, res[ik,ib], kind = 'linear', axis = -1)
            interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis = -1)
            #print("SKYDEBUT newen", newen[0], newen[-1])
            #print("SKYDEBUT en", en[0], en[-1])
            tmpres = interpres(newen)
            redenom = newen - hartree[ik,ib] - interpres(newen)
            tmpim = interpims(newen)
            spfkb =  abs(tmpim)/np.pi/(redenom**2 + tmpim**2)
            spfkb_pjt1 = spfkb*pjt1[ik,ib]*cs1#*wtk[ik] 
            spfkb_pjt2 = spfkb*pjt2[ik,ib]*cs2#*wtk[ik]
            spfkb_pjt3 = spfkb*pjt3[ik,ib]*cs3#*wtk[ik]

            spftot += spfkb*wtk[ik]

            spf_sumb += spfkb

            spf_sumb1 += spfkb*pjt1[ik,ib]*cs1 
            spf_sumb2 += spfkb*pjt2[ik,ib]*cs2
            spf_sumb3 += spfkb*pjt3[ik,ib]*cs3

           # spftot_sumbp += spf_sumb1+spf_sumb2+spf_sumb3

            spftot_pjt1 += spfkb*wtk[ik]*pjt1[ik,ib]*cs1
            spftot_pjt2 += spfkb*wtk[ik]*pjt2[ik,ib]*cs2
            spftot_pjt3 += spfkb*wtk[ik]*pjt3[ik,ib]*cs3
            with open("spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
                 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
 		writer.writerow(['# w-fermi','# spf','# spf_s','# spf_p','# spf_d','# w-hartree-ReSigma', '# ReSigma','# ImSigma'])
                writer.writerows(zip (newen-gwfermi, spfkb,spfkb_pjt1, spfkb_pjt2, spfkb_pjt3,
                                      redenom, tmpres, tmpim))
            with open("Imsigma-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
                 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip (newen, tmpim))
            #outnamekb = "spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
            #outfilekb = open(outnamekb,'w')
            #for ien in range(np.size(newen)):
            #    newen[ien] = newen[ien] - efermi
            #    outfilekb.write("%8.4f %12.8e %12.8e %12.8e %12.8e\n" % (newen[ien], spfkb[ien], redenom[ien], tmpres[ien], tmpim[ien]))
            #outfilekb.close()
        spftot_sumbp = spf_sumb1+spf_sumb2+spf_sumb3
        with open("spf_gw-k"+str("%02d"%(ikeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w-fermi','# sum on b','pjts','pjtp','pjtd',
                             'sum over b and pjt'])
            writer.writerows(zip(newen-gwfermi,spf_sumb,spf_sumb1,spf_sumb2,spf_sumb3,spftot_sumbp))
    sumkbp=spftot_pjt1+spftot_pjt2+spftot_pjt3
    return newen-gwfermi, spftot, spftot_pjt1, spftot_pjt2, spftot_pjt3,sumkbp



def find_eqp_resigma(en, resigma, gwfermi):
    """
    This function is supposed to deal with the plasmaron problem 
    and calculate the quasiparticle energy once it is fed with 
    resigma = \omega - \epsilon_H - \Re\Sigma. 
    It expects an array of increasing values on the x axis 
    and it will return 
    the x value of the last resigma=0 detected. 
    It should return the value of eqp and the number of zeros
    found (useful in case there are plasmarons or for debugging). 
    If no zeros are found, it will fit resigma with a line and 
    extrapolate a value.
    """
    nzeros = 0
    zeros = []
    tmpeqp = en[0]
    tol_fermi = 1e-3
    for i in xrange(1,np.size(resigma)):
        #print(resigma[i]*resigma[i-1] # DEBUG)
        if  resigma[i] == 0: # Yes, it can happen
            tmpeqp = en[i] 
            zeros.append(en[i])
            nzeros+=1
        elif (resigma[i]*resigma[i-1] < 0):
            tmpeqp = en[i-1] - resigma[i-1]*(en[i] - en[i-1])/(resigma[i] - resigma[i-1]) # High school formula
            zeros.append(tmpeqp)
            nzeros+=1
    if tmpeqp - gwfermi > tol_fermi: 
        tmpeqp=zeros[0]
    if nzeros==0 : 
        #print()
        #print (" WARNING: No eqp found! ")
        def fit_func(x, a, b): 
            return a*x + b
        from scipy.optimize import curve_fit
        params = curve_fit(fit_func, en, resigma)
        [a, b] = params[0]
        if -b/a < en[-1]:
            print("WTF!!! BYE!")
            #sys.exit()
        tmpeqp = -b/a
        zeros.append(tmpeqp)
   # elif nzeros>1 : 
   #     print(" WARNING: Plasmarons")
    return tmpeqp, nzeros

def calc_eqp_imeqp(nspin,spf_qp, wtk,bdrange, kptrange,bdgw_min, en,enmin,
                   enmax, res, ims, hartree, gwfermi, nkpt, nband, scgw,
                   Elda,pjt1,pjt2,pjt3,cs1,cs2,cs3):
    """
    This function calculates qp energies and corresponding
    values of the imaginary part of sigma for a set of
    k points and bands. 
    The function find_eqp_resigma() is used here.
    eqp and imeqp are returned. 
    """
    import csv
    from scipy import interp
    print("Calculating the QP energies::")
    eqp = np.zeros((nkpt,nband))
    imeqp = np.zeros((nkpt,nband))
    hartree = np.array(hartree)
    outname = "eqp.dat"
    outfile2 = open(outname,'w')
    outname = "imeqp.dat"
    outfile3 = open(outname,'w')
    newdx = 0.005
    newen = np.arange(en[0], en[-1], newdx)
    qpspftot = np.zeros((np.size(newen)))
    #qpspftot = np.zeros((np.size(newen)))
    qpspftot_up = np.zeros((np.size(newen)))
    qpspftot_down = np.zeros((np.size(newen)))
    #for ik in kptrange:
    #    for ib in bdrange:
    for ik in xrange(nkpt):
        ikeff = ik + 1
        for ib in xrange(nband):
            ibeff = ib + bdgw_min
            interpres = interp1d(en, res[ik,ib], kind = 'linear', axis = -1)
            tmpres = interpres(newen)
            temparray = np.array(newen - hartree[ik,ib] - tmpres)
            #temparray = newen - hartree[ik,ib] - tmpres
            
            #with open("ShiftReSigma"+str("%02d"%(ik))+"-b"+str("%02d"%(ib))+".dat", 'w') as f:
            #    writer = csv.writer(f, delimiter = '\t')
            #    writer.writerows(zip (newen-gwfermi, temparray))
            
            interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis = -1)
            tempim = interpims(newen)
            # New method to overcome plasmaron problem
            eqp[ik,ib], nzeros = find_eqp_resigma(newen,temparray,gwfermi)
            if nzeros==0: 
                print("WARNING NO QP at ik, ib:", ikeff, ibeff)
            if (eqp[ik,ib] > newen[0]) and (eqp[ik,ib] < newen[-1]): 
                if scgw == 1:
                    Elda_kb = eqp[ik,ib]
                else:
                    Elda_kb = Elda[ik,ib]
                imeqp[ik,ib] = interpims(Elda_kb)
            if spf_qp == 1 and nspin == 1:
                qpspfkb =  abs(imeqp[ik,ib])/np.pi/((newen-eqp[ik,ib])**2 + imeqp[ik,ib]**2)
                spfkb_pjt1 = qpspfkb*pjt1[ik,ib]*cs1 
                spfkb_pjt2 = qpspfkb*pjt2[ik,ib]*cs2 
                spfkb_pjt3 = qpspfkb*pjt3[ik,ib]*cs3 
                spfkb_spd = spfkb_pjt1+spfkb_pjt2+spfkb_pjt3
                qpspftot += qpspfkb*wtk[ik]
                with open("spf_qp"+"-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
 		    writer.writerow(['# w-fermi','# QP spectra', "pjts", "pjtp",
                        "pjtd", "sum_pjt"])
                    writer.writerows(zip (newen-gwfermi, qpspfkb,
                                          spfkb_pjt1,spfkb_pjt2,spfkb_pjt3,spfkb_spd))
            if spf_qp == 1 and nspin == 2 and ik%2 == 0:
                ikeff = int(ik/2 + 1)
                qpspfkb =  abs(imeqp[ik,ib])/np.pi/((newen-eqp[ik,ib])**2 + imeqp[ik,ib]**2)
                qpspfkb_1 =  abs(imeqp[ik,ib])/np.pi/((newen-eqp[ik,ib])**2 + imeqp[ik,ib]**2)
                qpspftot_up += qpspfkb*wtk[int(ik/2)]
                with open("spf_qp"+"-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-spin-up"+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
 		    writer.writerow(['# w-fermi','# QP spectra for spin up channel'])
                    writer.writerows(zip (newen-gwfermi, qpspfkb))

            if spf_qp == 1 and nspin == 2 and ik%2 != 0:
                ikeff = int(ik/2 + 1)
                qpspfkb =  abs(imeqp[ik,ib])/np.pi/((newen-eqp[ik,ib])**2 + imeqp[ik,ib]**2)
                qpspftot_down += qpspfkb*wtk[int(ik/2)]
                with open("spf_qp"+"-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-spin-down"+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
 		    writer.writerow(['# w-fermi','# QP spectra for spin down channel'])
                    writer.writerows(zip (newen-gwfermi, qpspfkb))
          #  else:
          #      imeqp[ik,ib] = interp(eqp[ik,ib], en, ims[ik,ib])
          #  ## Warning if imaginary part of sigma < 0 (Convergence problems?)
            #if imeqp[ik,ib] <= 0 : # SKYDEBUG do we really need to worry about this?? 
            #    print()
            #    print(""" WARNING: im(Sigma(e_k)) <= 0 !!! ik ib e_k
            #          im(Sigma(e_k)) = """, ik+1, ib+1, eqp[ik,ib], imeqp[ik,ib])
            outfile2.write("%14.5f" % (eqp[ik,ib]-gwfermi))
            outfile3.write("%14.5f" % (imeqp[ik,ib]))

        outfile2.write("\n")
        outfile3.write("\n")
    outfile2.close()
    outfile3.close()
    
    if spf_qp == 1 and nspin == 1:
        with open("spftot_qp.dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
 	    writer.writerow(['# w-fermi','# QP spectra'])
            writer.writerows(zip (newen-gwfermi, qpspftot))
    if spf_qp == 1 and nspin == 2:
        with open("spftot_qp.dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
 	    writer.writerow(['# w-fermi','# QP spectra for spin up', '# QP spectra for spin down'])
            writer.writerows(zip (newen-gwfermi, qpspftot_up, qpspftot_down))
    print("QP spectra calculation done!")
    return eqp, imeqp

def calc_ShiftImSig(en, ims_tmp, ikeff, ibeff ,Elda_kb, xfermi, Eplasmon,
                    metal_valence, invar_den, Rx, Ry, wps1, wps2, extrinsic, rc_toc = 0 ):
    """
    This module calculates Imsigma(w+e) which can be as input
    of the integration over omega, i.e., calc_integ_Imsig
    """
    import csv
    print ("SKY DEBUG en original:", en[0], en[-1],len(en))  
    ene=np.insert(en,0,-1000.0) 
    en2=np.insert(ene,len(ene), 1000.0)
    print ("SKY DEBUG en extended:", en2[0], en2[-1], len(en2))
    if rc_toc == 0: ## toc calculation 
        if metal_valence ==1:
            print("""
              WARNING: You are using TOC to calculate valence
              band of metal !! Calculate core states 
              together with the valence is not recommended. 
              """)

        NewEn_min =  - 10*(Eplasmon+abs(xfermi))  #default calculate 10 plasmon
        NewEn_min  = int(NewEn_min)
        if metal_valence == 1:
            NewEn_max = -(Elda_kb-xfermi) 
        else:
            NewEn_max = 2*(Eplasmon+abs(xfermi))
        print ("SKYDEBUG NewEn",  NewEn_min, NewEn_max)

        ims_tmp=np.insert(ims_tmp,0,ims_tmp[0])
        ims_tmp=np.insert(ims_tmp,len(ims_tmp),ims_tmp[-1])

        interpims = interp1d(en2, ims_tmp, kind = 'linear', axis
                                 = -1)

        imeqp_kb = interpims(Elda_kb)
        print("ImSigma(eqp)", imeqp_kb)
        newdx = invar_den  # must be chosen carefully so that 0 is
        # included in NewEn if metal_valence is on. invar_den can be 0.1*0.5^n, or 0.2. 
        NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
        NewEn_tmp = [x for x in NewEn_0 if abs(x) > 1e-6]
        NewEn_tmp = np.asarray(NewEn_tmp)
        NewEn_size = len(NewEn_tmp)
        if NewEn_tmp[-1]>=0 and NewEn_size == len(NewEn_0):
            print(""" ERROR: Zero is not in the intergration of
                  ImSigma(w) but your band crosess Fermi. Check your
                  invar_den.
                  """)
        ShiftEn = NewEn_tmp + Elda_kb 
        ShiftEn_0 = NewEn_0 + Elda_kb 
        if extrinsic == 0:              # SKY RRRR
          #  print("SKYDEBUG, wps", wps1, wps2)
            ShiftIms_tmp = interpims(ShiftEn)+(wps1*NewEn_tmp+wps2)*interpims(ShiftEn*np.sqrt(2))
            ShiftIms_0 = interpims(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))
        else: 
            interpR = interp1d(Rx, Ry, kind = 'linear', axis=-1) # SKY RRRR
            ShiftIms_tmp=interpims(ShiftEn)*interpR(NewEn_tmp)+(wps1*NewEn_tmp+wps2)*interpims(ShiftEn*np.sqrt(2))*interpR(NewEn_tmp*np.sqrt(2))
            ShiftIms_0=interpims(ShiftEn_0)*interpR(NewEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))*interpR(NewEn_0*np.sqrt(2))
            with open("R_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip (NewEn_tmp, ShiftIms_tmp,interpR(NewEn_tmp),interpR(ShiftEn)))

        with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w','# ImSigma(w-eqp)','##ImSigma'])
            writer.writerows(zip (NewEn_0, ShiftIms_0,interpims(NewEn_0)))
    return NewEn_tmp, ShiftIms_tmp

def calc_integ_Imsig(NewEn, ShiftIms, trange ):
    """
    This function takes care of the integration of Imsigma over
    frequency, which return a function f(t) and can be FFT to frequency
    plan by calc_FFT().
    """
    gtlist=[]

    for t in trange:
        tImag = t*1.j 
        area_tmp1 = 1.0/np.pi*abs(ShiftIms)*(np.exp(-(NewEn)*tImag)-1.0)*(1.0/((NewEn)**2))
        ct_tmp1 = np.trapz(area_tmp1, NewEn)

        ct_tot = ct_tmp1 
        gt_tmp = np.exp(ct_tot)
        gtlist.append(gt_tmp)
    return gtlist

def prep_FFT(invar_den, fftsize ):

    tfft_min = -2*np.pi/invar_den
    tfft_max = 0
    trange0 = np.linspace(tfft_min, tfft_max, fftsize)
    dtfft0 = abs(trange0[-1]-trange0[0])/fftsize

    denfft0 = 2*np.pi/abs(trange0[-1]-trange0[0])
    print("the energy resolution after FFT is",denfft0)
    fften_min0 = -2*np.pi/dtfft0
    fften_max0 = 0
    print ("the time step is", dtfft0)
    print("the size of fft is", fftsize)
    return trange0, dtfft0,denfft0, fften_min0 


def calc_FFT(eqp_kb, gt_list, fftsize,dtfft, enmin, newen_toc, denfft, invar_eta):

    """
    This function takes care of the FFT and retruns A(w) 
    """
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d

    fft_in = pyfftw.empty_aligned(fftsize, dtype='complex128')
    fft_out = pyfftw.empty_aligned(fftsize, dtype='complex128')
    ifft_object = pyfftw.FFTW(fft_in, fft_out,
                      direction='FFTW_BACKWARD',threads
                              = 1)
    cw=ifft_object(gt_list)*(fftsize*dtfft)

    freq = fftfreq(fftsize,dtfft)*2*np.pi
    s_freq = fftshift(freq)  
    s_go = fftshift(cw)

    im_sgo=s_go.imag
    re_sgo=s_go.real
    #with open("s_go"+str("%02d"%(ikeff))+".dat", 'w') as f:
    #    writer = csv.writer(f, delimiter = '\t')
    #    writer.writerows(zip (enrange, re_sgo,im_sgo))

    eta = 1.j*invar_eta
    wlist = np.arange(enmin,newen_toc[-1]+denfft, denfft)
    gwlist = []
    for w in wlist:
        Area2 = s_go/(w-eqp_kb-s_freq-eta) 
        c = np.trapz(Area2, dx = denfft)
        cwIm = 1./np.pi*c.imag
        gwlist.append(0.5/np.pi*cwIm)

    return wlist, gwlist


def calc_toc11(wps1,wps2,gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence,nkpt,nband,Rx, Ry,
                   extrinsic,pjt1,pjt2,pjt3,cs1,cs2,cs3):
    import numpy as np
    from scipy.interpolate import interp1d
    import csv
    ddinter = 0.005 
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc_tot =  np.zeros((np.size(newen_toc))) 
    spftot_pjt1 = np.zeros((np.size(newen_toc)));
    spftot_pjt2 = np.zeros((np.size(newen_toc)));
    spftot_pjt3 = np.zeros((np.size(newen_toc)));
    tol_fermi = 1e-3
    fftsize = FFTtsize
    norm = np.zeros((nkpt,nband))
    outname = "Norm_check_toc11.dat"
    outfile = open(outname,'w')
    trange, dtfft,denfft, fften_min  = prep_FFT(invar_den, fftsize)
    for ik in kptrange:
        ikeff = ik + 1
        spf_sumb =  np.zeros((np.size(newen_toc))) 
        spf_sumb_pjt1 =  np.zeros((np.size(newen_toc))) 
        spf_sumb_pjt2 =  np.zeros((np.size(newen_toc))) 
        spf_sumb_pjt3 =  np.zeros((np.size(newen_toc))) 
        spf_sumbp = np.zeros((np.size(newen_toc))) 
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib:",ikeff, ibeff)
            eqp_kb = eqp[ik, ib]
            if scgw == 1:
                Elda_kb = eqp[ik, ib] 
            else:
                Elda_kb = Elda[ik, ib] 

            if scgw == 1:
                xfermi = gwfermi 
            else:
                xfermi = lda_fermi 
            print("eqp:", eqp_kb-gwfermi)
            print("Elda:", Elda_kb-xfermi)
            print("xfermi:", xfermi)
            ims_tmp=ims[ik,ib] 
            if Elda_kb - xfermi <= tol_fermi:

                NewEn, ShiftIms = calc_ShiftImSig(en, ims_tmp, ikeff, ibeff,
                                                  Elda_kb, xfermi, Eplasmon,
                                                  metal_valence, invar_den, Rx,
                                                  Ry,wps1, wps2, extrinsic, rc_toc = 0 )

                gt_list = calc_integ_Imsig(NewEn, ShiftIms, trange ) 
                w_list, gw_list = calc_FFT(eqp_kb,gt_list, fftsize,dtfft, enmin,
                                            newen_toc, denfft, invar_eta)

                interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                interp_en = newen_toc
                spfkb = interp_toc(interp_en)
                #print("SKY DEBUG cs", cs1, cs2)
                spfkb_pjt1 = spfkb*pjt1[ik,ib]*cs1 
                spfkb_pjt2 = spfkb*pjt2[ik,ib]*cs2 
                spfkb_pjt3 = spfkb*pjt3[ik,ib]*cs3 
                spfkb_spd = spfkb_pjt1+spfkb_pjt2+spfkb_pjt3
                toc_tot += spfkb*wtk[ik]
                spftot_pjt1 += spfkb*pjt1[ik,ib]*wtk[ik]*cs1
                spftot_pjt2 += spfkb*pjt2[ik,ib]*wtk[ik]*cs2
                spftot_pjt3 += spfkb*pjt3[ik,ib]*wtk[ik]*cs3
                spf_sumb += spfkb
                spf_sumb_pjt1 += spfkb*pjt1[ik,ib]*cs1
                spf_sumb_pjt2 += spfkb*pjt2[ik,ib]*cs2
                spf_sumb_pjt3 += spfkb*pjt3[ik,ib]*cs3

                with open("TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-ext"+str(extrinsic)+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w-fermi','# spf_toc11',
                                     'spf_toc11_pjts',
                                     'spf_toc11_pjtp','spf_toc11_pjtd',
                                     'sum_spd'])
                    writer.writerows(zip (interp_en-gwfermi,
                                          spfkb,spfkb_pjt1,spfkb_pjt2,spfkb_pjt3,
                                         spfkb_spd))
                norm[ik,ib] = np.trapz(spfkb,interp_en)
                print("check the renormalization : :")
                print()
                print("the normalization of the spectral function is",norm[ik,ib])
                if abs(1-norm[ik,ib])>0.01:
                    print("WARNING: the renormalization is too bad!\n"+\
                          "Increase the time size to converge better.", ikeff,ibeff)
    
                outfile.write("%14.5f" % (norm[ik,ib]))
        outfile.write("\n")
        spf_sumbp = spf_sumb_pjt1+spf_sumb_pjt2+spf_sumb_pjt3
        with open("TOC11-k"+str("%02d"%(ikeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w-fermi','# spf_toc11 sum on b','# pjts',
                             '# pjtp','# pjtd', 'sum b and projections'])
            writer.writerows(zip (interp_en-gwfermi, spf_sumb, spf_sumb_pjt1,
                                  spf_sumb_pjt2,spf_sumb_pjt3, spf_sumbp))


    outfile.close()
    sumkbp=spftot_pjt1+spftot_pjt2+spftot_pjt3
    return interp_en-gwfermi, toc_tot,spftot_pjt1, spftot_pjt2, spftot_pjt3,sumkbp

def calc_toc11_new(wps1,wps2,gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence,nkpt,nband,Rx, Ry,
                   extrinsic,pjt1,pjt2,pjt3,cs1,cs2,cs3):
    import numpy as np
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    import csv
    print("calc_toc11 : :")
    print ("SKY DEBUG en original:", en[0], en[-1],len(en))  
    ene=np.insert(en,0,-1000.0) 
    en2=np.insert(ene,len(ene), 1000.0)
    print ("SKY DEBUG en original:", en2[0], en2[-1], len(en2))
    if metal_valence ==1:
        print("""
              WARNING: You are using TOC to calculate valence
              band of metal !! Calculate core states 
              together with the valence is not recommended. 
              """)
    ddinter = 0.005 
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc_tot =  np.zeros((np.size(newen_toc))) 
    spftot_pjt1 = np.zeros((np.size(newen_toc)));
    spftot_pjt2 = np.zeros((np.size(newen_toc)));
    spftot_pjt3 = np.zeros((np.size(newen_toc)));
    tol_fermi = 1e-3
    fftsize = FFTtsize
    norm = np.zeros((nkpt,nband))
    outname = "Norm_check_toc11.dat"
    outfile = open(outname,'w')
    for ik in kptrange:
        ikeff = ik + 1
        spf_sumb =  np.zeros((np.size(newen_toc))) 
        spf_sumb_pjt1 =  np.zeros((np.size(newen_toc))) 
        spf_sumb_pjt2 =  np.zeros((np.size(newen_toc))) 
        spf_sumb_pjt3 =  np.zeros((np.size(newen_toc))) 
        spf_sumbp = np.zeros((np.size(newen_toc))) 
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib:",ikeff, ibeff)
            eqp_kb = eqp[ik, ib]
            if scgw == 1:
                Elda_kb = eqp[ik, ib] 
            else:
                Elda_kb = Elda[ik, ib] 

            if scgw == 1:
                xfermi = gwfermi 
            else:
                xfermi = lda_fermi 
            print("eqp:", eqp_kb-gwfermi)
            print("Elda:", Elda_kb-xfermi)
            print("xfermi:", xfermi)
            if Elda_kb - xfermi <= tol_fermi:
                
                NewEn_min =  - 10*(Eplasmon+abs(xfermi))  #default calculate 10 plasmon
                #satellites
                NewEn_min  = int(NewEn_min)
                if metal_valence ==1:
                    NewEn_max = -(Elda_kb-xfermi) 
                else:
                    NewEn_max = 2*(Eplasmon+abs(xfermi))
                print ("SKYDEBUG NewEn",  NewEn_min, NewEn_max)
                tfft_min = -2*np.pi/invar_den
                tfft_max = 0
                trange = np.linspace(tfft_min, tfft_max, fftsize)
                dtfft = abs(trange[-1]-trange[0])/fftsize
                print ("the time step is", dtfft)
                print("the size of fft is", fftsize)
                
                ims_tmp=ims[ik,ib]
                ims_tmp=np.insert(ims_tmp,0,ims_tmp[0])
                ims_tmp=np.insert(ims_tmp,len(ims_tmp),ims_tmp[-1])

                interpims = interp1d(en2, ims_tmp, kind = 'linear', axis
                                         = -1)

                with open("Ims_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w','# ImSigma(w-eqp)'])
                    writer.writerows(zip (en, ims[ik,ib]))
                with open("Ims_extend_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w','# ImSigma(w-eqp)'])
                    writer.writerows(zip (en2, interpims(en2)))

                imeqp_kb = interpims(eqp_kb)
                print("ImSigma(eqp)", imeqp_kb)
                gt_list = []
                newdx = invar_den  # must be chosen carefully so that 0 is
                # included in NewEn if metal_valence is on. invar_den can be 0.1*0.5^n, or 0.2. 
                NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
                NewEn = [x for x in NewEn_0 if abs(x) > 1e-6]
                NewEn = np.asarray(NewEn)
                NewEn_size = len(NewEn)
                if NewEn[-1]>=0 and NewEn_size == len(NewEn_0):
                    print(""" ERROR: Zero is not in the intergration of
                          ImSigma(w) but your band crosess Fermi. Check your
                          invar_den.
                          """)
                    #sys.exit(0)
                ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                #print("SKYDEBUG ShiftEn", ShiftEn[0], ShiftEn[-1])
                ShiftEn_0 = NewEn_0 + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                #print("SKYDEBUG ShiftEn_0", ShiftEn_0[0], ShiftEn_0[-1])
                ShiftIms = interpims(ShiftEn)
                #with open ('Encut.dat', 'w') as f:
                #    writer = csv.writer(f, delimiter = '\t')
                #    writer.writerows(zip (NewEn, ShiftIms))
                if extrinsic == 0:              # SKY RRRR
                    print("SKYDEBUG, wps", wps1, wps2)
                    ShiftIms = interpims(ShiftEn)+(wps1*NewEn+wps2)*interpims(ShiftEn*np.sqrt(2))
                    ShiftIms_0 = interpims(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))
                else: 
                    interpR = interp1d(Rx, Ry, kind = 'linear', axis=-1) # SKY RRRR
                    #ShiftIms=interpims(ShiftEn)*interpR(ShiftEn)+(wps1*NewEn+wps2)*interpims(ShiftEn*np.sqrt(2))*interpR(ShiftEn*np.sqrt(2))
                    #ShiftIms_0=interpims(ShiftEn_0)*interpR(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))*interpR(ShiftEn_0*np.sqrt(2))
                    ShiftIms=interpims(ShiftEn)*interpR(NewEn)+(wps1*NewEn+wps2)*interpims(ShiftEn*np.sqrt(2))*interpR(NewEn*np.sqrt(2))
                    ShiftIms_0=interpims(ShiftEn_0)*interpR(NewEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))*interpR(NewEn_0*np.sqrt(2))
                    with open("R_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                        writer = csv.writer(f, delimiter = '\t')
                        writer.writerows(zip (NewEn, ShiftIms,interpR(NewEn),interpR(ShiftEn)))

                with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w','# ImSigma(w-eqp)','##ImSigma'])
                    writer.writerows(zip (NewEn_0, ShiftIms_0,interpims(NewEn_0)))
                    #interp_SIms = interp1d(NewEn, ShiftIms, kind = 'cubic', axis=-1) # SKY RRRR
                    #ShiftIms = interp_SIms(NewEn)
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

                im_sgo=s_go.imag
                re_sgo=s_go.real
                #with open("s_go"+str("%02d"%(ikeff))+".dat", 'w') as f:
                #    writer = csv.writer(f, delimiter = '\t')
                #    writer.writerows(zip (enrange, re_sgo,im_sgo))

                eta = 1.j*invar_eta
                w_list = np.arange(enmin,newen_toc[-1]+denfft,denfft)
                gw_list = []
                for w in w_list:
                    Area2 = s_go/(w-eqp_kb-s_freq-eta) 
                    c = np.trapz(Area2, dx = denfft)
                    cwIm = 1./np.pi*c.imag
                    gw_list.append(0.5/np.pi*cwIm)

                print("IFFT done .....")
                interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                interp_en = newen_toc

                spfkb = interp_toc(interp_en)
                #print("SKY DEBUG cs", cs1, cs2)
                spfkb_pjt1 = spfkb*pjt1[ik,ib]*cs1 
                spfkb_pjt2 = spfkb*pjt2[ik,ib]*cs2 
                spfkb_pjt3 = spfkb*pjt3[ik,ib]*cs3 
                spfkb_spd = spfkb_pjt1+spfkb_pjt2+spfkb_pjt3
                toc_tot += spfkb*wtk[ik]
                spftot_pjt1 += spfkb*pjt1[ik,ib]*wtk[ik]*cs1
                spftot_pjt2 += spfkb*pjt2[ik,ib]*wtk[ik]*cs2
                spftot_pjt3 += spfkb*pjt3[ik,ib]*wtk[ik]*cs3
                spf_sumb += spfkb
                spf_sumb_pjt1 += spfkb*pjt1[ik,ib]*cs1
                spf_sumb_pjt2 += spfkb*pjt2[ik,ib]*cs2
                spf_sumb_pjt3 += spfkb*pjt3[ik,ib]*cs3

                with open("TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-ext"+str(extrinsic)+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w-fermi','# spf_toc11',
                                     'spf_toc11_pjts',
                                     'spf_toc11_pjtp','spf_toc11_pjtd',
                                     'sum_spd'])
                    writer.writerows(zip (interp_en-gwfermi,
                                          spfkb,spfkb_pjt1,spfkb_pjt2,spfkb_pjt3,
                                         spfkb_spd))
                #outnamekb = "TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
                #outfilekb = open(outnamekb,'w')
                #en_toc11 = []
                #for i in range(len(interp_en)):
                #    en_toc11.append(interp_en[i])
                #    outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
                #outfilekb.close()
                norm[ik,ib] = np.trapz(spfkb,interp_en)
                print("check the renormalization : :")
                print()
                print("the normalization of the spectral function is",norm[ik,ib])
                if abs(1-norm[ik,ib])>0.01:
                    print("WARNING: the renormalization is too bad!\n"+\
                          "Increase the time size to converge better.", ikeff,ibeff)
    
                outfile.write("%14.5f" % (norm[ik,ib]))
        outfile.write("\n")
        spf_sumbp = spf_sumb_pjt1+spf_sumb_pjt2+spf_sumb_pjt3
        with open("TOC11-k"+str("%02d"%(ikeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w-fermi','# spf_toc11 sum on b','# pjts',
                             '# pjtp','# pjtd', 'sum b and projections'])
            writer.writerows(zip (interp_en-gwfermi, spf_sumb, spf_sumb_pjt1,
                                  spf_sumb_pjt2,spf_sumb_pjt3, spf_sumbp))
    outfile.close()
    sumkbp=spftot_pjt1+spftot_pjt2+spftot_pjt3
    return interp_en-gwfermi, toc_tot,spftot_pjt1, spftot_pjt2, spftot_pjt3,sumkbp

def calc_rc (wps1,wps2,gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta, wtk,nkpt,nband,
            Rx, Ry, extrinsic, core ,Eplasmon,pjt1,pjt2,pjt3,cs1,cs2,cs3):
    import numpy as np
    import pyfftw
    import csv
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    print("calc_rc : :")

    ddinter = 0.005 
    newen_rc = np.arange(enmin, enmax, ddinter)
    rc_tot =  np.zeros((np.size(newen_rc))) 
    spftot_pjt1 = np.zeros((np.size(newen_rc)));
    spftot_pjt2 = np.zeros((np.size(newen_rc)));
    spftot_pjt3 = np.zeros((np.size(newen_rc)));
    #pdos = np.array(pdos,nkpt,nband)
    fftsize = FFTtsize
    norm = np.zeros((nkpt,nband))
    outname = "Norm_check_rc.dat"
    outfile = open(outname,'w')
    for ik in kptrange:
        ikeff = ik + 1
        spf_sumb =  np.zeros((np.size(newen_rc))) 
        spf_sumb_pjt1 =  np.zeros((np.size(newen_rc))) 
        spf_sumb_pjt2 =  np.zeros((np.size(newen_rc))) 
        spf_sumb_pjt3 =  np.zeros((np.size(newen_rc))) 
        spf_sumbp = np.zeros((np.size(newen_rc))) 
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib:",ikeff, ibeff)
            eqp_kb = eqp[ik, ib]
            if scgw == 1:
                Elda_kb = eqp[ik, ib]
            else:
                Elda_kb = Elda[ik, ib]

            if scgw == 1:
                xfermi = gwfermi
            else:
                xfermi = lda_fermi
            print("eqp:", eqp_kb-gwfermi)
            print("Elda:", Elda_kb-xfermi)
            Done = False
            Es2 = 0
            while not Done:
                NewEn_min = int(en[0] + Es2)
                Es2 += 1
                if core == 0 and (NewEn_min + Elda_kb)*np.sqrt(2) > en[0]:
                    Done = True
                elif core == 1 and int(Elda_kb-3*Eplasmon) > en[0]:
                    NewEn_min = int(Elda_kb-3*Eplasmon) #-0.005
                    Done = True
            Done_max = False
            Es = 0
            while not Done_max:
                NewEn_max = en[-1] - Es
                Es += 1
                if NewEn_max*np.sqrt(2) < en[-1] and (NewEn_max+Elda_kb)*np.sqrt(2) < en[-1]:
                    Done_max = True
            tfft_min = -2*np.pi/invar_den
            tfft_max = 0
            trange = np.linspace(tfft_min, tfft_max,fftsize)
            dtfft = abs(trange[-1]-trange[0])/fftsize
            print ("the time step is", dtfft)
            print("the size of fft is", fftsize)
            interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis=-1)
            newdx = invar_den  # must be chosen carefully so that 0 is
            # included in NewEn. invar_den can be 0.1*0.5^n, or 0.2. 
            NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
            NewEn = [x for x in NewEn_0 if abs(x) > 1e-6]
            #print ("SKYDEBUG NewEn", NewEn_min, NewEn_max)

            NewEn = np.asarray(NewEn)
            NewEn_size = len(NewEn)
            if NewEn_size == len(NewEn_0):
                print("""ERROR:invar_den should  be 0.1*0.5*n where n is
                      integer number!!!""")
                sys.exit(0)
            ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
            ShiftEn_0 = NewEn_0 + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
            if extrinsic == 0:              # SKY RRRR
                print("SKYDEBUG, wps", wps1, wps2)
                ShiftIms = interpims(ShiftEn)+(wps1*NewEn+wps2)*interpims(ShiftEn*np.sqrt(2))
                ShiftIms_0 = interpims(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))
            else: 
                interpR = interp1d(Rx, Ry, kind = 'linear', axis=-1) # SKY RRRR
                print("SKYDEBUG ShiftEn", ShiftEn[0], ShiftEn[-1])
                print("SKYDEBUG NewEn", NewEn[0]*np.sqrt(2), NewEn[-1]*np.sqrt(2))
                #print("SKYDEBUG ShiftEn", ShiftEn[0], ShiftEn[-1])
                #ShiftIms = interpims(ShiftEn)*interpR(NewEn)
                #ShiftIms_0 = interpims(NewEn_0+Elda_kb)*interpR(NewEn_0)
                #ShiftIms=interpims(ShiftEn)*interpR(ShiftEn)+(wps1*NewEn+wps2)*interpims(ShiftEn*np.sqrt(2))*interpR(ShiftEn*np.sqrt(2))
                #ShiftIms_0=interpims(ShiftEn_0)*interpR(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))*interpR(ShiftEn_0*np.sqrt(2))
                ShiftIms=interpims(ShiftEn)*interpR(NewEn)+(wps1*NewEn+wps2)*interpims(ShiftEn*np.sqrt(2))*interpR(NewEn*np.sqrt(2))
                ShiftIms_0=interpims(ShiftEn_0)*interpR(NewEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))*interpR(NewEn_0*np.sqrt(2))

            with open("ShiftIms_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(['# w','# ImSigma(w-eqp)'])
                writer.writerows(zip (NewEn_0, ShiftIms_0, interpims(NewEn_0)))

            gt_list = []
            for t in trange:
                tImag = t*1.j 
                area_tmp = 1.0/np.pi*abs(ShiftIms)*(np.exp(-(NewEn)*tImag)-1.0)*(1.0/((NewEn)**2))
                ct_tmp = np.trapz(area_tmp, NewEn)
                gt_tmp = np.exp(ct_tmp)
                gt_list.append(gt_tmp)
            #with open("ShiftIms_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
            #                      'w') as f:
            #    writer = csv.writer(f, delimiter = '\t')
            #    writer.writerow(['# w','# ImSigma(w-eqp)'])
            #    writer.writerows(zip (NewEn, ShiftIms))

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

            eta = 1.j*invar_eta #the eta in the theta function 
            gw_list = []
            w_list = np.arange(enmin,newen_rc[-1]+denfft,denfft)
            for w in w_list:
                Area2 = s_go/(w-eqp_kb-s_freq-eta) 
                c = np.trapz(Area2, dx = denfft)
                cwIm = 1./np.pi*c.imag
                gw_list.append(0.5/np.pi*cwIm)

            print("IFFT done .....")
            interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
            interp_en = newen_rc
            spfkb = interp_toc(interp_en)

            spfkb_pjt1 = spfkb*pjt1[ik,ib]*cs1 
            spfkb_pjt2 = spfkb*pjt2[ik,ib]*cs2 
            spfkb_pjt3 = spfkb*pjt3[ik,ib]*cs3 
            spfkb_spd = spfkb_pjt1+spfkb_pjt2+spfkb_pjt3
            spftot_pjt1 += spfkb*pjt1[ik,ib]*wtk[ik]*cs1
            spftot_pjt2 += spfkb*pjt2[ik,ib]*wtk[ik]*cs2
            spftot_pjt3 += spfkb*pjt3[ik,ib]*wtk[ik]*cs3
            spf_sumb_pjt1 += spfkb*pjt1[ik,ib]*cs1
            spf_sumb_pjt2 += spfkb*pjt2[ik,ib]*cs2
            spf_sumb_pjt3 += spfkb*pjt3[ik,ib]*cs3

            rc_tot += spfkb*wtk[ik]
            spf_sumb += spfkb

            with open("spf_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-ext"+str(extrinsic)+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(['# w-fermi','# spf_toc11',
                                 'spf_toc11_pjts',
                                 'spf_toc11_pjtp','spf_toc11_pjtd',
                                 'sum_spd'])
                writer.writerows(zip (interp_en-gwfermi,
                                      spfkb,spfkb_pjt1,spfkb_pjt2,spfkb_pjt3,
                                     spfkb_spd))
            norm[ik,ib] = np.trapz(spfkb,interp_en)
            print("check the renormalization : :")
            print()
            print("the normalization of the spectral function is",norm[ik,ib])
            if abs(1-norm[ik,ib])>0.01:
                print("WARNING: the renormalization is too bad!\n"+\
                      "Increase the time size to converge better.", ikeff,ibeff)
    
            outfile.write("%14.5f" % (norm[ik,ib]))
        outfile.write("\n")
        spf_sumbp = spf_sumb_pjt1+spf_sumb_pjt2+spf_sumb_pjt3

        with open("spf_rc-k"+str("%02d"%(ikeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w-fermi','# spf_toc11 sum on b','# pjts',
                             '# pjtp','# pjtd', 'sum b and projections'])
            writer.writerows(zip (interp_en-gwfermi, spf_sumb, spf_sumb_pjt1,
                                  spf_sumb_pjt2,spf_sumb_pjt3, spf_sumbp))
    outfile.close()
    sum_kbp=spftot_pjt1+spftot_pjt2+spftot_pjt3
    return interp_en-gwfermi, rc_tot, spftot_pjt1, spftot_pjt2,spftot_pjt3,sum_kbp

def calc_rc_Josh (gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta, wtk, ehf):
    import numpy as np
    import pyfftw
    import csv
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    print("calc_rc : :")
    ddinter = 0.005 
    newen_rc = np.arange(enmin, enmax, ddinter)
    rc_tot =  np.zeros((np.size(newen_rc))) 
    #pdos = np.array(pdos)
    fftsize = FFTtsize
    outname = "Norm_check_Jrc.dat"
    outfile = open(outname,'w')

    for ik in kptrange:
        ikeff = ik + 1
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib:",ikeff, ibeff)
            eqp_kb = eqp[ik, ib]
            ehf_kb = ehf[ik, ib]
            if scgw == 1:
                Elda_kb = eqp[ik, ib]
            else:
                Elda_kb = Elda[ik, ib]

            if scgw == 1:
                xfermi = gwfermi
            else:
                xfermi = lda_fermi
            print("eqp:", eqp_kb-gwfermi)
            print("Elda:", Elda_kb-xfermi)
            Done = False
            Es2 = 0
            while not Done:
                NewEn_min = int(en[0] + Es2)
                Es2 += 1
                if NewEn_min > en[0] and NewEn_min + Elda_kb > en[0]:
                    Done = True
            Done_max = False
            Es = 0
            while not Done_max:
                NewEn_max = en[-1] - Es
                Es += 1
                if NewEn_max < en[-1] and NewEn_max+Elda_kb < en[-1]:
                    Done_max = True
            tfft_min = -2*np.pi/invar_den
            tfft_max = 0
            trange = np.linspace(tfft_min, tfft_max,fftsize)
            dtfft = abs(trange[-1]-trange[0])/fftsize
            print ("the time step is", dtfft)
            print("the size of fft is", fftsize)
            interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis=-1)
            newdx = invar_den  # must be chosen carefully so that 0 is
            # included in NewEn. invar_den can be 0.1*0.5^n, or 0.2. 
            NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
            
            NewEn = NewEn_0 #[x for x in NewEn_0 if abs(x) > 1e-6]
            NewEn = np.asarray(NewEn)
            NewEn_size = len(NewEn)
            #if NewEn_size == len(NewEn_0):
            #    print("""invar_den should  be 0.1*0.5*n where n is
            #          integer number!!!""")

            #    sys.exit(0)
            ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
            ShiftIms = interpims(ShiftEn)
            ShiftIms_0 = interpims(NewEn_0+Elda_kb)
            gt_list = []
            for t in trange:
                tImag = t*1.j 
                area_tmp = 1.0/np.pi*abs(ShiftIms)*(np.exp(-(NewEn)*tImag)-1.0+NewEn*tImag)*(1.0/((NewEn)**2))
                ct_tmp = np.trapz(area_tmp, NewEn)
                gt_tmp = np.exp(ct_tmp)
                gt_list.append(gt_tmp)
            #with open("ShiftIms_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
            #                      'w') as f:
            #    writer = csv.writer(f, delimiter = '\t')
            #    writer.writerows(zip (NewEn, ShiftIms))

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

            eta = 1.j*invar_eta #the eta in the theta function 
            gw_list = []
            w_list = np.arange(enmin,newen_rc[-1]+denfft,denfft)
            for w in w_list:
                Area2 = s_go/(w-ehf_kb-s_freq-eta) 
                c = np.trapz(Area2, dx = denfft)
                #c = 0
                #for i in range(fftsize-1):
                #    Area2 = 0.5*denfft*(s_go[i]/(w-eqp_kb-s_freq[i]-eta)
                #                + s_go[i+1]/(w-eqp_kb-s_freq[i+1]-eta))
                #    c += Area2
                cwIm = 1./np.pi*c.imag
                gw_list.append(0.5*wtk[ik]/np.pi*cwIm)

            print("IFFT done .....")
            interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
            interp_en = newen_rc
            #print("""the new energy range is (must be inside of above
             #     range)""",interp_en[0], interp_en[-1])
            spfkb = interp_toc(interp_en)
            rc_tot += spfkb
            with open ("spf_rc_Josh-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip(interp_en-gwfermi, spfkb/wtk[ik]))
            #spfkb = gw_list
            #toc_tot = [sum(i) for i in zip(toc_tot,gw_list)]
            #outnamekb = "spf_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
            #outfilekb = open(outnamekb,'w')
            #en_toc11 = []
            #for i in range(len(interp_en)):
            #    en_toc11.append(interp_en[i])
            #    outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
            #outfilekb.close()
            norm = np.trapz(spfkb,interp_en)/(wtk[ik])
            print("check the renormalization : :")
            print()
            print("the normalization of the spectral function is",norm)
            if abs(1-norm)>0.01:
                print("WARNING: the renormalization is too bad!\n"+\
                      "Increase the time size to converge better.", ikeff,ibeff)
            outfile.write("%14.5f \n" % (norm))
        outfile.write("\n")
    outfile.close()
    return interp_en-gwfermi, rc_tot


def calc_toc_Fabio(gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence,nkpt,nband):
    import numpy as np
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    import csv
    print("calc_toc11 : :")
    if metal_valence ==1:
        print("""
              WARNING: You are using TOC to calculate valence
              band of metal !! Please be sure that in SIG file,
              the maximum energy covers all -eqp_kb, otherwise, 
              the code might not run !!!
              """)
    ddinter = 0.005 
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc_tot =  np.zeros((np.size(newen_toc))) 
    tol_fermi = 1e-3
    fftsize = FFTtsize
    norm = np.zeros((nkpt,nband))
    outname = "Norm_check_toc11_Fabio.dat"
    outfile = open(outname,'w')
    for ik in kptrange:
        ikeff = ik + 1
        spf_sumb =  np.zeros((np.size(newen_toc))) 
        for ib in bdrange:
            ibeff = ib + bdgw_min
            print(" ik, ib:",ikeff, ibeff)
            eqp_kb = eqp[ik, ib]
            if scgw == 1:
                Elda_kb = eqp[ik, ib] 
            else:
                Elda_kb = Elda[ik, ib] 

            if scgw == 1:
                xfermi = gwfermi 
            else:
                xfermi = lda_fermi 
            print("eqp:", eqp_kb-gwfermi)
            print("Elda:", Elda_kb-xfermi)
            print("xfermi:", xfermi)
            if Elda_kb - xfermi <= tol_fermi:
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
                    NewEn_max = -(Elda_kb - xfermi) - Es
                    Es += 1
                    if NewEn_max < en[-1] and -NewEn_max+Elda_kb < en[-1]:
                        Done_max = True
                #if metal_valence == 1 and -Elda_kb < en[-1]:
                #    NewEn_max = -Elda_kb #-0.005
                tfft_min = -2*np.pi/invar_den
                tfft_max = 0
                trange = np.linspace(tfft_min, tfft_max, fftsize)
                dtfft = abs(trange[-1]-trange[0])/fftsize
                print ("the time step is", dtfft)
                print("the size of fft is", fftsize)
                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis
                                         = -1)
                imeqp_kb = interpims(eqp_kb)
                print("ImSigma(eqp)", imeqp_kb)
                gt_list = []
                newdx = invar_den  # must be chosen carefully so that 0 is
                # included in NewEn. invar_den can be 0.1*0.5^n, or 0.2. 
                NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
                NewEn = [x for x in NewEn_0 if abs(x) > 1e-6]
                NewEn = np.asarray(NewEn)
                NewEn_size = len(NewEn)
                if NewEn[-1]>=0 and NewEn_size == len(NewEn_0):
                    print("""Zero is not in the intergration of ImSigma(w),
                          please check invar_den""")

                    sys.exit(0)
                ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                ShiftIms = interpims(ShiftEn)
                ShiftIms_0 = interpims(NewEn_0+Elda_kb)
                #with open ('Encut.dat', 'w') as f:
                #    writer = csv.writer(f, delimiter = '\t')
                #    writer.writerows(zip (NewEn, ShiftIms))
                with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w','# ImSigma(w-eqp)'])
                    writer.writerows(zip (NewEn_0, ShiftIms_0))
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
                    gw_list.append(0.5*wtk[ik]/np.pi*cwIm)

                print("IFFT done .....")
                interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                interp_en = newen_toc

                spfkb = interp_toc(interp_en)
                toc_tot += spfkb
                spf_sumb += spfkb
                with open("TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['# w-fermi','# spf_toc11'])
                    writer.writerows(zip (interp_en-gwfermi, spfkb/wtk[ik]))
                #outnamekb = "TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
                #outfilekb = open(outnamekb,'w')
                #en_toc11 = []
                #for i in range(len(interp_en)):
                #    en_toc11.append(interp_en[i])
                #    outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
                #outfilekb.close()
                norm[ik,ib] = np.trapz(spfkb,interp_en)/(wtk[ik])
                print("check the renormalization : :")
                print()
                print("the normalization of the spectral function is",norm[ik,ib])
                if abs(1-norm[ik,ib])>0.01:
                    print("WARNING: the renormalization is too bad!\n"+\
                          "Increase the time size to converge better.", ikeff,ibeff)
    
                outfile.write("%14.5f" % (norm[ik,ib]))
        outfile.write("\n")
        with open("TOC11-k"+str("%02d"%(ikeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w-fermi','# spf_toc11 sum on b'])
            writer.writerows(zip (interp_en-gwfermi, spf_sumb/wtk[ik]))
    outfile.close()
    return interp_en-gwfermi, toc_tot
