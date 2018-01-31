#!/usr/bin/env python
"""
Cumulant code that reads GW outputs and 
calculates the cumulant spectra functions.
TODO: mpi4py to run MPI
pyNFFT to realize the non-uniform FFT from 
gt to gw, this will make the convergence
faster.
"calc_toc11" calculates the time-ordered
cumulant formula in [my prb] 
"""
from __future__ import print_function
import numpy as np;
import matplotlib.pylab as plt;
from scipy.interpolate import interp1d
import sys
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir


def calc_ShiftImSig(en, ims_tmp, ikeff, ibeff ,Elda_kb, xfermi, Eplasmon,
                    metal_valence, invar_den, Rx, Ry, wps1, wps2, extrinsic, rc_toc  ):
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
        
        #print ("SKY DEBUG TC calcualtion:", rc_toc )
        if metal_valence ==1:
            print("""
              WARNING: You are using TOC to calculate valence
              band of metal !! Calculate core states 
              together with the valence is not recommended. 
              """)

        NewEn_min =  -6*Eplasmon #+ Elda_kb  #default calculate 10 plasmon
        NewEn_min  = int(NewEn_min)
        
        if metal_valence == 1:
            NewEn_max = -Elda_kb+xfermi 
        else:
            NewEn_max = 1*Eplasmon+xfermi #+ Elda_kb #1*(Eplasmon+abs(xfermi))
        print ("SKYDEBUG NewEn",  NewEn_min, NewEn_max)

        ims_tmp=np.insert(ims_tmp,0,ims_tmp[0])
        ims_tmp=np.insert(ims_tmp,len(ims_tmp),ims_tmp[-1])

        interpims = interp1d(en2, ims_tmp, kind = 'linear', axis
                                 = -1)

        imeqp_kb = interpims(Elda_kb)
        #print("ImSigma(eqp)", imeqp_kb)
        newdx = invar_den  # must be chosen carefully so that 0 is
        # included in NewEn if metal_valence is on. invar_den can be 0.1*0.5^n, or 0.2. 
        NewEn_0 = np.arange(NewEn_min, NewEn_max, newdx)
        NewEn_tmp = [x for x in NewEn_0 if abs(x) > 1e-6]
        NewEn_tmp = np.asarray(NewEn_tmp)
        NewEn_size = len(NewEn_tmp)
        if NewEn_tmp[-1]>=0 and NewEn_size == len(NewEn_0) and metal_valence == 1:
            print(""" ERROR: Zero is not in the intergration of
                  ImSigma(w) but your band crosess Fermi. Check your
                  invar_den.
                  """)
            sys.exit(1)

        ShiftEn = NewEn_tmp + Elda_kb 
        ShiftEn_0 = NewEn_0 + Elda_kb 
        En_sp = NewEn_tmp*np.sqrt(2)+Elda_kb # surface plasmon energy
        En_sp0 = NewEn_0*np.sqrt(2)+Elda_kb # surface plasmon energy
        if extrinsic == 0:              # SKY RRRR
          #  print("SKYDEBUG, wps", wps1, wps2)
            ShiftIms_tmp = interpims(ShiftEn)+(wps1*NewEn_tmp+wps2)*interpims(En_sp)
            ShiftIms_0 = interpims(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(En_sp0)
        else: 
            interpR = interp1d(Rx, Ry, kind = 'linear', axis=-1) # SKY RRRR
            beta_sp = interpims(En_sp)*interpR(NewEn_tmp*np.sqrt(2))
            beta_sp0 = interpims(En_sp0)*interpR(NewEn_0*np.sqrt(2))
            ShiftIms_tmp=interpims(ShiftEn)*interpR(NewEn_tmp)+(wps1*NewEn_tmp+wps2)*beta_sp
            ShiftIms_0=interpims(ShiftEn_0)*interpR(NewEn_0)+(wps1*NewEn_0+wps2)*beta_sp0
            with open("R_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip (NewEn_tmp,
                                      ShiftIms_tmp,interpR(NewEn_tmp),interpims(En_sp),beta_sp))

        with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w','# ImSigma(w-eqp)','##ImSigma'])
            writer.writerows(zip (NewEn_0, ShiftIms_0,interpims(NewEn_0)))

    if rc_toc == 1: ## rc calculation 

        NewEn_min =  -6*Eplasmon # - Elda_kb  
        NewEn_min  = int(NewEn_min)
        NewEn_max = 8*Eplasmon #  
        #NewEn_max = 2.0  # TOC value  
        #NewEn_max = 5.0  #   
        #NewEn_max = 10.0  #   
        #NewEn_max = 15.0  #   
        #NewEn_max = 20.0  #   
        #NewEn_max = 30.0  #   
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
        En_sp = NewEn_tmp*np.sqrt(2)+Elda_kb # surface plasmon energy
        En_sp0 = NewEn_0*np.sqrt(2)+Elda_kb # surface plasmon energy
        if extrinsic == 0:              # SKY RRRR
          #  print("SKYDEBUG, wps", wps1, wps2)
            ShiftIms_tmp = interpims(ShiftEn)+(wps1*NewEn_tmp+wps2)*interpims(En_sp)
            ShiftIms_0 = interpims(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(En_sp0)
        else: 
            interpR = interp1d(Rx, Ry, kind = 'linear', axis=-1) # SKY RRRR
            beta_sp = interpims(En_sp)*interpR(NewEn_tmp*np.sqrt(2))
            beta_sp0 = interpims(En_sp0)*interpR(NewEn_0*np.sqrt(2))
            ShiftIms_tmp=interpims(ShiftEn)*interpR(NewEn_tmp)+(wps1*NewEn_tmp+wps2)*beta_sp
            ShiftIms_0=interpims(ShiftEn_0)*interpR(NewEn_0)+(wps1*NewEn_0+wps2)*beta_sp0
            with open("R_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip (NewEn_tmp,
                                      ShiftIms_tmp,interpR(NewEn_tmp),interpims(En_sp),beta_sp))

        with open("ShiftIms_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w','# ImSigma(w-eqp)'])
            writer.writerows(zip (NewEn_0, ShiftIms_0))

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

    freq = fftfreq(fftsize,dtfft)*2*np.pi ##
    s_freq = fftshift(freq) ##  
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
#<<<<<<< HEAD:sf_toc11.py
#                print("SKY DEBUG pjt1, pjt2 ", pjt1[ik,ibeff-1],
#                      pjt2[ik,ibeff-1])
#                spfkb_pjt1 = spfkb*pjt1[ik,ibeff-1]*cs1 
#                spfkb_pjt2 = spfkb*pjt2[ik,ibeff-1]*cs2 
#                spfkb_pjt3 = spfkb*pjt3[ik,ibeff-1]*cs3 
#=======
                print("SKY DEBUG pjt1, pjt2 ", pjt1[ik,ib], pjt2[ik,ib])
                spfkb_pjt1 = spfkb*pjt1[ik,ib]*cs1 
                spfkb_pjt2 = spfkb*pjt2[ik,ib]*cs2 
                spfkb_pjt3 = spfkb*pjt3[ik,ib]*cs3 
                spfkb_spd = spfkb_pjt1+spfkb_pjt2+spfkb_pjt3
                toc_tot += spfkb*wtk[ik]
                spftot_pjt1 += spfkb*pjt1[ik,ibeff-1]*wtk[ik]*cs1
                spftot_pjt2 += spfkb*pjt2[ik,ibeff-1]*wtk[ik]*cs2
                spftot_pjt3 += spfkb*pjt3[ik,ibeff-1]*wtk[ik]*cs3
                spf_sumb += spfkb
                spf_sumb_pjt1 += spfkb*pjt1[ik,ibeff-1]*cs1
                spf_sumb_pjt2 += spfkb*pjt2[ik,ibeff-1]*cs2
                spf_sumb_pjt3 += spfkb*pjt3[ik,ibeff-1]*cs3

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
###############################################################################
## below we implent the TOC original formula

def calc_ShiftImSig0(en, ims_tmp, ikeff, ibeff ,Elda_kb, xfermi, Eplasmon,
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

        NewEn_min = -6*Eplasmon #+ Elda_kb  #default calculate 10 plasmon
        NewEn_min  = int(NewEn_min)
        
        if metal_valence == 1:
            NewEn_max = -Elda_kb 
        else:
            NewEn_max = 1.0 #1*Eplasmon #+ Elda_kb #1*(Eplasmon+abs(xfermi))
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
        if NewEn_tmp[-1]>=0 and NewEn_size == len(NewEn_0) and metal_valence == 1:
            print(""" ERROR: Zero is not in the intergration of
                  ImSigma(w) but your band crosess Fermi. Check your
                  invar_den.
                  """)
            sys.exit(1)
        ShiftEn = NewEn_tmp + Elda_kb 
        ShiftEn_0 = NewEn_0 + Elda_kb 
        if extrinsic == 0:              # SKY RRRR
          #  print("SKYDEBUG, wps", wps1, wps2)
            ShiftIms_tmp = interpims(ShiftEn)+(wps1*NewEn_tmp+wps2)*interpims(ShiftEn*np.sqrt(2))
            ShiftIms_0 = interpims(ShiftEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))
        else: 
            #print("SKYDEBUG, ShiftEn", ShiftEn[0], ShiftEn[-1],
            #      ShiftEn[0]*np.sqrt(2), ShiftEn[-1]*np.sqrt(2))
            interpR = interp1d(Rx, Ry, kind = 'linear', axis=-1) # SKY RRRR
            print("SKYDEBUG, RX", Rx[0], Rx[-1])
            ShiftIms_tmp=interpims(ShiftEn)*interpR(NewEn_tmp)+(wps1*NewEn_tmp+wps2)*interpims(ShiftEn*np.sqrt(2))*interpR(NewEn_tmp*np.sqrt(2))
            ShiftIms_0=interpims(ShiftEn_0)*interpR(NewEn_0)+(wps1*NewEn_0+wps2)*interpims(ShiftEn_0*np.sqrt(2))*interpR(NewEn_0*np.sqrt(2))
            with open("R_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip (NewEn_tmp, ShiftIms_tmp,interpR(NewEn_tmp),interpR(ShiftEn)))

        with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w','# ImSigma(w-eqp)','##ImSigma'])
            writer.writerows(zip (NewEn_0, ShiftIms_0,interpims(NewEn_0)))

    if rc_toc == 1: ## rc calculation 

        NewEn_min =  -6*Eplasmon # - Elda_kb  
        NewEn_min  = int(NewEn_min)
        NewEn_max = 6*Eplasmon #  
        #NewEn_max = 2.0  # TOC value  
        #NewEn_max = 5.0  #   
        #NewEn_max = 10.0  #   
        #NewEn_max = 15.0  #   
        #NewEn_max = 20.0  #   
        #NewEn_max = 30.0  #   
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
            with open("R_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerows(zip (NewEn_tmp, ShiftIms_tmp,interpR(NewEn_tmp),interpR(ShiftEn)))

        with open("ShiftIms_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w','# ImSigma(w-eqp)'])
            writer.writerows(zip (NewEn_0, ShiftIms_0))

    return NewEn_0, ShiftIms_0

def calc_FFT0(ehf_kb, gt_list, fftsize,dtfft, enmin, newen_toc, denfft, invar_eta):

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

    freq = fftfreq(fftsize,dtfft)*2*np.pi ##
    s_freq = fftshift(freq) ##  
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
        Area2 = s_go/(w-ehf_kb-s_freq-eta) # here eqp=ehf 
        c = np.trapz(Area2, dx = denfft)
        cwIm = 1./np.pi*c.imag
        gwlist.append(0.5/np.pi*cwIm)

    return wlist, gwlist

def calc_integ_Imsig0(NewEn, ShiftIms, trange ):
    """
    This function takes care of the integration of Imsigma over
    frequency, which return a function f(t) and can be FFT to frequency
    plan by calc_FFT().
    """
    gtlist=[]

    for t in trange:
        tImag = t*1.j 
        area_tmp1 = 1.0/np.pi*abs(ShiftIms)*(np.exp(-(NewEn)*tImag)+tImag*NewEn-1.0)*(1.0/((NewEn)**2))
        ct_tmp1 = np.trapz(area_tmp1, NewEn)

        ct_tot = ct_tmp1 
        gt_tmp = np.exp(ct_tot)
        gtlist.append(gt_tmp)
    return gtlist

def calc_toc(ehf,wps1,wps2,gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
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
    outname = "Norm_check_toc.dat"
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
            print("ehf:", ehf_kb-gwfermi)
            print("Elda:", Elda_kb-xfermi)
            print("xfermi:", xfermi)
            ims_tmp=ims[ik,ib] 
            if Elda_kb - xfermi <= tol_fermi:

                NewEn, ShiftIms = calc_ShiftImSig0(en, ims_tmp, ikeff, ibeff,
                                                  Elda_kb, xfermi, Eplasmon,
                                                  metal_valence, invar_den, Rx,
                                                  Ry,wps1, wps2, extrinsic, rc_toc = 0 )

                gt_list = calc_integ_Imsig0(NewEn, ShiftIms, trange ) 
                w_list, gw_list = calc_FFT0(ehf_kb,gt_list, fftsize,dtfft, enmin,
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

                with open("TOC-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-ext"+str(extrinsic)+".dat", 'w') as f:
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
        with open("TOC-k"+str("%02d"%(ikeff))+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(['# w-fermi','# spf_toc11 sum on b','# pjts',
                             '# pjtp','# pjtd', 'sum b and projections'])
            writer.writerows(zip (interp_en-gwfermi, spf_sumb, spf_sumb_pjt1,
                                  spf_sumb_pjt2,spf_sumb_pjt3, spf_sumbp))


    outfile.close()
    sumkbp=spftot_pjt1+spftot_pjt2+spftot_pjt3
    return interp_en-gwfermi, toc_tot,spftot_pjt1, spftot_pjt2, spftot_pjt3,sumkbp

