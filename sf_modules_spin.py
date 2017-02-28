#!/usr/bin/env python
"""
cumulant code that reads GW outputs and 
calculates the cumulant spectra functions.
"""
from __future__ import print_function
import numpy as np;
import matplotlib.pylab as plt;
from scipy.interpolate import interp1d
import sys
import csv
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir

def calc_spf_gw_spin(bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res,
                     ims, hartree, gwfermi):
    print("calc_spf_gw_spin : :")
    import numpy as np;
    newdx = 0.005
    if enmin < en[0] and enmax >= en[-1]:
        newen = np.arange(en[0],en[-1],newdx)
    elif enmin < en[0]:
        newen = np.arange(en[0],enmax,newdx)
    elif enmax >= en[-1] :
        newen = np.arange(enmin,en[-1],newdx)
    else :
        newen = np.arange(enmin,enmax,newdx)
    print (" ### Interpolation and calculation of spin-polarized A(\omega)_GW ...  ")
    spftot_up = np.zeros((np.size(newen)));
    spftot_down = np.zeros((np.size(newen)));
    for ik in kptrange:
        if ik %2 == 0: #spin up chanel 
            ikeff = int(ik/2 + 1)
            ikwtk1 = int(ik/2)
            print( " spin up channel, k point = %02d " % (ikeff))
            for ib in bdrange:
                ibeff = ib + bdgw_min
                interpres = interp1d(en, res[ik,ib], kind = 'linear', axis = -1)
                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis = -1)
                tmpres = interpres(newen)
                redenom = newen - hartree[ik,ib] - interpres(newen)
                #print "ik ib minband maxband ibeff hartree[ik,ib]", ik, ib, minband, maxband, ibeff, hartree[ik,ib]
                tmpim = interpims(newen)
                spfkb = wtk[ikwtk1] * abs(tmpim)/np.pi/(redenom**2 + tmpim**2)
                spftot_up += spfkb
                with open("spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-spin-up"+".dat",
                    'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerows(zip(newen-gwfermi, spfkb/wtk[ikwtk1],
                                         redenom, tmpres, tmpim))
        else:
            ikeff = int(ik/2 + 1)
            ikwtk2 = int(ik/2) 
            print( " spin down channel, k point = %02d " % (ikeff))
            for ib in bdrange:
                ibeff = ib + bdgw_min
                interpres = interp1d(en, res[ik,ib], kind = 'linear', axis = -1)
                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis = -1)
                tmpres = interpres(newen)
                redenom = newen - hartree[ik,ib] - interpres(newen)
                #print "ik ib minband maxband ibeff hartree[ik,ib]", ik, ib, minband, maxband, ibeff, hartree[ik,ib]
                tmpim = interpims(newen)
                spfkb = wtk[ikwtk2] * abs(tmpim)/np.pi/(redenom**2 + tmpim**2)
                spftot_down += spfkb
                with open("spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-spin-down"+".dat",
                    'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerows(zip(newen-gwfermi, spfkb/wtk[ikwtk2],
                                         redenom, tmpres, tmpim))

    return newen-gwfermi, spftot_up, spftot_down

def calc_toc11_spin (gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize,
                     en,enmin, enmax, eqp,Elda, scgw, Eplasmon, ims, invar_den,
                     invar_eta, wtk, metal_valence):
    import numpy as np
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    print("calc_toc11_spin : :")
    ddinter = 0.005 
    tol_fermi = 1e-3
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc_tot_up =  np.zeros((np.size(newen_toc))) 
    toc_tot_down =  np.zeros((np.size(newen_toc))) 
    #pdos = np.array(pdos)
    fftsize = FFTtsize
    outname = "Norm_check_toc11.dat"
    outfile = open(outname,'w')

    for ik in kptrange:
        if ik %2 == 0:
            ikeff = int(ik/2 + 1)
            ikwtk1 = int(ik/2)
            print( "spin up channel, k point = %02d " % (ikeff))
            for ib in bdrange:
                ibeff = ib + bdgw_min
                print(" ik, ib:",ikeff, ibeff)
                eqp_kb = eqp[ik,ib]
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
                        if NewEn_max < en[-1] and NewEn_max+Elda_kb < en[-1]:
                            Done_max = True
                    if metal_valence == 1 and -Elda_kb < en[-1]:
                        NewEn_max = -Elda_kb #-0.005
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
                        print("""Zero is not in the intergration of
                              ImSigma(\omega), please check invar_den""")

                        sys.exit(0)
                    ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                    ShiftIms = interpims(ShiftEn)
                    ShiftIms_0 = interpims(NewEn_0+Elda_kb)
                    with open("ShiftIms_toc11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-up"+".dat", 'w') as f:
                        writer = csv.writer(f, delimiter = '\t')
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
                        gw_list.append(0.5*wtk[ikwtk1]/np.pi*cwIm)

                    print("IFFT done .....")
                    interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                    interp_en = newen_toc

                    spfkb = interp_toc(interp_en)
                    toc_tot_up += spfkb
                    with open("TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-up"+".dat", 'w') as f:
                        writer = csv.writer(f, delimiter = '\t')
                        writer.writerows(zip (interp_en-gwfermi,
                                              spfkb/wtk[ikwtk1]))
                    norm1 = np.trapz(spfkb,interp_en)/(wtk[ikwtk1])
                    print("the normalization of the spectral function is",norm1)
                    if abs(1-norm1)>0.01:
                        print("WARNING: the renormalization of ik,ib is too bad!\n"+\
                              "Increase the time size to converge better.",ikeff,ibeff)
    
        else: 
            print( " spin down channel, k point = %02d " % (ikeff))

            ikeff = int(ik/2 + 1)
            ikwtk2 = int(ik/2)
            for ib in bdrange:
                ibeff = ib + bdgw_min
                print(" ik, ib:",ikeff, ibeff)
                eqp_kb = eqp[ik,ib]
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
                        if NewEn_max < en[-1] and NewEn_max+Elda_kb < en[-1]:
                            Done_max = True
                    if metal_valence == 1 and -Elda_kb < en[-1]:
                        NewEn_max = -Elda_kb #-0.005
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
                        print("""Zero is not in the intergration of
                              ImSigma(\omega), please check invar_den""")

                        sys.exit(0)
                    ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                    ShiftIms = interpims(ShiftEn)
                    ShiftIms_0 = interpims(NewEn_0+Elda_kb)
                    with open("ShiftIms_toc11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-down"+".dat", 'w') as f:
                        writer = csv.writer(f, delimiter = '\t')
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
                        gw_list.append(0.5*wtk[ikwtk2]/np.pi*cwIm)

                    print("IFFT done .....")
                    interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                    interp_en = newen_toc

                    spfkb = interp_toc(interp_en)
                    toc_tot_down += spfkb
                    with open("TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-down"+".dat", 'w') as f:
                        writer = csv.writer(f, delimiter = '\t')
                        writer.writerows(zip (interp_en-gwfermi,
                                              spfkb/wtk[ikwtk2]))
                    norm2 = np.trapz(spfkb,interp_en)/(wtk[ikwtk2])
                    print("the normalization of the spectral function is",norm2)
                    if abs(1-norm2)>0.01:
                        print("WARNING: the renormalization of ik,ib is too bad!\n"+\
                              "Increase the time size to converge better.",ikeff,ibeff)
                    outfile.write("%12.8e %12.8e \n" %( norm1, norm2))

    outfile.close()
    return interp_en-gwfermi, toc_tot_up, toc_tot_down

