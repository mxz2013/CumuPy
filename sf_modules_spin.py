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
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir

def calc_spf_gw_spin(bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res, ims, hartree, efermi):

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
    print (" ### Interpolation and calculation of A(\omega)_GW...  ")
    spftot_up = np.zeros((np.size(newen)));
    spftot_down = np.zeros((np.size(newen)));
    # Here we interpolate re and im sigma
    # for each band and k point
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
                outnamekb="spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-spin-up"+".dat"
                outfilekb = open(outnamekb,'w')
                for ien in xrange(np.size(newen)) :
                    outfilekb.write("%8.4f %12.8e %12.8e %12.8e %12.8e\n" % (newen[ien], spfkb[ien], redenom[ien], tmpres[ien], tmpim[ien]))
                outfilekb.close()
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
                outnamekb="spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-spin-down"+".dat"
                outfilekb = open(outnamekb,'w')
                for ien in xrange(np.size(newen)) :
                    outfilekb.write("%8.4f %12.8e %12.8e %12.8e %12.8e\n" % (newen[ien], spfkb[ien], redenom[ien], tmpres[ien], tmpim[ien]))
                outfilekb.close()

    return newen, spftot_up, spftot_down

def integ_w(x , ShiftIms,NewEn,tImag): # pay attention of NewEn==0!!!!!
    return 1.0/np.pi*abs(ShiftIms[x])*(np.exp(-(NewEn[x])*tImag)-1.0)*(1.0/(NewEn[x]**2))

def calc_toc11_spin (bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax, eqp,
                encut, metal_valence, ims, invar_den, invar_eta, wtk):
    import numpy as np
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    print("calc_toc11_spin : :")
 
    ddinter = 0.005 
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc_tot_up =  np.zeros((np.size(newen_toc))) 
    toc_tot_down =  np.zeros((np.size(newen_toc))) 
    #pdos = np.array(pdos)
    fftsize = FFTtsize
    tol_ecut = encut
    outname = "Norm_check_toc11.dat"
    outfile = open(outname,'w')

    for ik in kptrange:
        if ik %2 == 0:
            ikeff = int(ik/2 + 1)
            ikwtk1 = int(ik/2)
            print( " spin up chanel, k point = %02d " % (ikeff))
            for ib in bdrange:
                ibeff = ib + bdgw_min
                print(" ik, ib:",ikeff, ibeff)
                eqp_kb = eqp[ik,ib]
                print("eqp:", eqp_kb)
                if eqp_kb <= 0:
                    Eshift = False
                    Es = 0
                    while not Eshift:
                        NewEn_max = en[-1] - eqp_kb - Es
                        Es += 1
                        if NewEn_max < en[-1] and NewEn_max + eqp_kb < en[-1]:
                            Eshift = True
                            converged = False
                            tol_area = 0.02 #invar_den
                            area_0 = -1e6
                            newdx = 0.1
                            print("converging newdx ...")
                            print(" TOlerance: ", tol_area)
                            while not converged:
                                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis
                                                     = -1)
                                NewEn_min = int(en[0]-eqp_kb)
                                if metal_valence == 1 and -eqp_kb < en[-1]:
                                    NewEn_max = -eqp_kb
                                NewEn = np.arange(NewEn_min, NewEn_max, newdx)
                                NewEn_size = NewEn.size
                                NewIms = interpims(NewEn)
                                ShiftEn = np.arange(NewEn_min + eqp_kb, NewEn_max + eqp_kb,
                                                    newdx)
                                ShiftIms = interpims(ShiftEn)
                                ct_tmp = 0
                                for i in np.arange(0,NewEn_size-1,1):
                                    if abs(NewEn[i]) < 1e-6 : #finding w=0 and then put cutoff.
                                        print("Zero is in NewEn")
                                        en1 = np.arange(0,i - tol_ecut, 1) # cut elements on the left
                                        en2 = np.arange(i + tol_ecut+1, NewEn_size - 1, 1) # cut element
                                    #on the right
                                        for j in np.concatenate((en1, en2), axis = 0):
                                            area_tmp = 0.5*newdx*(integ_w(j,ShiftIms,NewEn,-10.j)
                                                                  +integ_w(j+1,ShiftIms,NewEn,-10.j))
                                            ct_tmp+=area_tmp 
                                gt_tmp = np.exp(ct_tmp)
                                if gt_tmp == 1.0:
                                    print("Zero is not in NewEn !!!")
                                    newdx = newdx*0.5
                                    continue

                                d_area = area_0 - gt_tmp.imag
                                area_0 = gt_tmp.imag
                                newdx = newdx*0.5
                                if abs(d_area) <= tol_area:
                                    converged = True
                                    newdx = newdx/0.5
                                    print("the converged newdx is", newdx) 

                                    imeqp = interpims(eqp_kb)
                                    print("ImSigma(Eqp): {}".format(interpims(eqp_kb)))
                                    outnamekb = "ShiftIms_toc11"+"-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"up"+".dat"
                                    outfilekb = open(outnamekb,'w')
                                    for ien in xrange(NewEn_size):
                                        outfilekb.write("%8.4f %12.8e\n" % (NewEn[ien], ShiftIms[ien]))
                                    outfilekb.close()

                                    tfft_min = -2*np.pi/invar_den
                                    tfft_max = 0
                                    trange = np.linspace(tfft_min, tfft_max,fftsize)
                                    dtfft = abs(trange[-1]-trange[0])/fftsize
                                    print ("the time step is", dtfft)
                                    denfft = 2*np.pi/abs(trange[-1]-trange[0])
                                    print("the energy resolution after FFT is",
                                          denfft)
                                    fften_min = -2*np.pi/dtfft
                                    fften_max = 0
                                    enrange = np.arange(fften_min,NewEn[-1],denfft)
                                    gt_list = []
                                    Regt_list = []
                                    Imgt_list = []
                                    print("the size of fft is", fftsize)
                                    for t in trange:
                                        tImag = t*1.j
                                        ct = 0
                                        ecut_tmp = 1e-6
                                        for i in np.arange(0,NewEn_size-1,1):
                                            if abs(NewEn[i]) < ecut_tmp : #finding w=0 and then put cutoff.
                                                en1 = np.arange(0,i - tol_ecut, 1) # cut elements on the left
                                                en2 = np.arange(i + tol_ecut+1, NewEn_size - 1, 1) # cut element
                                    #on the right
                                                for j in np.concatenate((en1, en2), axis = 0):  
                                                    area = 0.5*newdx*(integ_w(j,ShiftIms,NewEn,tImag)
                                                                      +integ_w(j+1,ShiftIms,NewEn,tImag))
                                                    ct += area 

                                        gt = np.exp(ct)
                                        gt_list.append(gt)
                                        Regt_list.append(gt.real)
                                        Imgt_list.append(gt.imag)
                                    print("IFFT of ")
                                    print("kpoint = %02d" % (ikeff))
                                    print("band=%02d" % (ibeff))

                                    fft_in = pyfftw.empty_aligned(fftsize, dtype='complex128')
                                    fft_out = pyfftw.empty_aligned(fftsize, dtype='complex128')
                                    ifft_object = pyfftw.FFTW(fft_in, fft_out,
                                                      direction='FFTW_BACKWARD',threads=1)
                                    cw = ifft_object(gt_list)*(fftsize*dtfft)

                                    freq = fftfreq(fftsize,dtfft)*2*np.pi
                                    s_freq = fftshift(freq) # To have the correct energies (hopefully!)
                                    s_go = fftshift(cw)

                                    eta = 1.j*invar_eta # the eta in the theta function that can be changed when the satellite is very close to
                                                 #the QP.
                                    gw_list = []
                                    w_list = np.arange(enmin,newen_toc[-1]+denfft,denfft)
                                    for w in w_list:
                                        c = 0
                                        for i in xrange(fftsize-1):
                                            Area2 = 0.5*denfft*(s_go[i]/(w-eqp_kb-s_freq[i]-eta)
                                                                +s_go[i+1]/(w-eqp_kb-s_freq[i+1]-eta))
                                            c+=Area2
                                        cwIm = 1./np.pi*c.imag
                                        gw_list.append(0.5*wtk[ikwtk1]/np.pi*cwIm)

                                    print("IFFT done .....")
                                    interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                                    interp_en = newen_toc

                                    spfkb = interp_toc(interp_en)
                                    toc_tot_up += spfkb
                                    outnamekb ="TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-up"+".dat"
                                    outfilekb = open(outnamekb,'w')
                                    en_toc11 = []
                                    for i in xrange(len(interp_en)):
                                        en_toc11.append(interp_en[i])
                                        outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
                                    outfilekb.close()
                                    norm1 = np.trapz(spfkb,interp_en)/(wtk[ikwtk1])
                                    print("check the renormalization : :")
                                    print()
                                    print("the normalization of the spectral function is",norm1)
                                    if norm1<=0.9:
                                        print(""" WARNING: the renormalization is too bad, you need to
                                              converge your spf using other input variables """)
    
        else: 
            ikeff = int(ik/2 + 1)
            ikwtk2 = int(ik/2)
            print( " spin down channel, k point = %02d " % (ikeff))
            for ib in bdrange:
                ibeff = ib + bdgw_min
                print(" ik, ib:",ikeff, ibeff)
                eqp_kb = eqp[ik,ib]
                print("eqp:", eqp_kb)
                if eqp_kb <= 0:
                    Eshift = False
                    Es = 0
                    while not Eshift:
                        NewEn_max = en[-1] - eqp_kb - Es
                        Es += 1
                        if NewEn_max < en[-1] and NewEn_max + eqp_kb < en[-1]:
                            Eshift = True
                            converged = False
                            tol_area = 0.02
                            area_0 = -1e6
                            newdx = 0.1
                            print("converging newdx ...")
                            print(" TOlerance: ", tol_area)
                            while not converged:
                                interpims = interp1d(en, ims[ik,ib], kind = 'linear', axis
                                                     = -1)
                                NewEn_min = int(en[0]-eqp_kb)
                                if metal_valence == 1 and -eqp_kb < en[-1]:
                                    NewEn_max = -eqp_kb
                                NewEn = np.arange(NewEn_min, NewEn_max, newdx)
                                NewEn_size = NewEn.size
                                NewIms = interpims(NewEn)
                                ShiftEn = np.arange(NewEn_min + eqp_kb, NewEn_max + eqp_kb,
                                                    newdx)
                                ShiftIms = interpims(ShiftEn)
                                ct_tmp = 0
                                for i in np.arange(0,NewEn_size-1,1):
                                    if abs(NewEn[i]) < 1e-6 : #finding w=0 and then put cutoff.
                                        print("Zero is in NewEn, GOOD!")
                                        en1 = np.arange(0,i - tol_ecut, 1) # cut elements on the left
                                        en2 = np.arange(i + tol_ecut+1, NewEn_size - 1, 1) # cut element
                                    #on the right
                                        for j in np.concatenate((en1, en2), axis = 0):
                                            area_tmp = 0.5*newdx*(integ_w(j,ShiftIms,NewEn,-10.j)
                                                                  +integ_w(j+1,ShiftIms,NewEn,-10.j))
                                            ct_tmp += area_tmp 
                                gt_tmp = np.exp(ct_tmp)
                                if gt_tmp == 1.0:
                                    print("Zero is not in NewEn !!!")
                                    newdx = newdx*0.5
                                    continue

                                d_area = area_0 - gt_tmp.imag
                                area_0 = gt_tmp.imag
                                newdx = newdx*0.5
                                if abs(d_area) <= tol_area:
                                    converged = True
                                    newdx = newdx/0.5
                                    print("the converged newdx is", newdx) 

                                    imeqp = interpims(eqp_kb)
                                    print("ImSigma(Eqp): {}".format(interpims(eqp_kb)))
                                    outnamekb ="ShiftIms_toc11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-down"+".dat"
                                    outfilekb = open(outnamekb,'w')
                                    for ien in xrange(NewEn_size):
                                        outfilekb.write("%8.4f %12.8e\n" % (NewEn[ien], ShiftIms[ien]))
                                    outfilekb.close()

                                    tfft_min = -2*np.pi/invar_den
                                    tfft_max = 0
                                    trange = np.linspace(tfft_min, tfft_max,fftsize)
                                    dtfft = abs(trange[-1]-trange[0])/fftsize
                                    print ("the time step is", dtfft)
                                    denfft = 2*np.pi/abs(trange[-1]-trange[0])
                                    print("the energy resolution after FFT is",denfft)
                                    fften_min = -2*np.pi/dtfft
                                    fften_max = 0
                                    enrange = np.arange(fften_min,NewEn[-1],denfft)
                                    gt_list = []
                                    Regt_list = []
                                    Imgt_list = []
                                    print("the size of fft is", fftsize)
                                    for t in trange:
                                        tImag = t*1.j
                                        ct = 0
                                        ecut_tmp = 1e-6
                                        for i in np.arange(0,NewEn_size-1,1):
                                            if abs(NewEn[i]) < (ecut_tmp) :
                                                en1 = np.arange(0,i - tol_ecut, 1)
                                                en2 = np.arange(i + tol_ecut+1, NewEn_size - 1, 1)
                                                for j in np.concatenate((en1, en2), axis = 0):  
                                                    area = 0.5*newdx*(integ_w(j,ShiftIms,NewEn,tImag)
                                                                      + integ_w(j+1,ShiftIms,NewEn,tImag))
                                                    ct += area 

                                        gt = np.exp(ct)
                                        gt_list.append(gt)
                                        Regt_list.append(gt.real)
                                        Imgt_list.append(gt.imag)
                                    print("IFFT of ")
                                    print("kpoint = %02d" % (ikeff))
                                    print("band=%02d" % (ibeff))

                                    fft_in = pyfftw.empty_aligned(fftsize, dtype='complex128')
                                    fft_out = pyfftw.empty_aligned(fftsize, dtype='complex128')
                                    ifft_object = pyfftw.FFTW(fft_in, fft_out,
                                                      direction='FFTW_BACKWARD',threads=1)
                                    cw=ifft_object(gt_list)*(fftsize*dtfft)

                                    freq = fftfreq(fftsize,dtfft)*2*np.pi
                                    s_freq = fftshift(freq)
                                    s_go = fftshift(cw)

                                    eta = 1.j*invar_eta
                                    gw_list = []
                                    w_list = np.arange(enmin,newen_toc[-1]+denfft,denfft)
                                    for w in w_list:
                                        c = 0
                                        for i in xrange(fftsize-1):
                                            Area2 = 0.5*denfft*(s_go[i]/(w-eqp_kb-s_freq[i]-eta)
                                                               + s_go[i+1]/(w-eqp_kb-s_freq[i+1]-eta))
                                            c+=Area2
                                        cwIm = 1./np.pi*c.imag
                                        gw_list.append(0.5*wtk[ikwtk2]/np.pi*cwIm)

                                    print("IFFT done .....")
                                    interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                                    interp_en = newen_toc

                                    spfkb = interp_toc(interp_en)
                                    toc_tot_down += spfkb
                                    #spfkb = gw_list
                                    #toc_tot = [sum(i) for i in zip(toc_tot,gw_list)]
                                    outnamekb="TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+"-down"+".dat"
                                    outfilekb = open(outnamekb,'w')
                                    en_toc11 = []
                                    for i in xrange(len(interp_en)):
                                        en_toc11.append(interp_en[i])
                                        outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
                                    outfilekb.close()
                                    norm2 = np.trapz(spfkb,interp_en)/(wtk[ikwtk2])
                                    print("check the renormalization : :")
                                    print()
                                    print("the normalization of the spectral function is",norm2)
                                    if norm2 <= 0.9:
                                        print(""" WARNING: the renormalization is too bad, you need to
                                              converge your spf using other input variables """)
                                    outfile.write("%8.4f %12.8e %12.8e \n" %(newdx, norm1, norm2))
    outfile.close()
    return en_toc11, toc_tot_up, toc_tot_down

