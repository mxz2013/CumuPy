#!/usr/bin/env python
"""
Two retarded cumulant modeules. "calc_rc" is the main 
RC moudle in [my prb]. "calc_rc_Josh" is the used to calculate RC 
of J. Kas in Phys. Rev. B 90, 085112 (2014), where the
Hartree Fock enery is also needed.
"""
from __future__ import print_function
import numpy as np;
from multipole import *
from sf_toc11 import *
import matplotlib.pylab as plt;
from scipy.interpolate import interp1d
import sys
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir



def calc_rc (wps1,wps2,gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta, wtk,nkpt,nband,
            Rx, Ry, extrinsic, core ,Eplasmon,pjt1,pjt2,pjt3,cs1,cs2,cs3):
    import numpy as np
    import pyfftw
    import csv
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    print("calc_rc : :")
    metal_valence = 0 # no meaning but needed to call
                    #  calc_ShiftImSig
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
    trange, dtfft,denfft, fften_min  = prep_FFT(invar_den, fftsize)
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
            
            ims_tmp=ims[ik,ib] 

            NewEn, ShiftIms = calc_ShiftImSig(en, ims_tmp, ikeff, ibeff,
                                              Elda_kb, xfermi, Eplasmon,
                                              metal_valence, invar_den, Rx,
                                              Ry,wps1, wps2, extrinsic, rc_toc = 1)

            gt_list = calc_integ_Imsig(NewEn, ShiftIms, trange) 
            w_list, gw_list = calc_FFT(eqp_kb,gt_list, fftsize,dtfft, enmin, newen_rc, denfft, invar_eta)

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


