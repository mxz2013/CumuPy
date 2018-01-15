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
########### delete the rest from here ##################
