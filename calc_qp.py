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

