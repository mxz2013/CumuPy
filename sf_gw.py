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
            
            print("SKYDEBUT hartree", hartree[ik,ib])
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



