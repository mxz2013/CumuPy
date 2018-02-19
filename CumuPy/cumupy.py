#!/usr/bin/env python
from __future__ import print_function
from outread_modules import *
import numpy as np;
import matplotlib.pylab as plt;
plt.figure(1)
import sys
import csv
from os.path import isfile, join, isdir
from os import getcwd, pardir, mkdir, chdir
import time
### ============================= ###
###  ==  PROGRAM BEGINS HERE  ==  ###
### ============================= ###
start_time = time.time()
# ======== READING INPUT VARIABLES ======= #
print(" Reading invar file... ")
invar = {}
if isfile("invar.in"):
    infile = open("invar.in")
    for line in infile.readlines():
        word = line.split()
        invar[word[-1]] = word[0]
    infile.close()
    if 'sigmafile' in invar:  ## The name of the self-energy file.
        sigfilename = invar['sigmafile'];
    else:
        sigfilename = 'default_SIG';

    print ("name of the sigma file:",  sigfilename   )
    if 'outname' in invar:  ## using wtk.dat or not 
        outname = str(invar['outname']);
    else:
        outname = 'Spfunctions';
    print ("name of the output folder:",  outname   )

    if 'wtk' in invar:  ## using wtk.dat or not 
        flag_wtk = int(invar['wtk']);
    else:
        flag_wtk = 1;
        print ("including weight of the k points or not:", flag_wtk)

    if 'extrinsic' in invar:  ## calculate extrinsic spf 
        extrinsic = int(invar['extrinsic']);
    else:
        extrinsic = 0;
        print ("extrinsic and interference or not:", extrinsic)

    if 'pjt' in invar:  ## projections 
        flag_pjt = int(invar['pjt']);
    else:
        flag_pjt = 0;

    print ("include projections or not:", flag_pjt)

    if 'Ephoton' in invar:  ## photon-energy 
        Ephoton = float(invar['Ephoton']);
    else:
        Ephoton = 0.0;

    print ("photon energy (only if extrinsic is on or projections are provided):", Ephoton)
    if 'scgw' in invar:  #one-shot G0W0 or scGW self-energy
        scgw = int(invar['scgw']); 
    else:
        scgw = 1;
    print ("scgw or not:", scgw)

    if 'nspin' in invar: ## spin-polarized or not
        nspin = int(invar['nspin']);
    else:
        nspin = 1;
    print ("number of spin chanel (1 or 2):", nspin)

    if 'Eplasmon' in invar: # an estimation of the plasmon energy  
                            # normaly taken to be the clasical plasmon energy
                            # at q=0, i.e., wp=sqrt(4\pi n) where n is the
                            # valence electron density.
        Eplasmon = int(invar['Eplasmon']) #range for C(t)
    else:
        Eplasmon = 20
    print ("The classical plasmon energy is about: ", Eplasmon)
    if 'minband' in invar: #the first band to be calculated
        minband = int(invar['minband'])
    else:
        minband=1
    
    if 'maxband' in invar:  # the last band to be calculated
        maxband = int(invar['maxband'])
    else:
        maxband = 1
    print ("the first and last band to be calculated :", minband,maxband)

    if 'bdgw_min' in invar: # the label of the first band in SIG file.
        bdgw_min = int(invar['bdgw_min']) # consistent with input of abinit:bdgw 
    else:
    	bdgw_min = 1
    if 'bdgw_max' in invar:  # the label of the first band in SIG file.
        bdgw_max = int(invar['bdgw_max']) # consistent with input of abinit:bdgw
    else:
    	bdgw_max = 1
    print ("the first and last band in GW calculation :", bdgw_min, bdgw_max)
    print ("""Note that minband cannot be smaller than bdga_min, and maxband
           cannot be bigger than bdgw_max""")

    if 'minkpt' in invar: # the first k to be calculated
        minkpt = int(invar['minkpt'])
    else:
    	minkpt = 1
    if 'maxkpt' in invar:
    	maxkpt = int(invar['maxkpt'])
    else:
    	maxkpt = 1  # the last k to be calculated

    if 'nkpt' in invar:
    	nkpt = int(invar['nkpt'])*nspin
    else:
    	nkpt = maxkpt - minkpt + 1

    print ("total k number, mink, and maxk to be calculated are :",  nkpt,
           minkpt, maxkpt)

    if 'enmin' in invar: # the minimum \omega in A(\omega)
    	enmin = float(invar['enmin'])
    else:
    	enmin = -20.0
    if 'enmax' in invar: # the maximum \omega in A(\omega)
    	enmax = float(invar['enmax'])
    else:
    	enmax = 0.0 
    print("the minimum and maxmum frequency will be calculated in cumulant :",enmin, enmax)
    
    if 'wps' in invar:  # data from Josh used for surface plasmon calculation.  
        wps = int(invar['wps']);
    else:
        wps = 0

    print ("surface plasmons (1 yes and 0 no)", wps)

    if 'calc_gw' in invar:  #enable GW calculation
    	flag_calc_gw = int(invar['calc_gw'])
    else:
    	flag_calc_gw = 0
    if 'spf_qp' in invar:  #enable QP calculation
    	spf_qp = int(invar['spf_qp'])
    else:
    	spf_qp = 0
    if 'calc_toc96' in invar: #enable TOC96 calculation
    	flag_calc_toc96 = int(invar['calc_toc96'])
    else:
    	flag_calc_toc96 = 0
    if 'calc_toc11' in invar: #enable TOC11 calculation
    	flag_calc_toc11 = int(invar['calc_toc11'])
    else:
    	flag_calc_toc11 = 0
    if 'calc_toc_original' in invar: 
    	flag_calc_toc = int(invar['calc_toc_original'])
    else:
    	flag_calc_toc = 0
    if 'calc_rc' in invar: # enable retarded cumulant calculation
    	flag_calc_rc = int(invar['calc_rc'])
    else:
    	flag_calc_rc = 0
    if 'rc_Josh' in invar: # enable retarded cumulant calculation
    	rc_Josh = int(invar['rc_Josh'])
    else:
    	rc_Josh = 0
    if 'calc_crc' in invar: #enable CRC so as TOC96 calculation
    	flag_calc_crc = int(invar['calc_crc']) # CRC implementation is not
    else:
    	flag_calc_crc = 0
    print ("Spectral functions will be calculated are: ")
    print ("GW", flag_calc_gw)
    print ("QP", spf_qp)
    print ("TOC11", flag_calc_toc11)
    print ("TOC original", flag_calc_toc)
    print ("RC", flag_calc_rc)
    print ("RC of J. Kas (not recommended)", rc_Josh)
    print ("TOC96 (not recommended)", flag_calc_toc96)
    print ("CRC (not ready yet)", flag_calc_crc)

    if 'gwfermi' in invar: #Fermi enegy after GW calculation
        gwfermi = float(invar['gwfermi']);
    else:
        gwfermi = 0.0
    if 'lda_fermi' in invar: # Fermi energy after LDA calculation
        lda_fermi = float(invar['lda_fermi']); # will be used
    else:                                    #when scgw=0
        lda_fermi = 0.0

    print ("The LDA and GW fermi energies are :", lda_fermi, gwfermi)

    if 'invar_den' in invar: #d\omega in the cumulant A(\omega) 
    	invar_den = float(invar['invar_den']) # for the moment the choices 
    else:                                 # are 0.05, 0.025, 0.0125, 0.01,0.005
    	invar_den = 0.05
    if 'invar_eta' in invar: #lorentzian broadening of all cumulant A(\omega)
        invar_eta = float(invar['invar_eta'])
    else:
        invar_eta = 0.1
        
    print ("the energy resolution and Lorentzian broadening are :", invar_den, invar_eta)
    print (""""Note that invar_den must be chosen from 0.05, 0.025, 0.0125,
           0.01, 0.005, and invar_eta must be equal or bigger than invare_den
           for convergence""")
    if 'metal_valence' in invar: # TOC for metal valence is implemented
        metal_valence = float(invar['metal_valence']) #different with core
    else:
        metal_valence = 0
    if 'core_only' in invar: # TOC for metal valence is implemented
        core = float(invar['core_only']) #different with core
    else:
        core = 0
    #if 'Fermi_temp' in invar: # Temperature in Fermi-function Kelvin 
    #    Temp = float(invar['Fermi_temp'])
    #else:
    #    Temp = 300.0
    #if 'Gaussian' in invar: #lorentzian broadening of all cumulant A(\omega)
    #    gbro = float(invar['Gaussian'])
    #else:
    #    gbro = 0.1
    if 'npoles' in invar: #lorentzian broadening of all cumulant A(\omega)
        npoles = int(invar['npoles'])
    else:
        npoles = int(1)

    if 'FFTtsize' in invar: #the number of time steps used in FFT
        FFTtsize = int(invar['FFTtsize'])
        if FFTtsize % 2 != 0:
            FFTtsize = int(FFTtsize+1)
    else:
        FFTtsize = int(5000)
    print ("the time step for FFT (need to be converged!) :", FFTtsize)    
    if 'abinit_eqp' in invar: # use eqp from abinit or recalculated
    	abinit_eqp = int(invar['abinit_eqp']) #in this code.
    else:
    	abinit_eqp = 0 
    print("input GW QP energy or not: :", abinit_eqp)

    print (""" if aninit_eqp = 1, the code will not calculate the QP energy
           using w-e-ReSigma(w)=0, but read directly the input file
           eqp_abinit.dat.
          """)
else :
    print ("Invar file not found (invar.in). Impossible to continue.")
    sys.exit(1)
print ("Reading invar done.")

#npoles = int(150)  #for sampling Im\Sigma lesser to calculate crc_unocc
nband = bdgw_max - bdgw_min + 1
#hartree, hartree_ks = read_hartree()
hartree= read_hartree()
#with open("hartree_gw.dat", 'w') as f:
#    writer = csv.writer(f, delimiter = '\t')
#    writer.writerows(zip (hartree))
#with open("hartree_ks.dat", 'w') as f:
#    writer = csv.writer(f, delimiter = '\t')
#    writer.writerows(zip (hartree_ks))
Sigx = read_hf()
ehf = hartree + Sigx


if scgw == 0:
    print("""
          WARNING:: You are using the one-shot GW self-energy
          to calculate cumulant so you have to give
          fermi energy of E0, e.g., LDA fermi enegy.
          """)
    Elda = read_lda()
    Elda = Elda
else:
    Elda = np.zeros((nkpt,nband))
#import csv
#with open("Elda_fermi.dat", 'w') as f:
#    writer = csv.writer(f, delimiter = '\t')
#    writer.writerows(zip(Elda))
if flag_wtk == 1:
    wtk = read_wtk()
else:
    print("""
          WARNING: Weight of k points are neglected!
         """)
    wtk = [1]*nkpt

if wps == 1:
    wps1, wps2 = read_wps()
else:
    print(""" WARNING:
          Surface plasmons are not calculated!
         """)
    wps1 = 0.
    wps2 = 0.

if flag_pjt == 1:
    pjt1, pjt2, pjt3 =read_pjt_new(nkpt,nband,bdgw_min,nspin) #  read_pjt()
    #pjt1, pjt2, pjt3 =read_pjt_new(145,8,1,nspin) # modified by SKY on 26
    #December 2018 for calculating aluminum core 
    cs1, cs2, cs3 = read_cs(Ephoton)
else:
    print("""
          WARNING:  no projections of s, p, or d!
         """)
    pjt1 = np.zeros((nkpt,nband))
    pjt2 = np.zeros((nkpt,nband))
    pjt3 = np.zeros((nkpt,nband))
    cs1 = 0.
    cs2 = 0.
    cs3 = 0.

with open("pjt1.dat", 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerows(zip (pjt1))
with open("pjt2.dat", 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerows(zip (pjt2))
with open("pjt3.dat", 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerows(zip (pjt3))

#sys.exit(1)

en, res, ims = read_sigfile(sigfilename, nkpt, bdgw_min, bdgw_max)

if extrinsic == 1:
    Rx, Ry = read_R(lda_fermi,gwfermi,scgw,Ephoton)
else:
    Rx = 1
    Ry = 1
    #Rx,res, Ry = read_sigfile(sigfilename, nkpt, bdgw_min, bdgw_max) ##SKYDEBUG

#with open("R3.9313.dat", 'w') as f:
#    writer = csv.writer(f, delimiter = '\t')
#    writer.writerows(zip (Rx, Ry))

bdrange = xrange(minband - bdgw_min, maxband - bdgw_min + 1)
kptrange = xrange(minkpt - 1, maxkpt*nspin)

### ===================================================== ###
#give the option of using abinit QP energies of the QP energies 
## recalculate from cumulant code by finding zero crossing
## of the shifted Re\Sigma

#newdirname = "QP_SPF"
#origdir = getcwd() # remember where we are
#newdir = join(origdir, newdirname) # Complete path of the new directory
#print(" Moving into output directory:\n ", newdir)
#if not isdir(newdir) :
#    mkdir(newdir)
################# the Fermi function
def fermi_function(x, Temp):
    Bolz = 1.3806485279e-23 # J/K.
    mu = 0.0
    Joule2eV = 6.241506363094e18
    f = 1./(1.+ np.exp((x-mu)/(Bolz*Temp*Joule2eV)))
    return f
###################################
### ===================================================== ###
print(" # ------------------------------------------------ # ")
# Here we move to a subdirectory to avoid flooding-up the current directory
newdirname = outname
#newdirname = "Spfunctions"
origdir = getcwd() # remember where we are
newdir = join(origdir, newdirname) # Complete path of the new directory
print(" Moving into output directory:\n ", newdir)
if not isdir(newdir) :
    mkdir(newdir)
chdir(newdir)

if abinit_eqp == 1:   
    print("""
          WARNING: you choose to provide the GW QP energy
          as imput. Please provide the file named "eqp_abinit.dat"
          in the good format.
         """)
    eqp_abinit = read_eqp_abinit()
    eqp = eqp_abinit
    #with open("eqp_abinit.dat", 'w') as f:
    #    writer = csv.writer(f, delimiter = '\t')
    #    writer.writerows(zip (eqp-gwfermi))
else:

    print("""
          WARNING: you choose to calculate the GW QP energy from 
          this code by finding the solution of 
          "w-e_{ks}+V_{xc}-Re\Sigma_{xc}(w)=0". Cross check is 
          recommended with the GW QP energy from abinit or other 
          code.
         """)
    from calc_qp import *
    eqp, imeqp = calc_eqp_imeqp(nspin,spf_qp,wtk,bdrange,kptrange,bdgw_min, en, enmin, enmax, res, ims,
                                hartree, gwfermi, nkpt, nband, scgw,
                                Elda,pjt1,pjt2,pjt3,cs1,cs2,cs3)

### ================================= ###
### ===== GW SPECTRAL FUNCTION ====== ###
# GW spectral function part
t_pregw = time.time() 
if flag_calc_gw == 1:
    from sf_gw import * 
    print(" # ------------------------------------------------ # ")
    print("Calculate GW begins ")
    if nspin == 2:
        from sf_modules_spin import calc_spf_gw_spin
        newen, spftot_up,up1,up2,up3, spftot_down, d1,d2,d3 = calc_spf_gw_spin(pjt1,pjt2,pjt3,bdrange, kptrange,
                                                         bdgw_min, wtk, en, enmin, enmax,
                                                         res,ims, hartree, gwfermi)
       # print(" ### Writing out A(\omega)_GW...  ")
        with open("spftot_gw.dat",'w') as f:
             writer = csv.writer(f, delimiter = '\t')
             writer.writerow(['# w-fermi','# spftot_up','# spftot_up_s','# spftot_up_p','# spftot_up_d','# spftot_down','# spftot_down_s','# spftot_down_p','# spftot_down_d'])
             writer.writerows(zip( newen, spftot_up, up1,up2,up3, spftot_down, d1,d2,d3))
        plt.plot(newen,spftot_up,label="ftot_gw_SpinUp")
        plt.plot(newen,spftot_down,label="ftot_gw_SpinDown")


    elif nspin == 1:
        newen, spftot, spftot_pjt1, spftot_pjt2, spftot_pjt3,sumkbp = calc_spf_gw(cs1,cs2,cs3,pjt1,pjt2,pjt3,bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res, ims, hartree, gwfermi, invar_eta)
        with open("spftot_gw.dat",'w') as f:
             writer = csv.writer(f, delimiter = '\t')
             writer.writerow(['# w-fermi','# spftot','# spftot_s','#spftot_p','# spftot_d','sumkbp'])
             writer.writerows(zip( newen, spftot, spftot_pjt1, spftot_pjt2,
                                  spftot_pjt3,sumkbp))

        plt.plot(newen,spftot,label="spf-gw");
#######################################################################################

if flag_calc_toc11 == 1:
    if nspin == 2: 
        from sf_modules_spin import calc_toc11_spin
        print("# ------------------------------------------------ #")
        print( "Calculating spin polarized TOC11 begins")
        interp_en, up, up1, up2, up3, down, d1, d2, d3 =  calc_toc11_spin(pjt1,
                                                                          pjt2,
                                                                          pjt3,
                                                                          gwfermi,
                                                                          lda_fermi,
                                                                          bdrange,
                                                                          bdgw_min,
                                                                          kptrange,
                                                                          FFTtsize,
                                                                          en,
                                                                          enmin,
                                                                          enmax,
                                                                          eqp,
                                                                          Elda,
                                                                          scgw,
                                                                          Eplasmon,
                                                                          ims, invar_den, invar_eta, wtk, metal_valence)
        

        with open("spftot_toc11.dat",'w') as f:
             writer = csv.writer(f, delimiter = '\t')
             writer.writerow(['# w-fermi','# spftot_up','# spftot_up_s','# spftot_up_p','# spftot_up_d','# spftot_down','# spftot_down_s','# spftot_down_p','# spftot_down_d'])
             writer.writerows(zip( newen, up, up1,up2,up3, down, d1,d2,d3))
       # print(" ### Writing out A(\omega)_TOC11...  ")

        plt.plot(interp_en,up,label="ftot_toc11_SpinUp");
        plt.plot(interp_en,down,label="ftot_toc11_SpinDown");

    else:
        from sf_toc11 import *
        print("# ------------------------------------------------ #")
        print( "Calculating TOC11 begins")
        en_toc11, toc11_tot, tot_s,tot_p,tot_d,sumkbp = calc_toc11(wps1,wps2,gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence,nkpt,nband,Rx, Ry,
                   extrinsic,pjt1,pjt2,pjt3,cs1,cs2,cs3)
        
        
        with open("spftot_toc11"+"-ext"+str(extrinsic)+".dat",'w') as f:
             writer = csv.writer(f, delimiter = '\t')
             writer.writerow(['# w-fermi','# spftot','# spftot_s','#spftot_p','# spftot_d','sumkbp'])
             writer.writerows(zip( en_toc11, toc11_tot, tot_s, tot_p,tot_d,
                                  sumkbp))
        plt.plot(en_toc11, toc11_tot,label="spf-toc11");

##################################################################################
if flag_calc_crc == 1:
    print("# ------------------------------------------------ #")
    print("Calulating CRC begines::")
    #omegampole, ampole = calc_multipole (scgw,Elda,lda_fermi,nkpt,
    #                                     nband,gwfermi,npoles, ims, kptrange,
    #                                     bdrange,bdgw_min, eqp, en, enmin,
    #                                     enmax)
    en_crc, toc96tot, crc_tot =  calc_toc96_crc (gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence, imeqp,nkpt,nband,
                npoles)
    
    #calc_crc(invar_eta,gwfermi, wtk, kptrange, bdrange, bdgw_min, omegampole, ampole, npoles,
    #                      beta_greater, en_toc96, toc96_tot, imeqp, eqp)
    outname = "spftot_toc96"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(en_crc)):
        outfile.write("%8.4f %12.8e \n" % (en_crc[i], toc96tot[i]))
    outfile.close()
    outname = "spftot_crc"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(en_crc)):
        outfile.write("%8.4f %12.8e \n" % (en_crc[i], crc_tot[i]))
    outfile.close()
    plt.plot(en_crc,crc_tot,label="ftot_crc");
    plt.plot(en_crc,toc96tot,label="ftot_toc96");

if flag_calc_rc == 1:
    from sf_toc11 import *
    from sf_rc import *
    print("# ------------------------------------------------ #")
    print ("Calculating RC begins")
    en_rc, spftot,tots,totp,totd,sumkbp = calc_rc (wps1,wps2,gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta,wtk,nkpt,nband,Rx, Ry, extrinsic,
                             core,Eplasmon,pjt1,pjt2,pjt3,cs1,cs2,cs3) 
    print (" ### Writing out A(\omega)_rc...  ")

    with open("spftot_rc"+"-ext"+str(extrinsic)+".dat",'w') as f:
         writer = csv.writer(f, delimiter = '\t')
         writer.writerow(['# w-fermi','# spftot','# spftot_s','# spftot_p','#spftot_d','sum_kbp'])
         writer.writerows(zip( en_rc, spftot, tots,totp,totd,sumkbp))
    print (" A(\omega)_rc written in", outname)

    plt.plot( en_rc, spftot,label="spf-rc");
#######################################################################################
### this module calculates the RC of Josh
if rc_Josh == 1:
    from sf_toc11 import *
    from sf_rc import *
    print("# ------------------------------------------------ #")
    print ("Calculating RC of Josh begins")
    en_rc, spftot,tots,totp,totd,sumkbp = calc_rc_Josh (ehf,wps1,wps2,gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta,wtk,nkpt,nband,Rx, Ry, extrinsic,
                             core,Eplasmon,pjt1,pjt2,pjt3,cs1,cs2,cs3) 

    with open("spftot_rc_Josh"+"-ext"+str(extrinsic)+".dat",'w') as f:
         writer = csv.writer(f, delimiter = '\t')
         writer.writerow(['# w-fermi','# spftot','# spftot_s','# spftot_p','#spftot_d','sum_kbp'])
         writer.writerows(zip( en_rc, spftot, tots,totp,totd,sumkbp))
    print (" A(\omega)_rc written in", outname)

    plt.plot( en_rc, spftot,label="spf-rc-Josh");
if flag_calc_toc == 1:
    from sf_toc11 import *
    print("# ------------------------------------------------ #")
    print( "Calculating TOC oringial begins")
    en_toc11, toc11_tot, tot_s,tot_p,tot_d,sumkbp = calc_toc(ehf,wps1,wps2,gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                eqp, Elda, scgw, Eplasmon, ims, invar_den,
                invar_eta, wtk, metal_valence,nkpt,nband,Rx, Ry,
               extrinsic,pjt1,pjt2,pjt3,cs1,cs2,cs3)
    
    
    with open("spftot_toc11"+"-ext"+str(extrinsic)+".dat",'w') as f:
         writer = csv.writer(f, delimiter = '\t')
         writer.writerow(['# w-fermi','# spftot','# spftot_s','#spftot_p','# spftot_d','sumkbp'])
         writer.writerows(zip( en_toc11, toc11_tot, tot_s, tot_p,tot_d,
                              sumkbp))

    plt.plot(en_toc11, toc11_tot,label="spf-toc");
#print ("Moving back to parent directory:\n", origdir)
#chdir(newdir)

print ("Moving back to parent directory:\n", origdir)
chdir(newdir)
end_time = time.time()
print()

t2 = t_pregw - start_time
t3 = end_time - start_time
#title = 'Spectral function '+ 'A (' + r'$\omega $ ' + ') - '+r'$ h\nu = $'+str(penergy)+' eV'
#plt.title(title)
plt.legend(loc=2);
plt.savefig('spftot.eps', format='eps', dpi=1000)
plt.show();

t4 = t3
t5 = t3
if int(t3/3600) >= 1: 
    t4 = t3 - int(t3/3600)*3600
if int(t4/60) >= 1: 
    t5 = t4 - int(t4/60)*60
print(" Calculation lasted "+str(int(t3/3600))+" Hours, "+str(int(t4/60))+" Minutes and "+str(int(t5))+" Seconds")
print(" "+30*"-")

print("# ------------------------------------------------ #")
print("Suggested references for the acknowledgment of CumuPy usage.")
print("""
      [1]Dynamical effects in electron spectroscopy. Jianqiang Sky Zhou, JJ
      Kas, Lorenzo Sponza, Igor Reshetnyak, Matteo Guzzo, Christine Giorgetti,
      Matteo Gatti, Francesco Sottile, JJ Rehr, Lucia Reining,
      The Journal of Chemical Physics 143, 184109 (2015).

      [2] Cumulant Green's function calculations of plasmon satellites in bulk
      sodium: Influence of screening and the crystal environment, Jianqiang Sky
      Zhou, Matteo Gatti, JJ Kas, JJ Rehr, Lucia Reining, Phys. Rev. B 97,
      035137 (2018).
      """)
print("# ------------------------------------------------ #")

print(" End of program reached.")
