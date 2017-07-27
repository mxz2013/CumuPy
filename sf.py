#!/usr/bin/env python
from __future__ import print_function
from sf_modules import *
from outread_modules import *
from sf_crc_modules import *
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
        #print "invar:", invar
    infile.close()
    if 'sigmafile' in invar:  ## The name of the self-energy file.
        sigfilename = invar['sigmafile'];
    else:
        sigfilename = 'default_SIG';

    if 'outname' in invar:  ## using wtk.dat or not 
        outname = str(invar['outname']);
    else:
        outname = 'Spfunctions';

    if 'rs_heg' in invar:  ## using wtk.dat or not 
        rs = float(invar['rs_heg']);
    else:
        rs = 0;
    if 'wtk' in invar:  ## using wtk.dat or not 
        flag_wtk = int(invar['wtk']);
    else:
        flag_wtk = 1;
    if 'core_rc' in invar:  ##  
        core = int(invar['core_rc']);
    else:
        core = 0;
    if 'extrinsic' in invar:  ## calculate extrinsic spf 
        extrinsic = int(invar['extrinsic']);
    else:
        extrinsic = 0;
    if 'background' in invar:  ## using wtk.dat or not 
        bg = int(invar['background']);
    else:
        bg = 0;

    if 'pjt' in invar:  ## projections 
        flag_pjt = int(invar['pjt']);
    else:
        flag_pjt = 0;
    if 'Ephoton' in invar:  ## photon-energy 
        Ephoton = float(invar['Ephoton']);
    else:
        Ephoton = 0.0;
    if 'scgw' in invar:  #one-shot G0W0 or scGW self-energy
        scgw = int(invar['scgw']); 
    else:
        scgw = 1;
    if 'nspin' in invar: ## spin-polarized or not
        nspin = int(invar['nspin']);
    else:
        nspin = 1;
    if 'Eplasmon' in invar: # for advanced user, an estimation of integration
        Eplasmon = int(invar['Eplasmon']) #range for C(t)
    else:
        Eplasmon = 50
    if 'minband' in invar: #the first band to be calculated
        minband = int(invar['minband'])
    else:
        minband=1
    if 'maxband' in invar:  # the last band to be calculated
        maxband = int(invar['maxband'])
    else:
        maxband = 1
    if 'bdgw_min' in invar: # the label of the first band in SIG file.
        bdgw_min = int(invar['bdgw_min']) # consistent with input of abinit:bdgw 
    else:
    	bdgw_min = 1
    if 'bdgw_max' in invar:  # the label of the first band in SIG file.
        bdgw_max = int(invar['bdgw_max']) # consistent with input of abinit:bdgw
    else:
    	bdgw_max = 1
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
    if 'enmin' in invar: # the minimum \omega in A(\omega)
    	enmin = float(invar['enmin'])
    else:
    	enmin = -20.0
    if 'enmax' in invar: # the maximum \omega in A(\omega)
    	enmax = float(invar['enmax'])
    else:
    	enmax = 0.0 
    
    if 'sfactor' in invar: # not implemented yet
    	sfac = float(invar['sfactor'])
    else:
    	sfac=1.0
    if 'pfactor' in invar:  # not implemented yet
    	pfac = float(invar['pfactor'])
    else:
    	pfac=1.0
    if 'penergy' in invar: # not implemented yet
    	penergy = int(invar['penergy'])
    else:
    	penergy = 0
    if 'extinf' in invar:  # not implemented yet
        extinf = float(invar['extinf']);
    else:
        extinf = 0
    if 'calc_gw' in invar:  #enable GW calculation
    	flag_calc_gw = int(invar['calc_gw'])
    else:
    	flag_calc_gw = 0
    if 'spf_qp' in invar:  #enable GW calculation
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
    
    if 'calc_rc' in invar: # enable retarded cumulant calculation
    	flag_calc_rc = int(invar['calc_rc'])
    else:
    	iflag_calc_rc = 0
    if 'rc_Josh' in invar: # enable retarded cumulant calculation
    	rc_Josh = int(invar['rc_Josh'])
    else:
    	rc_Josh = 0
    
    if 'calc_crc' in invar: #enable CRC so as TOC96 calculation
    	flag_calc_crc = int(invar['calc_crc']) # CRC implementation is not
                                            #ready yet!!
    else:
    	flag_calc_crc = 0
    if 'gwfermi' in invar: #Fermi enegy after GW calculation
        gwfermi = float(invar['gwfermi']);
    else:
        gwfermi = 0.0
    if 'lda_fermi' in invar: # Fermi energy after LDA calculation
        lda_fermi = float(invar['lda_fermi']); # will be used
    else:                                    #when scgw=0
        lda_fermi = 0.0
    if 'invar_den' in invar: #d\omega in the cumulant A(\omega) 
    	invar_den = float(invar['invar_den']) # for the moment the choices 
    else:                                 # are 0.05, 0.025, 0.0125, 0.01,0.005
    	invar_den = 0.05
    if 'metal_valence' in invar: # TOC for metal valence is implemented
        metal_valence = float(invar['metal_valence']) #different with core
    else:
        metal_valence = 0
    if 'invar_eta' in invar: #lorentzian broadening of all cumulant A(\omega)
        invar_eta = float(invar['invar_eta'])
    else:
        invar_eta = 0.1
    if 'Gaussian' in invar: #lorentzian broadening of all cumulant A(\omega)
        gbro = float(invar['Gaussian'])
    else:
        gbro = 0.01
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
    
    if 'abinit_eqp' in invar: # use eqp from abinit or recalculated
    	abinit_eqp = int(invar['abinit_eqp']) #in this code.
    else:
    	abinit_eqp = 0 
else :
    print ("Invar file not found (invar.in). Impossible to continue.")
    sys.exit(1)
print ("Reading invar done.")
#print(" "+"===="+" Input variables "+"====")
#print()
#npoles = int(150)  #for sampling Im\Sigma lesser to calculate crc_unocc
nband = bdgw_max - bdgw_min + 1
hartree = read_hartree()
if rc_Josh == 1:
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
if flag_pjt == 1:
    pjt1, pjt2, pjt3 =read_pjt_new(nkpt,nband,bdgw_min,nspin) #  read_pjt()
    cs1, cs2, cs3 = read_cs(Ephoton)
else:
    print("""
          WARNING:  no projections of s, p, or d!
         """)
    pjt1 = np.zeros((nkpt,nband))
    pjt2 = np.zeros((nkpt,nband))
    pjt3 = np.zeros((nkpt,nband))
    cs1 = 0
    cs2 = 0
    cs3 = 0

with open("pjt1.dat", 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerows(zip (pjt1))
with open("pjt2.dat", 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerows(zip (pjt2))
with open("pjt3.dat", 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerows(zip (pjt3))
en, res, ims = read_sigfile(sigfilename, nkpt, bdgw_min, bdgw_max)

if extrinsic == 1:
    Rx, Ry = read_R(rs)
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

def gbroaden(x,f,sigma):
	x = np.asarray(x)
	f = np.asarray(f)
	xsize = np.size(x)
	broadf = np.zeros(f.size)
	print("gbroaden() called, x.size:", x.size)
	#dx =  x[1] - x[0] 
        dx =  ( x[-1] - x[0] ) / float(xsize - 1)
	print("gbroaden() called, gaussian window size (4 x sigma):", 4*sigma)
	for n in xrange(xsize):
		if  abs( x[n] - x[0] ) > 4*sigma : 
			nrange = n
			print("gbroaden() called, gaussian window size (4 x sigma) in x units (e.g. eV?):", x[nrange] - x[0])
			break
	if 4*sigma < dx*xsize/2 :
		# First chunk (beginning)
		for n in xrange(0,nrange):
#	#		print "Processing... "+str(int(n/xsize)*100)+"\r",
			for m in xrange(n-nrange,n+nrange):
#	#			gaub = np.exp( - ( x[n] - x[m] )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				gaub = dx * np.exp( - ( dx * ( m - n ) )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				if m >= 0 : 
					broadf[n] += f[m] * gaub
				else : 
					broadf[n] += f[0] * gaub
		# Last chunk (end)
		for n in xrange(xsize-nrange,xsize):
#	#		print "Processing... "+str(int(nrange+n/xsize)*100)+"\r",
			for m in xrange(n-nrange,n+nrange):
#	#			gaub = np.exp( - ( x[n] - x[m] )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				gaub = dx * np.exp( - ( dx * ( m - n ) )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				if m < xsize : 
					broadf[n] += f[m] * gaub
				else : 
					broadf[n] += f[-1] * gaub
		# Middle chunk (treated with the standard formula)
		for n in xrange(nrange,xsize-nrange):
#	#		print "Processing... "+str(int(n/xsize)*100)+"\r",
			for m in xrange(n-nrange,n+nrange):
				gaub = dx * np.exp( - ( dx * ( m - n ) )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				#gaub = dx * np.exp( - ( x[n] - x[m] )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				broadf[n] += f[m] * gaub
	else :
		# Standard formula regardless of boundaries (it should work decently in all cases)
		for n in xrange(xsize):
#	#		print "Processing... "+str(int(n/xsize)*100)+"\r",
			for m in xrange(xsize):
				gaub = dx * np.exp( - ( dx * ( m - n ) )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				#gaub = dx * np.exp( - ( x[n] - x[m] )**2 / 2 / sigma**2 ) / np.sqrt(2*np.pi) / sigma
				broadf[n] += f[m] * gaub
#	af = np.trapz(f)
#	abf = np.trapz(broadf)
#	print af, abf
#	broadf = broadf * af / abf
	return broadf

#print ("Moving back to parent directory:\n", origdir)
#chdir(newdir)
# ======== READING _SIG FILE ======= #
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
    eqp_abinit = read_eqp_abinit()
    eqp = eqp_abinit
    with open("eqp_abinit.dat", 'w') as f:
        writer = csv.writer(f, delimiter = '\t')
        writer.writerows(zip (eqp-gwfermi))
else:
    eqp, imeqp = calc_eqp_imeqp(nspin,spf_qp,wtk,bdrange,kptrange,bdgw_min, en, enmin, enmax, res, ims,
                                hartree, gwfermi, nkpt, nband, scgw, Elda)

### ================================= ###
### ===== GW SPECTRAL FUNCTION ====== ###
# GW spectral function part
t_pregw = time.time() 
if flag_calc_gw == 1:
    print(" # ------------------------------------------------ # ")
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
        #outname = "spftot_gw"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        #outfile = open(outname,'w')
        #for i in xrange(np.size(newen)):
        #    outfile.write("%7.4f %15.10e %15.10e\n"% (newen[i],
        #                                              spftot_up[i], spftot_down[i])) # Dump string representations of arrays
        #outfile.close()
        #print(" A(\omega)_GW written in", outname)
        plt.plot(newen,spftot_up,label="ftot_gw_SpinUp")
        plt.plot(newen,spftot_down,label="ftot_gw_SpinDown")

       # plt.plot( newen,spftot_down,label="ftot_gw_SpinDown")

    elif nspin == 1:
        newen, spftot, spftot_pjt1, spftot_pjt2, spftot_pjt3 = calc_spf_gw(cs1,cs2,cs3,pjt1,pjt2,pjt3,bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res, ims, hartree, gwfermi, invar_eta)
       # print(" ### Writing out A(\omega)_GW...  ")
        with open("spftot_gw.dat",'w') as f:
             writer = csv.writer(f, delimiter = '\t')
             writer.writerow(['# w-fermi','# spftot','# spftot_s','# spftot_p','# spftot_d'])
             writer.writerows(zip( newen, spftot, spftot_pjt1, spftot_pjt2, spftot_pjt3))

        spftot_brd =  gbroaden(newen,spftot, gbro) 
        with open("spftot_gw_gbro-"+str(gbro)+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerows(zip (newen, spftot_brd))

        plt.plot(newen,spftot_brd,label="ftot_gw_bro");

if flag_calc_toc11 == 1:
    if nspin == 2: 
        from sf_modules_spin import calc_toc11_spin
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

        #outname = "spftot_toc11"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        #outfile = open(outname,'w')
        #for i in xrange(len(interp_en)):
        #    outfile.write("%8.4f %12.8e %12.8e\n" % (interp_en[i], toc_tot_up[i],
        #                                      toc_tot_down[i]))
        #outfile.close()
        #print(" A(\omega)_TOC11 written in", outname)
        plt.plot(interp_en,up,label="ftot_toc11_SpinUp");
        plt.plot(interp_en,down,label="ftot_toc11_SpinDown");

       # print (" ### Writing out A(\omega)_TOC11..")
    else:
        print( "Calculating TOC11 begins")
        en_toc11, toc11_tot, tot_s, tot_p,tot_d =calc_toc11_new(gwfermi,lda_fermi, bdrange, bdgw_min, kptrange,
                       FFTtsize, en,enmin, enmax, eqp, Elda,scgw, Eplasmon, ims,
                                            invar_den, invar_eta, wtk,
                                            metal_valence,nkpt,nband, Rx, Ry,
                                             extrinsic,pjt1,pjt2,pjt3,cs1,cs2,cs3)
        
        with open("spftot_toc11"+"-ext"+str(extrinsic)+".dat",'w') as f:
             writer = csv.writer(f, delimiter = '\t')
             writer.writerow(['# w-fermi','# spftot','# spftot_s','# spftot_p','# spftot_d'])
             writer.writerows(zip( en_toc11, toc11_tot, tot_s, tot_p,tot_d))
        #outname = "spftot_toc11"+"-ext"+str(extrinsic)+".dat"
        #outfile = open(outname,'w')
        #for i in xrange(len(en_toc11)):
        #    outfile.write("%8.4f %12.8e\n" % (en_toc11[i], toc11_tot[i]))
        #outfile.close()
        #print(" A(\omega)_TOC11 written in", outname)
        #print (" ### Writing out A(\omega)_TOC11..")
        spftot_brd =  gbroaden(en_toc11, toc11_tot, gbro) 
        with open("spftot_toc11_gbro-"+str(gbro)+".dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerows(zip ( en_toc11, spftot_brd))
        plt.plot(en_toc11,spftot_brd,label="ftot_toc11_brd");
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
    print("# ------------------------------------------------ #")
    print ("Calculating RC begins")
    toten, spftot = calc_rc (gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den,
                             invar_eta,wtk,nkpt,nband,Rx, Ry, extrinsic,
                             core,Eplasmon) 
    print (" ### Writing out A(\omega)_rc...  ")

    outname = "spftot_rc"+"-ext"+str(extrinsic)+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(toten)):
        outfile.write("%8.4f %12.8e\n" % (toten[i], spftot[i]))
    outfile.close()
    print (" A(\omega)_rc written in", outname)

    spftot_brd =  gbroaden(toten,spftot, gbro) 
    with open("spftot_rc_gbro-"+str(gbro)+".dat", 'w') as f:
        writer = csv.writer(f, delimiter = '\t')
        writer.writerows(zip (toten, spftot_brd))
    
    plt.plot(toten,spftot_brd,label="ftot_rc_bro");
    if extrinsic == 1 and bg == 1:

        interptot = interp1d(toten, spftot_brd, kind = 'linear', axis = -1)
        spfbg = []
        enqp_exp = -0.13       # all of these should
        spfqp_exp = 4.833522   # be read from input
        en0_exp = -16.671968   # or a file of exp
        spf0_exp = 2.1747096   # spectrum data
        beta = spf0_exp * 1./ np.trapz(spftot_brd[(toten>=
                            en0_exp)&(toten<=0)],toten[(toten>=en0_exp)&(toten<=0)])
        alpha = 1./interptot(enqp_exp)*abs(spfqp_exp - beta * np.trapz(spftot_brd[(toten>=enqp_exp)&
                                                   (toten<=0)],toten[(toten>=enqp_exp)&(toten<=0)])) 
        for w in toten:
            if w < 0:
                spf_tmp = np.trapz(spftot_brd[(toten>=w)&(toten<=0)],toten[(toten>=w)&(toten<=0)])
            else:
                spf_tmp = 0
            spf = alpha*interptot(w) + beta*spf_tmp
            spfbg.append(spf)

        with open("spftot_rc_full.dat", 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerows(zip (toten, spfbg))
    
        plt.plot(toten,spfbg,label="ftot_rc_ext_bg");


if rc_Josh == 1:

    toten, spftot = calc_rc_Josh (gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta, wtk, ehf) 

    outname = "spftot_rc_Josh"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(toten)):
        outfile.write("%8.4f %12.8e\n" % (toten[i], spftot[i]))
    outfile.close()
    plt.plot(toten,spftot,label="ftot_rc_Josh");

print ("Moving back to parent directory:\n", origdir)
chdir(newdir)
end_time = time.time()
print()

t2 = t_pregw - start_time
t3 = end_time - start_time
#title = 'Spectral function '+ 'A (' + r'$\omega $ ' + ') - '+r'$ h\nu = $'+str(penergy)+' eV'
#plt.title(title)
plt.legend(loc=2);
plt.show();

t4 = t3
t5 = t3
if int(t3/3600) >= 1: 
    t4 = t3 - int(t3/3600)*3600
if int(t4/60) >= 1: 
    t5 = t4 - int(t4/60)*60
print(" Calculation lasted "+str(int(t3/3600))+" Hours, "+str(int(t4/60))+" Minutes and "+str(int(t5))+" Seconds")
print(" "+30*"-")
print(" End of program reached.")
