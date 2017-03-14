#!/usr/bin/env python
from __future__ import print_function
from sf_modules_new import *
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
    if 'flag_wtk' in invar:  ## using wtk.dat or not 
        flag_wtk = int(invar['flag_wtk']);
    else:
        flag_wtk = 1;
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
        invar_eta = 0.2
        
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

en, res, ims = read_sigfile(sigfilename, nkpt, bdgw_min, bdgw_max)

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


#print ("Moving back to parent directory:\n", origdir)
#chdir(newdir)
# ======== READING _SIG FILE ======= #
### ===================================================== ###
print(" # ------------------------------------------------ # ")
# Here we move to a subdirectory to avoid flooding-up the current directory
newdirname = "Spfunctions"
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
    eqp, imeqp = calc_eqp_imeqp(spf_qp,wtk,bdrange,kptrange,bdgw_min, en, enmin, enmax, res, ims,
                                hartree, gwfermi, nkpt, nband, scgw, Elda)

### ================================= ###
### ===== GW SPECTRAL FUNCTION ====== ###
# GW spectral function part
t_pregw = time.time() 
if flag_calc_gw == 1:
    print(" # ------------------------------------------------ # ")
    if nspin == 2:
        from sf_modules_spin import calc_spf_gw_spin
        newen, spftot_up, spftot_down = calc_spf_gw_spin(bdrange, kptrange,
                                                         bdgw_min, wtk, en, enmin, enmax,
                                                         res,ims, hartree, gwfermi)
       # print(" ### Writing out A(\omega)_GW...  ")
        outname = "spftot_gw"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(np.size(newen)):
            outfile.write("%7.4f %15.10e %15.10e\n"% (newen[i],
                                                      spftot_up[i], spftot_down[i])) # Dump string representations of arrays
        outfile.close()
        print(" A(\omega)_GW written in", outname)
        plt.plot(newen,spftot_up,label="ftot_gw_SpinUp")
       # plt.plot( newen,spftot_down,label="ftot_gw_SpinDown")

    elif nspin == 1:
        newen, spftot = calc_spf_gw(bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res,
                    ims, hartree, gwfermi, invar_eta)
       # print(" ### Writing out A(\omega)_GW...  ")
        outname = "spftot_gw"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(np.size(newen)):
            outfile.write("%7.4f %15.10e\n"% (newen[i],spftot[i])) # Dump string representations of arrays
        outfile.close()
        print(" A(\omega)_GW written in", outname)
        plt.plot(newen,spftot,label="ftot_gw");

if flag_calc_toc11 == 1:
    if nspin == 2: 
        from sf_modules_spin import calc_toc11_spin

        print( "Calculating spin polarized TOC11 begins")
        interp_en, toc_tot_up, toc_tot_down = calc_toc11_spin(gwfermi,lda_fermi,bdrange, bdgw_min, kptrange,
                        FFTtsize, en,enmin, enmax, eqp,Elda,scgw,Eplasmon, ims,
                        invar_den, invar_eta, wtk, metal_valence)
        
       # print(" ### Writing out A(\omega)_TOC11...  ")

        outname = "spftot_toc11"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(len(interp_en)):
            outfile.write("%8.4f %12.8e %12.8e\n" % (interp_en[i], toc_tot_up[i],
                                              toc_tot_down[i]))
        outfile.close()
        print(" A(\omega)_TOC11 written in", outname)
        plt.plot(interp_en,toc_tot_up,label="ftot_toc11_SpinUp");
       # print (" ### Writing out A(\omega)_TOC11..")
    else:
        print( "Calculating TOC11 begins")
        en_toc11, toc11_tot = calc_toc11_new(gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax, 
                                         eqp, Elda,scgw, Eplasmon, ims,
                                            invar_den, invar_eta, wtk,
                                            metal_valence)
        
       # print(" ### Writing out A(\omega)_TOC11...  ")
        outname = "spftot_toc11"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(len(en_toc11)):
            outfile.write("%8.4f %12.8e\n" % (en_toc11[i], toc11_tot[i]))
        outfile.close()
        print(" A(\omega)_TOC11 written in", outname)
        plt.plot(en_toc11,toc11_tot,label="ftot_toc11");
        print (" ### Writing out A(\omega)_TOC11..")
        
#if flag_calc_toc96 ==1:       
#    print("# ------------------------------------------------ #")
#    print("Calulating toc96 begins::")
#    
#    en_toc96, toc96_tot, beta_greater = calc_toc96(gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax, 
#                                         eqp, Elda,scgw, Eplasmon, ims,
#                                            invar_den, invar_eta, wtk,
#                                            metal_valence, imeqp,nkpt, nband)
#
#    #print(" ### Writing out A(\omega)_TOC96 and CRC...  ")
#    outname = "spftot_toc96"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
#    outfile = open(outname,'w')
#    for i in xrange(len(en_toc96)):
#        outfile.write("%8.4f %12.8e \n" % (en_toc96[i], toc96_tot[i]))
#    outfile.close()
#    print(" A(\omega)_TOC96 and CRC written in", outname)
#    plt.plot(en_toc96,toc96_tot,label="ftot_toc96");
#    print (" ### Writing out A(\omega)_TOC11..")
    
if flag_calc_crc == 1:
    print("# ------------------------------------------------ #")
    print("Calulating CRC begines::")
    omegampole, ampole = calc_multipole (scgw,Elda,lda_fermi,nkpt,
                                         nband,gwfermi,npoles, ims, kptrange,
                                         bdrange,bdgw_min, eqp, en, enmin,
                                         enmax)
    en_crc, toc96tot, crc_tot =  calc_toc96_crc (gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence, imeqp,nkpt,
                nband,ampole,npoles,omegampole)
    
    #calc_crc(invar_eta,gwfermi, wtk, kptrange, bdrange, bdgw_min, omegampole, ampole, npoles,
    #                      beta_greater, en_toc96, toc96_tot, imeqp, eqp)
    outname = "spftot_toc96"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(en_crc)):
        outfile.write("%8.4f %12.8e \n" % (en_crc[i], toc96tot[i]))
    outfile.close()
    outname = "spftot_crc"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(en_crc)):
        outfile.write("%8.4f %12.8e \n" % (en_crc[i], crc_tot[i]))
    outfile.close()
    plt.plot(en_crc,crc_tot,label="ftot_crc");
    plt.plot(en_crc,toc96tot,label="ftot_toc96");

if flag_calc_rc == 1:
    print("# ------------------------------------------------ #")
    print ("Calculating RC begins")
   # e0=time.time()
   # c0=time.clock()
   # elaps1=time.time() - e0
   # cpu1=time.clock() - c0
   # print ("Starting time (elaps, cpu): %10.6e %10.6e"% (elaps1, cpu1))
    #print (" ### Calculation of exponential A(\omega)_TOC96..  ")
    toten, spftot = calc_rc (gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta, wtk) 
    print (" ### Writing out A(\omega)_rc...  ")

    outname = "spftot_rc"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(toten)):
        outfile.write("%8.4f %12.8e\n" % (toten[i], spftot[i]))
    outfile.close()
    print (" A(\omega)_rc written in", outname)
    plt.plot(toten,spftot,label="ftot_rc");
   # elaps2 = time.time() - elaps1 - e0
    #cpu2 = time.clock() - cpu1 - c0
    #print(" Used time (elaps, cpu): %10.6e %10.6e"% (elaps2, cpu2))
    print (" ### Writing out A(\omega)_rc.")

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
