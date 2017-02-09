#!/usr/bin/env python
from __future__ import print_function
from sf_modules_new import *
import numpy as np;
import matplotlib.pylab as plt;
plt.figure(1)
import sys
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
    if 'sigmafile' in invar:
        sigfilename = invar['sigmafile'];
    else:
        sigfilename = 'default_SIG';
    if 'flag_wtk' in invar:
        flag_wtk = int(invar['flag_wtk']);
    else:
        flag_wtk = 1;
    if 'scgw' in invar:
        scgw = int(invar['scgw']);
    else:
        scgw = 1;
    if 'spin_on' in invar:
        spin_on = int(invar['spin_on']);
    else:
        spin_on = 0;
    if 'Eplasmon' in invar:
        Eplasmon = int(invar['Eplasmon'])
    else:
        Eplasmon = 20
    if 'minband' in invar:
        minband = int(invar['minband'])
    else:
        minband=1
    if 'maxband' in invar:
        maxband = int(invar['maxband'])
    else:
        maxband = 1
    if 'minkpt' in invar:
        minkpt = int(invar['minkpt'])
    else:
    	minkpt = 1
    if 'maxkpt' in invar:
    	maxkpt = int(invar['maxkpt'])
    else:
    	maxkpt = 1
    if 'nkpt' in invar:
    	nkpt = int(invar['nkpt'])
    else:
    	nkpt = maxkpt - minkpt + 1
    if 'enmin' in invar:
    	enmin = float(invar['enmin'])
    else:
    	enmin = -20.0
    if 'enmax' in invar:
    	enmax = float(invar['enmax'])
    else:
    	enmax = 0.0 
    
    if 'sfactor' in invar:
    	sfac = float(invar['sfactor'])
    else:
    	sfac=1.0
    if 'pfactor' in invar:
    	pfac = float(invar['pfactor'])
    else:
    	pfac=1.0
    if 'penergy' in invar:
    	penergy = int(invar['penergy'])
    else:
    	penergy = 0
    if 'calc_gw' in invar:
    	flag_calc_gw = int(invar['calc_gw'])
    else:
    	flag_calc_gw = 0
    if 'calc_toc11' in invar:
    	flag_calc_toc11 = int(invar['calc_toc11'])
    else:
    	flag_calc_toc11 = 0
    
    if 'calc_rc' in invar:
    	flag_calc_rc = int(invar['calc_rc'])
    else:
    	iflag_calc_rc = 0
    
    if 'calc_crc' in invar:
    	flag_calc_crc = int(invar['calc_crc'])
    else:
    	flag_calc_crc = 0
    if 'extinf' in invar:
        extinf = float(invar['extinf']);
    else:
        extinf = 0
    if 'efermi' in invar:
        efermi = float(invar['efermi']);
    else:
        efermi = 0.0
    if 'lda_fermi' in invar:
        lda_fermi = float(invar['lda_fermi']);
    else:
        lda_fermi = 0.0
    if 'invar_den' in invar:
    	invar_den = float(invar['invar_den'])
    else:
    	invar_den = 0.05
    if 'metal_valence' in invar:
        metal_valence = float(invar['metal_valence'])
    else:
        metal_valence = 0
    if 'invar_eta' in invar:
        invar_eta = float(invar['invar_eta'])
    else:
        invar_eta = 0.2
        
    if 'FFTtsize' in invar:
        FFTtsize = int(invar['FFTtsize'])
        if FFTtsize % 2 != 0:
            FFTtsize = FFTtsize+1
    else:
        FFTtsize = 5000
    
    if 'bdgw_min' in invar:
    	bdgw_min = int(invar['bdgw_min'])
    else:
    	bdgw_min = 1
    if 'bdgw_max' in invar:
    	bdgw_max = int(invar['bdgw_max'])
    else:
    	bdgw_max = 1
    if 'abinit_eqp' in invar:
    	abinit_eqp = int(invar['abinit_eqp'])
    else:
    	abinit_eqp = 0 
else :
    print ("Invar file not found (invar.in). Impossible to continue.")
    sys.exit(1)
print ("Reading invar done.")
nband = bdgw_max - bdgw_min + 1
hartree = read_hartree()
#hartree = hartree - efermi

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
#print "WTK is:", wtk

en, res, ims = read_sigfile(sigfilename, nkpt, bdgw_min, bdgw_max, spin=0, nspin=0)
#en = en - efermi


bdrange = range(minband - bdgw_min, maxband - bdgw_min + 1)
kptrange = range(minkpt - 1, maxkpt)

### ===================================================== ###
#give the option of using abinit QP energies of the QP energies 
## recalculate from cumulant code by finding zero crossing
## of the shifted Re\Sigma

# ======== READING _SIG FILE ======= #
#enmit = enmin+efermi
#enmat= enmax+efermi
### ===================================================== ###
print(" # ------------------------------------------------ # ")
# Here we move to a subdirectory to avoid flooding-up the current directory
newdirname = "Spfunctions_test"
origdir = getcwd() # remember where we are
newdir = join(origdir, newdirname) # Complete path of the new directory
print(" Moving into output directory:\n ", newdir)
if not isdir(newdir) :
    mkdir(newdir)
chdir(newdir)

if abinit_eqp == 1:   
    eqp_abinit = read_eqp_abinit()
    eqp = eqp_abinit
else:
    eqp, imeqp = calc_eqp_imeqp(bdrange,kptrange, en, enmin, enmax, res, ims,
                                hartree, efermi, nkpt, nband, scgw, Elda)
### ================================= ###
### ===== GW SPECTRAL FUNCTION ====== ###
# GW spectral function part
t_pregw = time.time() 
print("the spin option is", spin_on)
if flag_calc_gw == 1:
    if spin_on == 1:
        from sf_modules_spin import calc_spf_gw_spin
        newen, spftot_up, spftot_down = calc_spf_gw_spin(bdrange, kptrange,
                                                         bdgw_min, wtk, en, enmin, enmax,
                                                         res,ims, hartree, efermi)
            ### ==== WRITING OUT GW SPECTRAL FUNCTION === ###
        print(" ### Writing out A(\omega)_GW...  ")
        outname = "spftot_gw"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(np.size(newen)):
            outfile.write("%7.4f %15.10e %15.10e\n"% (newen[i],
                                                      spftot_up[i], spftot_down[i])) # Dump string representations of arrays
        outfile.close()
        print(" A(\omega)_GW written in", outname)
        plt.plot(newen,spftot_up,label="ftot_gw_SpinUp")
       # plt.plot( newen,spftot_down,label="ftot_gw_SpinDown")

    elif spin_on == 0:
        newen, spftot = calc_spf_gw(bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res,
                    ims, hartree, efermi, invar_eta)
            ### ==== WRITING OUT GW SPECTRAL FUNCTION === ###
        print(" ### Writing out A(\omega)_GW...  ")
        outname = "spftot_gw"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(np.size(newen)):
            outfile.write("%7.4f %15.10e\n"% (newen[i],spftot[i])) # Dump string representations of arrays
        outfile.close()
        print(" A(\omega)_GW written in", outname)
        plt.plot(newen,spftot,label="ftot_gw");

if flag_calc_toc11 == 1:
    if spin_on == 1: 
        from sf_modules_spin import calc_toc11_spin

        print( "Calculating spin polarized TOC11 begins")
        interp_en, toc_tot_up, toc_tot_down = calc_toc11_spin(bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax, 
                                         eqp, encut, metal_valence, ims, invar_den, invar_eta, wtk)
        
        print(" ### Writing out A(\omega)_TOC11...  ")

        outname = "spftot_toc11"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(len(interp_en)):
            interp_en[i] = interp_en[i] - efermi
            outfile.write("%8.4f %12.8e %12.8e\n" % (interp_en[i], toc_tot_up[i],
                                              toc_tot_down[i]))
        outfile.close()
        print(" A(\omega)_TOC11 written in", outname)
        plt.plot(interp_en,toc_tot_up,label="ftot_toc11_SpinUp");
        print (" ### Writing out A(\omega)_TOC11..")
    else:
        print( "Calculating TOC11 begins")
        interp_en, toc_tot = calc_toc11_new(efermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax, 
                                         eqp, Elda,scgw, Eplasmon, ims,
                                            invar_den, invar_eta, wtk,
                                            metal_valence)
        
        print(" ### Writing out A(\omega)_TOC11...  ")
        outname = "spftot_toc11"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
        outfile = open(outname,'w')
        for i in xrange(len(interp_en)):
            outfile.write("%8.4f %12.8e\n" % (interp_en[i], toc_tot[i]))
        outfile.close()
        print(" A(\omega)_TOC11 written in", outname)
        plt.plot(interp_en,toc_tot,label="ftot_toc11");
        print (" ### Writing out A(\omega)_TOC11..")
        
if flag_calc_crc ==1:       
    print("Calulating CRC begins::")
    
    interp_en, toc_tot, crc_tot = calc_crc(bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax, 
                                         eqp,imeqp, Elda,scgw, Eplasmon, ims,
                                            invar_den, invar_eta, wtk,
                                            metal_valence)

    print(" ### Writing out A(\omega)_TOC96 and CRC...  ")
    outname = "spftot_toc+crc"+"_s"+str(sfac)+"_p"+str(pfac)+"_"+str(penergy)+"ev"+".dat"
    outfile = open(outname,'w')
    for i in xrange(len(interp_en)):
        outfile.write("%8.4f %12.8e %12.8e\n" % (interp_en[i], toc_tot[i], crc_tot[i]))
    outfile.close()
    print(" A(\omega)_TOC96 and CRC written in", outname)
    plt.plot(interp_en,crc_tot,label="ftot_crc");
    print (" ### Writing out A(\omega)_TOC11..")
    
if flag_calc_rc == 1:
    print ("Calculating RC begins")
   # e0=time.time()
   # c0=time.clock()
   # elaps1=time.time() - e0
   # cpu1=time.clock() - c0
   # print ("Starting time (elaps, cpu): %10.6e %10.6e"% (elaps1, cpu1))
    #print (" ### Calculation of exponential A(\omega)_TOC96..  ")
    toten, spftot = calc_rc (bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, encut, ims, invar_den, invar_eta, wtk) 
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
