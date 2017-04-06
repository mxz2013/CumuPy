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

def nonblank_lines(f):
   for l in f:
      line = l.rstrip()
      if line:
         yield line
 
def read_eqp_abinit():
    import numpy as np;
    if isfile('../eqp_abinit.dat'):
        print(" Reading file eqp_abinit.dat... ")
        eqpfile = open("../eqp_abinit.dat");
        eqp_abinit = [];
        for line in eqpfile.readlines():
            eqp_abinit.append(map(float,line.split()));
        eqpfile.close()
        print("Done.")
        eqp_abinit = np.array(eqp_abinit);
    else:
        print("eqp_abinit.dat not found!")
        sys.exit(1)
    return eqp_abinit

def read_hartree():
    import numpy as np;
    if isfile("hartree.dat"):
        print(" Reading file hartree.dat... ")
        hartreefile = open("hartree.dat");
        hartree = [];
        for line in hartreefile.readlines():
            hartree.append(map(float,line.split()));
        hartreefile.close()
        print("Done.")
        hartree = np.array(hartree);

    elif isfile("E_lda.dat") and isfile("Vxc.dat"):
        print(" Auxiliary file (hartree.dat) not found.")
        print(" Reading files E_lda.dat and Vxc.dat... ")
        Eldafile = open("E_lda.dat");
        Vxcfile = open("Vxc.dat");
        elda = [];
        vxc = [];
        for line in Eldafile.readlines():
            elda.append(map(float,line.split()));
        Eldafile.close()
        for line in Vxcfile.readlines():
            vxc.append(map(float,line.split()));
        Vxcfile.close()
        print("Done.")
        elda = np.array(elda);
        vxc = np.array(vxc);
        hartree = elda - vxc
    else:
        print ("hartree.dat not found! Impossible to continue!!")
        sys.exit(1)
    return hartree

def read_hf():
    import numpy as np;

    if isfile("Sig_x.dat"):
        print(" Reading Sig_x.dat... ")
        Sigxfile = open("Sig_x.dat");
        Sig_x = []; 
        for line in Sigxfile.readlines():
            Sig_x.append(map(float,line.split()));
        Sigxfile.close()
        print("Done.")
        Sig_x = np.array(Sig_x);
    return Sig_x

def read_lda():
    import numpy as np;
    if isfile("E_lda.dat"):
        print(" Reading file E_lda.dat... ")
        ldafile = open("E_lda.dat");
        Elda = [];
        for line in ldafile.readlines():
            Elda.append(map(float,line.split()));
        ldafile.close()
        print("Done.")
        Elda = np.array(Elda);

    else:
        print ("E_lda.dat not found!")
        sys.exit(1)
    return Elda
def read_pjt_new(nkpt,nband,bdgw_min,nspin):
    import numpy as np
    if isfile("pjt_s.dat") and isfile("pjt_p.dat") and isfile("pjt_d.dat"):
        lines1 = [line.rstrip('\n') for line in open('pjt_s.dat')]
        lines2 = [line.rstrip('\n') for line in open('pjt_p.dat')]
        lines3 = [line.rstrip('\n') for line in open('pjt_d.dat')]
        s_nk = []
        s_ns = []
        s_nb = []
        s_pjt = []
        p_pjt = []
        d_pjt = []
        for plotPair in nonblank_lines(lines1):
            if not plotPair.startswith('#'):
                data =  plotPair.split()
                s_ns.append(int(data[0].rstrip('\r')))
                s_nk.append(int(data[1].rstrip('\r')))
                s_nb.append(int(data[2].rstrip('\r')))
                s_pjt.append(float(data[3].rstrip('\r')))
        for plotPair in nonblank_lines(lines2):
            if not plotPair.startswith('#'):
                data =  plotPair.split()
                p_pjt.append(float(data[3].rstrip('\r')))
        for plotPair in nonblank_lines(lines3):
            if not plotPair.startswith('#'):
                data =  plotPair.split()
                d_pjt.append(float(data[3].rstrip('\r')))
    else:
        print ("pjt_x.dat not found!")
        sys.exit(1)
    tmp_k = 1
    spjt_new = []
    ppjt_new = []
    dpjt_new = []

    while tmp_k <= int(nkpt/nspin):
        tmp_s = 1
        while tmp_s <= nspin:
            for i in xrange(len(s_ns)):
                if s_nk[i] == tmp_k and s_ns[i] == tmp_s and s_nb[i] >= bdgw_min:
                    spjt_new.append(s_pjt[i])
                    ppjt_new.append(p_pjt[i])
                    dpjt_new.append(d_pjt[i])
            tmp_s += 1
        tmp_k += 1
    spjt=np.zeros((nkpt, nband))
    ppjt=np.zeros((nkpt, nband))
    dpjt=np.zeros((nkpt, nband))

    for i in xrange(nband):
       spjt[:,i] = spjt_new[i::nband]
       ppjt[:,i] = ppjt_new[i::nband]
       dpjt[:,i] = dpjt_new[i::nband]


    return spjt, ppjt, dpjt

def read_pjt():
    import numpy as np;
    if isfile("pjt_s.dat") and isfile("pjt_p.dat") and isfile("pjt_d.dat"):
        print(" Reading file pjt.dat... ")
        pjt1file = open("pjt_s.dat");
        pjt2file = open("pjt_p.dat");
        pjt3file = open("pjt_d.dat");
        pjt1 = [];
        pjt2 = [];
        pjt3 = [];
        for line in pjt1file.readlines():
            pjt1.append(map(float,line.split()));
        pjt1file.close()
        for line in pjt2file.readlines():
            pjt2.append(map(float,line.split()));
        pjt2file.close()
        for line in pjt3file.readlines():
            pjt3.append(map(float,line.split()));
        pjt3file.close()
        print("Done.")
        pjt1 = np.array(pjt1);
        pjt2 = np.array(pjt2);
        pjt3 = np.array(pjt3);

    else:
        print ("pjt.dat not found!")
        sys.exit(1)
    return pjt1, pjt2, pjt3

def read_wtk():
    import numpy as np;
    if isfile("wtk.dat"):
        wtkfile = open("wtk.dat");    
        wtk = [];
        for line in wtkfile.readlines():
            wtk.append((float(line)));
        wtkfile.close()
        wtk = np.array(wtk); 
    else :
        print("wtk.dat not found!")
        sys.exit(1)
    return wtk

def read_sigfile(sigfilename, nkpt, bdgw_min, bdgw_max):
    """
    this function reads _SIG file of abinit GW calculation
    and return the real and imaag part of self-energy for 
    the band range in (bdgw_min, bdgw_max)
    """
    import glob
    from time import sleep
    print("read_sigfile :: ")
    # We put the content of the file (lines) in this array
    #sigfilename = invar_dict['sigmafile']
    #spin = int(invar_dict['spin'])
    #nspin = int(invar_dict['nspin'])
    #en=[] 
    if isfile(sigfilename):
        insigfile = open(sigfilename);
    #elif sigfilename is None:
    else:
        print("File "+ sigfilename+" not found.")
        sigfilename = glob.glob('*_SIG')[0]
        print("Looking automatically for a _SIG file... ",sigfilename)
        insigfile = open(raw_input("Self-energy file name (_SIG): "))
    # We put the content of the file (lines) in this array
    filelines = insigfile.readlines()
    firstbd = 0
    lastbd = 0
    nbd = 0
    #sngl_row = True # are data not split in more than 1 line?
    with open(sigfilename) as insigfile:
        filelines = insigfile.readlines() 
        #nkpt = calc_nkpt_sigfile(insigfile,spin)
        #if invar_dict['gwcode'] == 'exciting':
         #   invar_dict['wtk'] = read_wtk_sigfile(insigfile)
        insigfile.seek(0)
        insigfile.readline()
        line = insigfile.readline()
        #firstbd = int(line.split()[-2])
        #lastbd =  int(line.split()[-1])
        firstbd = bdgw_min
        lastbd = bdgw_max
        nbd = lastbd - firstbd + 1
        print("nbd:",nbd)
        num_cols = len(insigfile.readline().split())
        num_cols2 = len(insigfile.readline().split())
        print("numcols:",num_cols)
        print("numcols2:",num_cols2)
        if num_cols != num_cols2: 
            print()
            print(" WARNING: newlines in _SIG file.")
            print(" Reshaping _SIG file structure...")
            print(" _SIG file length (rows):", len(filelines))
            new_list = []
            nline = 0
            a = []
            b = []
            for line in filelines:
                #if line.split()[0] == "#":
                if '#' in line:
                    print(line.strip('\n'))
                    continue
                elif nline == 0: 
                    a = line.strip('\n')
                    nline += 1
                else: 
                    b = line.strip('\n')
                    new_list.append(a + " " + b)
                    nline = 0
            print("New shape for _SIG array:",np.asarray(new_list).shape)
            tmplist = []
            tmplist2 = []
            for line in new_list:
                tmplist.append(map(float,line.split())[0])
                tmplist2.append(map(float,line.split())[1:])
            for el1 in tmplist2:
                for j in el1:
                    try:
                        float(j)
                    except:
                        print(j)
            #tmplist = map(float,tmplist)
            #tmplist2 = map(float,tmplist2)
            xen = np.asarray(tmplist)
            x = np.asarray(tmplist2)
        else:
            insigfile.seek(0)
            xen = np.genfromtxt(sigfilename,usecols = 0)
            insigfile.seek(0)
            x = np.genfromtxt(sigfilename,usecols = xrange(1,num_cols), filling_values = 'myNaN')
    #nkpt = int(invar_dict['nkpt'])
    print("nkpt:",nkpt)
    #print("spin:",spin)
    #print("nspin:",nspin)
    # From a long line to a proper 2D array, then only first row
    #print(xen.shape)
    print("x.shape", x.shape)
    #if spin == 1 and nspin == 0:
    #    nspin = 2
    #else:
    #    nspin = 1
    #print("nspin:",nspin)
    nspin = 1
    print("size(xen):",xen.size)
    print("The size of a single energy array should be",\
            float(np.size(xen))/nkpt/nspin)
    en = xen.reshape(nkpt*nspin,np.size(xen)/nkpt/nspin)[0]
    #en = xen.reshape(nkpt,np.size(xen)/nkpt)[0]
    print("New shape en:",np.shape(en))
    print("First row of x:",x[0])
    #nb_clos = 3
  #  if invar_dict['gwcode'] == 'abinit':
   #     nb_cols = 3
   # elif invar_dict['gwcode'] == 'exciting':
   #     nb_cols = 2
       #b = x.reshape(nkpt*nspin, np.size(x)/nkpt/nspin/nbd/3, 3*nbd)
    b = x.reshape(nkpt*nspin, np.size(x)/nkpt/nspin/nbd/3, 3*nbd)
    print("New shape x:", b.shape)
    y = b[0::nspin,:, 0::3]
    z = b[0::nspin,:, 1::3]
    res = np.rollaxis(y, -1, 1)
    ims = np.rollaxis(z, -1, 1)
    print("New shape res, ims:", res.shape)
    print("First and last band in _SIG file:", firstbd, lastbd)
    print(" Done.")

    return en, res, ims 

def calc_spf_gw(pjt1,pjt2,pjt3,bdrange, kptrange, bdgw_min, wtk, en, enmin, enmax, res,
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
            spfkb_pjt1 = spfkb*pjt1[ik,ib] 
            spfkb_pjt2 = spfkb*pjt2[ik,ib] 
            spfkb_pjt3 = spfkb*pjt3[ik,ib] 

            spftot += spfkb*wtk[ik]
            spftot_pjt1 += spfkb*wtk[ik]*pjt1[ik,ib]
            spftot_pjt2 += spfkb*wtk[ik]*pjt2[ik,ib]
            spftot_pjt3 += spfkb*wtk[ik]*pjt3[ik,ib]
            
            with open("spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
                 'w') as f:
                writer = csv.writer(f, delimiter = '\t')
 		writer.writerow(['# w-fermi','# spf','# spf_s','# spf_p','# spf_d','# w-hartree-ReSigma', '# ReSigma','# ImSigma'])
                writer.writerows(zip (newen-gwfermi, spfkb,spfkb_pjt1, spfkb_pjt2, spfkb_pjt3,
                                      redenom, tmpres, tmpim))
            #outnamekb = "spf_gw-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
            #outfilekb = open(outnamekb,'w')
            #for ien in range(np.size(newen)):
            #    newen[ien] = newen[ien] - efermi
            #    outfilekb.write("%8.4f %12.8e %12.8e %12.8e %12.8e\n" % (newen[ien], spfkb[ien], redenom[ien], tmpres[ien], tmpim[ien]))
            #outfilekb.close()
    return newen-gwfermi, spftot, spftot_pjt1, spftot_pjt2, spftot_pjt3



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

def calc_eqp_imeqp(nspin,spf_qp, wtk,bdrange, kptrange,bdgw_min, en,enmin, enmax, res, ims, hartree, gwfermi, nkpt, nband, scgw, Elda):
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
                qpspftot += qpspfkb*wtk[ik]
                with open("spf_qp"+"-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
 		    writer.writerow(['# w-fermi','# QP spectra'])
                    writer.writerows(zip (newen-gwfermi, qpspfkb))
            if spf_qp == 1 and nspin == 2 and ik%2 == 0:
                ikeff = int(ik/2 + 1)
                qpspfkb =  abs(imeqp[ik,ib])/np.pi/((newen-eqp[ik,ib])**2 + imeqp[ik,ib]**2)
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

def calc_toc11_new (gwfermi,lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, Eplasmon, ims, invar_den,
                    invar_eta, wtk, metal_valence,nkpt,nband):
    import numpy as np
    import pyfftw
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    import csv
    print("calc_toc11 : :")
    if metal_valence ==1:
        print("""
              WARNING: You are using TOC to calculate valence
              band of metal !! Please be sure that in SIG file,
              the maximum energy covers all -eqp_kb, otherwise, 
              the code might not run !!!
              """)
    ddinter = 0.005 
    newen_toc = np.arange(enmin, enmax, ddinter)
    toc_tot =  np.zeros((np.size(newen_toc))) 
    #pdos = np.array(pdos)
    tol_fermi = 1e-3
    fftsize = FFTtsize
    norm = np.zeros((nkpt,nband))
    outname = "Norm_check_toc11.dat"
    outfile = open(outname,'w')
    for ik in kptrange:
        ikeff = ik + 1
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
                    print("""Zero is not in the intergration of ImSigma(w),
                          please check invar_den""")

                    sys.exit(0)
                ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
                ShiftIms = interpims(ShiftEn)
                ShiftIms_0 = interpims(NewEn_0+Elda_kb)
                #with open ('Encut.dat', 'w') as f:
                #    writer = csv.writer(f, delimiter = '\t')
                #    writer.writerows(zip (NewEn, ShiftIms))
                with open("ShiftIms_toc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
 		    writer.writerow(['# w','# ImSigma(w-eqp)'])
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
                    gw_list.append(0.5*wtk[ik]/np.pi*cwIm)

                print("IFFT done .....")
                interp_toc = interp1d(w_list, gw_list, kind='linear', axis=-1)
                interp_en = newen_toc

                spfkb = interp_toc(interp_en)
                toc_tot += spfkb
                with open("TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat", 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
 		    writer.writerow(['# w-fermi','# spf_toc11'])
                    writer.writerows(zip (interp_en-gwfermi, spfkb/wtk[ik]))
                #outnamekb = "TOC11-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat"
                #outfilekb = open(outnamekb,'w')
                #en_toc11 = []
                #for i in range(len(interp_en)):
                #    en_toc11.append(interp_en[i])
                #    outfilekb.write("%8.4f %12.8e \n" % (interp_en[i],spfkb[i])) 
                #outfilekb.close()
                norm[ik,ib] = np.trapz(spfkb,interp_en)/(wtk[ik])
                print("check the renormalization : :")
                print()
                print("the normalization of the spectral function is",norm[ik,ib])
                if abs(1-norm[ik,ib])>0.01:
                    print("WARNING: the renormalization is too bad!\n"+\
                          "Increase the time size to converge better.", ikeff,ibeff)
    
                outfile.write("%14.5f" % (norm[ik,ib]))
        outfile.write("\n")
    outfile.close()
    return interp_en-gwfermi, toc_tot

def calc_rc (gwfermi, lda_fermi, bdrange, bdgw_min, kptrange, FFTtsize, en,enmin, enmax,
                    eqp, Elda, scgw, ims, invar_den, invar_eta, wtk,nkpt,nband):
    import numpy as np
    import pyfftw
    import csv
    from numpy.fft import fftshift,fftfreq
    from scipy.interpolate import interp1d
    print("calc_rc : :")

    ddinter = 0.005 
    newen_rc = np.arange(enmin, enmax, ddinter)
    rc_tot =  np.zeros((np.size(newen_rc))) 
    #pdos = np.array(pdos,nkpt,nband)
    fftsize = FFTtsize
    norm = np.zeros((nkpt,nband))
    outname = "Norm_check_rc.dat"
    outfile = open(outname,'w')
    for ik in kptrange:
        ikeff = ik + 1
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
            
            NewEn = [x for x in NewEn_0 if abs(x) > 1e-6]
            NewEn = np.asarray(NewEn)
            NewEn_size = len(NewEn)
            if NewEn_size == len(NewEn_0):
                print("""invar_den should  be 0.1*0.5*n where n is
                      integer number!!!""")

                sys.exit(0)
            ShiftEn = NewEn + Elda_kb #np.arange(NewEn_min + Elda_kb, NewEn_max
            ShiftIms = interpims(ShiftEn)
            ShiftIms_0 = interpims(NewEn_0+Elda_kb)
            gt_list = []
            for t in trange:
                tImag = t*1.j 
                area_tmp = 1.0/np.pi*abs(ShiftIms)*(np.exp(-(NewEn)*tImag)-1.0)*(1.0/((NewEn)**2))
                ct_tmp = np.trapz(area_tmp, NewEn)
                gt_tmp = np.exp(ct_tmp)
                gt_list.append(gt_tmp)
            with open("ShiftIms_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",
                                  'w') as f:
                writer = csv.writer(f, delimiter = '\t')
 		writer.writerow(['# w','# ImSigma(w-eqp)'])
                writer.writerows(zip (NewEn, ShiftIms))

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
                Area2 = s_go/(w-eqp_kb-s_freq-eta) 
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
            with open ("spf_rc-k"+str("%02d"%(ikeff))+"-b"+str("%02d"%(ibeff))+".dat",'w') as f:
                writer = csv.writer(f, delimiter = '\t')
 		writer.writerow(['# w-fermi','# spf_rc'])
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
            norm[ik,ib] = np.trapz(spfkb,interp_en)/(wtk[ik])
            print("check the renormalization : :")
            print()
            print("the normalization of the spectral function is",norm[ik,ib])
            if abs(1-norm[ik,ib])>0.01:
                print("WARNING: the renormalization is too bad!\n"+\
                      "Increase the time size to converge better.", ikeff,ibeff)
    
            outfile.write("%14.5f" % (norm[ik,ib]))
        outfile.write("\n")
    outfile.close()
    return interp_en-gwfermi, rc_tot

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
