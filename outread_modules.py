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
        print("ERROR: eqp_abinit.dat not found!")
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

    elif isfile("Elda.dat") and isfile("Vxc.dat"):
        print(" Reading files E_lda.dat and Vxc.dat... ")
        Eldafile = open("Elda.dat");
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
        print ("ERROR: E_lda.dat not found!")
        sys.exit(1)
    
    #return hartree, hartree_ks
    return hartree
    

def read_hf():
    import numpy as np;
    if isfile("Sigx.dat"):
        print(" Reading Sigx.dat... ")
        Sigxfile = open("Sigx.dat");
        Sig_x = []; 
        for line in Sigxfile.readlines():
            Sig_x.append(map(float,line.split()));
        Sigxfile.close()
        print("Done.")
        Sig_x = np.array(Sig_x);
    else:
        Sig_x = 0.0
        print("""
              WARNING: No Sigx.dat provided! Thus RC of Josh and TC original
              will not calculated!!
              """)
    return Sig_x

def read_lda():
    import numpy as np;
    if isfile("Elda.dat"):
        print(" Reading file Elda.dat... ")
        ldafile = open("Elda.dat");
        Elda = [];
        for line in ldafile.readlines():
            Elda.append(map(float,line.split()));
        ldafile.close()
        print("Done.")
        Elda = np.array(Elda);

    else:
        print ("ERROR: Elda.dat not found!")
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
        print ("ERROR: pjt_s.dat, or pjt_p.dat, or pjt_d.dat  not found!")
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

def read_cs(Ephoton):
    cs1=[]
    cs2=[]
    cs3=[]

    if isfile("cs_"+str(Ephoton)+".dat"):
        print(" Reading file cs_"+str(Ephoton)+".dat... ")
        lines_1 = [line.rstrip('\n') for line in
                   open("cs_"+str(Ephoton)+".dat")]
        for plotPair in nonblank_lines(lines_1):
            if not plotPair.startswith("#"):
                xANDy =  plotPair.split()
                cs1.append(float(xANDy[0].rstrip('\r')))
                cs2.append(float(xANDy[1].rstrip('\r')))  
                cs3.append(float(xANDy[2].rstrip('\r')))  

    else:
        print('cs_'+str(Ephoton)+'.dat not found')
    cs1 = float(cs1[0])
    cs2 = float(cs2[0])
    cs3 = float(cs3[0])
    return cs1, cs2, cs3
    
def read_wps():
    wps1=[]
    wps2=[]
    if isfile("wp_s"+".dat"):
        print(" Reading file wp_s.dat... ")
        lines_1 = [line.rstrip('\n') for line in
                   open("wp_s.dat")]
        for plotPair in nonblank_lines(lines_1):
            if not plotPair.startswith("#"):
                xANDy =  plotPair.split()
                wps1.append(float(xANDy[0].rstrip('\r')))
                wps2.append(float(xANDy[1].rstrip('\r')))  

    else:
        print("wp_s.dat not found")
    wps1 = float(wps1[0])
    wps2 = float(wps2[0])
    return wps1, wps2

def read_R(lda_fermi,gwfermi,scgw, Ephoton):
    if scgw == 1:
        xfermi = gwfermi
    else:
        xfermi = lda_fermi
    Rx = []
    Ry =[]
    if isfile("R_"+str(Ephoton)+".dat"):
        print(" Reading file R_"+str(Ephoton)+".dat... ")
        lines_1 = [line.rstrip('\n') for line in
                   open('R_'+str(Ephoton)+'.dat')]
        for plotPair in nonblank_lines(lines_1):
            if not plotPair.startswith("#"):
                xANDy =  plotPair.split()
                #Rx.append(float(xANDy[0].rstrip('\r'))+xfermi)
                Rx.append(float(xANDy[0].rstrip('\r'))+0)
                Ry.append(float(xANDy[1].rstrip('\r')))  
    else: ## using fit function of Josh
        print("ERROR: file R_"+str(Ephoton)+".dat not found! ")
      #  for i in np.arange(-60,60,0.01):
      #      Rx.append(i)
      #      if i < 0:
      #          eta_rs = 0.274/(rs+0.76)+1.263
      #          Ryi = (0.0025*rs**2 + 0.0225*rs - 0.0092)*(abs(i)**eta_rs)+1
      #      elif i >= 0:
      #          Ryi = 1
      #      Ry.append(Ryi)
    return Rx,Ry

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
        print ("ERROR: pjt.dat not found!")
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
        print("ERROR: wtk.dat not found!")
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
        print("ERROR File"+sigfilename+" not found.")
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
        #print("nbd:",nbd)
        num_cols = len(insigfile.readline().split())
        num_cols2 = len(insigfile.readline().split())
        #print("numcols:",num_cols)
        #print("numcols2:",num_cols2)
        if num_cols != num_cols2: 
            #print()
            #print(" WARNING: newlines in _SIG file.")
            #print(" Reshaping _SIG file structure...")
            #print(" _SIG file length (rows):", len(filelines))
            new_list = []
            nline = 0
            a = []
            b = []
            for line in filelines:
                #if line.split()[0] == "#":
                if '#' in line:
            #        print(line.strip('\n'))
                    continue
                elif nline == 0: 
                    a = line.strip('\n')
                    nline += 1
                else: 
                    b = line.strip('\n')
                    new_list.append(a + " " + b)
                    nline = 0
            #print("New shape for _SIG array:",np.asarray(new_list).shape)
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
    #print("nkpt:",nkpt)
    #print("spin:",spin)
    #print("nspin:",nspin)
    # From a long line to a proper 2D array, then only first row
    #print(xen.shape)
    #print("x.shape", x.shape)
    #if spin == 1 and nspin == 0:
    #    nspin = 2
    #else:
    #    nspin = 1
    #print("nspin:",nspin)
    nspin = 1
   # print("size(xen):",xen.size)
   # print("The size of a single energy array should be",\
   #         float(np.size(xen))/nkpt/nspin)
    en = xen.reshape(nkpt*nspin,np.size(xen)/nkpt/nspin)[0]
    #en = xen.reshape(nkpt,np.size(xen)/nkpt)[0]
    #print("New shape en:",np.shape(en))
    #print("First row of x:",x[0])
    #nb_clos = 3
  #  if invar_dict['gwcode'] == 'abinit':
   #     nb_cols = 3
   # elif invar_dict['gwcode'] == 'exciting':
   #     nb_cols = 2
       #b = x.reshape(nkpt*nspin, np.size(x)/nkpt/nspin/nbd/3, 3*nbd)
    b = x.reshape(nkpt*nspin, np.size(x)/nkpt/nspin/nbd/3, 3*nbd)
    #print("New shape x:", b.shape)
    y = b[0::nspin,:, 0::3]
    z = b[0::nspin,:, 1::3]
    res = np.rollaxis(y, -1, 1)
    ims = np.rollaxis(z, -1, 1)
    #print("New shape res, ims:", res.shape)
    #print("First and last band in _SIG file:", firstbd, lastbd)
    print(" Reading sigma Done.")

    return en, res, ims 

