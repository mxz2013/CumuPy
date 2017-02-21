# Cumulant_SPF
A python code to compute the cumulant expansion of the Green's function for theoretical spectroscopy.

The code is written based on the cumulant code of Dr. Matteo Guzzo.

It is written in python 2.7.12. The required modules are numpy, scipy, matplotlib, pyfftw, pyfftw.  

This code is written according to abinit GW output. Thus all the inputs in the test folder are from abinit calculation with some post process. To use the code, you need to provide the following inputs, i.e.,

1. SIG file, which contains frequency, Re\Sigma(\omega), Im\Sigma(\omega), and GW spectral funcitons for each k and b. Examples can be found in the test folder. If you have self-energy results from other codes, you'd better change them into the abinit format. Otherwise you might have some reading errors.

2. hartree.dat or E_lda.dat and Vxc.dat because the hartree energy can be calcualted from LDA and Vxc. The format of hartree.dat, E_lda.dat and Vxc.dat are nb*nk = number of columns * number of lines. Each colomn corresponds to one band, and in each colomn, it contains the energies of each k points. Use grep, awk, past commends you can get hatree.dat as the one in the test folder. 

3. wtk.dat if you want to have the k-integrated spectral function. The wtk gives  Otherwise you can put "flag_wtk = 0" in invar.in file. It is only a function a number of k. So it is always only one colomn and the line namber is the same as the number of k points.

4. E_lda.dat is necessary if you use a one-shot GW self-energy. 

5. eqp_abinit.dat is necessary if you want to use the QP energies from abinit by putting "abinit_eqp=1"

The last file you need to provide is invar.in where you define what kind of calculation you want:

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
    if 'spf_qp' in invar:  #enable QP spectra calculation
    	spf_qp = int(invar['spf_qp'])
    else:
    	spf_qp = 0
    if 'calc_toc11' in invar: #enable TOC11 calculation
    	flag_calc_toc11 = int(invar['calc_toc11'])
    else:
    	flag_calc_toc11 = 0
    
    if 'calc_rc' in invar: # enable retarded cumulant calculation
    	flag_calc_rc = int(invar['calc_rc'])
    else:
    	flag_calc_rc = 0
    
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
        
    if 'FFTtsize' in invar: #the number of time steps used in FFT
        FFTtsize = int(invar['FFTtsize'])
        if FFTtsize % 2 != 0:
            FFTtsize = FFTtsize+1
    else:
        FFTtsize = 5000
    
    if 'abinit_eqp' in invar: # use eqp from abinit or recalculated
    	abinit_eqp = int(invar['abinit_eqp']) #in this code.
    else:
    	abinit_eqp = 0 

