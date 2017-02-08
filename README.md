# Cumulant_SPF
A python code to compute the cumulante expansion of the Green's function for theoretical spectroscopy.

The code is written based on the cumulant code of Dr. Matteo Guzzo.

It is written in python 2.7.12. The required modules are numpy, scipy, matplotlib, pyfftw, pyfftw.  

This code is written according to abinit GW output. Thus all the inputs in the test folder are from abinit calculation with some post process. To use the code, you need to provide the following inputs, i.e.,

1. SIG file, which contains frequency, Re\Sigma(\omega), Im\Sigma(\omega), and GW spectral funcitons for each k and b. Examples can be found in the test folder. If you have self-energy results from other codes, you'd better change them into the abinit format. Otherwise you might have some reading errors.

2. hartree.dat or E_lda.dat and Vxc.dat because the hartree energy can be calcualted from LDA and Vxc. The format of hartree.dat, E_lda.dat and Vxc.dat are nb*nk = number of columns * number of lines. Each colomn corresponds to one band, and in each colomn, it contains the energies of each k points. Use grep, awk, past commends you can get hatree.dat as the one in the test folder. 

3. wtk.dat if you want to have the k-integrated spectral function. The wtk gives  Otherwise you can put "flag_wtk = 0" in invar.in file. It is only a function a number of k. So it is always only one colomn and the line namber is the same as the number of k points.

4. E_lda.dat is necessary if you use a one-shot GW self-energy. 

5. eqp_abinit.dat is necessary if you want to use the QP energies from abinit by putting "abinit_eqp=1"

The last file you need to provide is invar.in where you define what kind of calculation you want:

    if 'sigmafile' in invar: ## the name of SIG file
        sigfilename = invar['sigmafile'];
    else:
        sigfilename = 'default_SIG';
    if 'flag_wtk' in invar:  ## include wtk or not
        flag_wtk = int(invar['flag_wtk']);
    else:
        flag_wtk = 1;
    if 'scgw' in invar:   ## scGW self-energy is recommanded.
        scgw = int(invar['scgw']);
    else:
        scgw = 1;
    if 'spin_on' in invar:  ## spin polarized or not
        spin_on = int(invar['spin_on']);
    else:
        spin_on = 0;
    if 'Eplasmon' in invar:  ## can be used to control the energy range of integrating \Sigma(\omega)
        Eplasmon = int(invar['Eplasmon'])
    else:
        Eplasmon = 20
    if 'minband' in invar:   ## the first band to be calculated. Note that the band number is consistent with abinit
        minband = int(invar['minband'])
    else:
        minband=1
    if 'maxband' in invar:   ## the last band to be calculated. Note that the band number is consistent with abinit
        maxband = int(invar['maxband'])
    else:
        maxband = 1
    if 'minkpt' in invar: ## the first k point to be calculated 
        minkpt = int(invar['minkpt'])
    else:
    	minkpt = 1
    if 'maxkpt' in invar:  ## the last k point to be calculated
    	maxkpt = int(invar['maxkpt'])
    else:
    	maxkpt = 1
    if 'nkpt' in invar:  # total number of k points in SIG file 
    	nkpt = int(invar['nkpt'])
    else:
    	nkpt = maxkpt - minkpt + 1
    if 'enmin' in invar: # the minimum energy in cumulant spf
    	enmin = float(invar['enmin'])
    else:
    	enmin = -20.0
    if 'enmax' in invar:  # the minimum energy in cumulant spf
    	enmax = float(invar['enmax'])
    else:
    	enmax = 0.0 
    
    if 'sfactor' in invar: # not implenmented yet
    	sfac = float(invar['sfactor'])
    else:
    	sfac=1.0
    if 'pfactor' in invar: # not implenmented yet
    	pfac = float(invar['pfactor'])
    else:
    	pfac=1.0
    if 'penergy' in invar: # not implenmented yet
    	penergy = int(invar['penergy'])
    else:
    	penergy = 0
    if 'calc_gw' in invar: ## calculate GW spf
    	flag_calc_gw = int(invar['calc_gw'])
    else:
    	flag_calc_gw = 0
    if 'calc_toc11' in invar:  ## the cumulant of Matteo Guzzo in prl 2011
    	flag_calc_toc11 = int(invar['calc_toc11'])
    else:
    	flag_calc_toc11 = 0
    
    if 'calc_rc' in invar:  ## Josh's retarded cumulant 
    	flag_calc_rc = int(invar['calc_rc'])
    else:
    	iflag_calc_rc = 0
    
    if 'calc_crc' in invar:  ## constraint retarded cumulant in my thesis
    	flag_calc_crc = int(invar['calc_crc'])
    else:
    	flag_calc_crc = 0
    if 'efermi' in invar:  ## Fermi energy calculated according to QP of abinit code.
        efermi = float(invar['efermi']);
    else:
        efermi = 0.0
    if 'lda_fermi' in invar:  ## lda fermi energy, this is required when scgw=0, i.e. G0W0 self-energy
        lda_fermi = float(invar['lda_fermi']);
    else:
        lda_fermi = 0.0
    if 'invar_den' in invar: ## the energy resolution of cumulant spf which is also a convergent parameter
    	invar_den = float(invar['invar_den'])
    else:
    	invar_den = 0.05
    if 'metal_valence' in invar: # if you are calculating metal valence, the TOC formula is a bit different.
        metal_valence = float(invar['metal_valence'])
    else:
        metal_valence = 0
    if 'invar_eta' in invar:  # the broadening of cumulant spf.
        invar_eta = float(invar['invar_eta'])
    else:
        invar_eta = 0.2
        
    if 'FFTtsize' in invar:  # the size of FFT. Main parameter to be converged.
        FFTtsize = int(invar['FFTtsize'])
        if FFTtsize % 2 != 0:
            FFTtsize = FFTtsize+1
    else:
        FFTtsize = 5000
    
    if 'bdgw_min' in invar: #should be consistent with abinit code, bdgw
    	bdgw_min = int(invar['bdgw_min'])
    else:
    	bdgw_min = 1
    if 'bdgw_max' in invar: ##should be consistent with abinit code, bdgw
    	bdgw_max = int(invar['bdgw_max'])
    else:
    	bdgw_max = 1
    if 'abinit_eqp' in invar: ## use abinit eqp or not, if 0, we recalculate eqp by find zero crossing.
    	abinit_eqp = int(invar['abinit_eqp'])
    else:
    	abinit_eqp = 0 


