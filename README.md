# Cumypy2.7
A python code to compute the spectral functions in the cumulant expansion approximations.

The code is written based on the cumulant code of Dr. Matteo Guzzo.

It is written in python 2.7.12. The required modules are numpy, scipy, matplotlib, pyfftw.  

This code is written according to abinit GW output. Thus all the inputs in the test folder are from abinit calculation with some post process. To use the code, you need to provide the following inputs, i.e.,

1. "SIG" file, which contains frequency, Re\Sigma(\omega), Im\Sigma(\omega), and GW spectral funcitons for each k point and band. Examples can be found in the test folder. If you have self-energy results from other codes, you'd better change them into the abinit format. Otherwise you might have some reading errors.

2. hartree.dat or Elda.dat and Vxc.dat because the hartree energy can be calcualted from Elda and Vxc. The format of hartree.dat, E_lda.dat and Vxc.dat are nb*nk = number of columns * number of lines. Each colomn corresponds to one band, and in each colomn, it contains the energies of each k points. Use grep, awk, past commends you can get hatree.dat as the one in the test folder (see the script namde "make_input"). 

3. "wtk.dat" if you want to have the k-integrated spectral function. The wtk gives the wight of each k-point in the Brillion Zone. Otherwise you can put "flag_wtk = 0" in invar.in file. It is only a function a number of k-points, So it contains only one colomn and the line namber is the same as the number of k points.

4. "Elda.dat" is necessary if you use a one-shot GW self-energy. 

5. "eqp_abinit.dat" is necessary if you want to use the QP energies from abinit by putting "abinit_eqp=1". Otherwise the QP energies will be calculated by CumuPy.

Other required input files are for special calculations, and the users can lean more from the examples shown in the "test" folder.

All possible variables in invar.in are: (complete later) 

  

