===========
 CumuPy
===========

A python code to compute the spectral functions in the cumulant expansion approximations.

The code is written based on the cumulant code of Dr. Matteo Guzzo. Today it is maintained and developed by Jianqiang (Sky) Zhou.

It is written in python 2.7.12. The required modules are numpy, scipy, matplotlib, pyfftw(>=0.10.4). 

The main module is ``cumupy.py`` which can be excute by ``python cumupy.py >log &``. 
The ``outread_modules.py`` reads all the input files, ``calc_qp.py``, ``calc_gw.py``, ``sf_toc11.py``, and ``sf_rc.py`` 
are modules for the calcualtions of the spectra of QP, GW, TC, and RC, respectively.

This code is written according to abinit GW output. Thus all the inputs in the test folder are from abinit calculation 
with some post process. To use the code, you need to provide the following inputs, i.e.,

1. ``SIG`` file, which contains frequency, Re\Sigma(\omega)), Im\Sigma(\omega), and GW spectral funcitons for each k point and band. 
   Examples can be found in the test folder. If you have self-energy results from other codes, you'd better change them into the abinit 
   format. Otherwise you might have some reading errors.

2. ``hartree.dat`` or ``Elda.dat`` and ``Vxc.dat`` because the hartree energy can be calcualted from Elda and Vxc. The format of ``hartree.dat``, ``Elda.dat`` and ``Vxc.dat`` are nb*nk = number of columns * number of lines. Each colomn corresponds to one band, and in each colomn, it contains the energies of each k points. Use grep, awk, past commends you can get ``hatree.dat`` as the one in the test folder (see the script namde ``make_input``). 

3. ``wtk.dat`` if you want to have the k-integrated spectral function. The wtk gives the wight of each k-point in the Brillion Zone. Otherwise you can put ``flag_wtk = 0`` in ``invar.in`` file. It is only a function a number of k-points, So it contains only one colomn and the line namber is the same as the number of k points.

4. ``Elda.dat`` is necessary if you use a one-shot GW self-energy. 

5. ``eqp_abinit.dat`` is necessary if you want to use the QP energies from abinit by putting ``abinit_eqp=1``. Otherwise the QP energies will be calculated by CumuPy.

Other required input files are for special calculations, and the users can lean more from the examples shown in the ``test`` folder.

Suggested references for the acknowledgment of CumuPy usage are:

[1] Dynamical effects in electron spectroscopy. Jianqiang Sky Zhou, JJ
Kas, Lorenzo Sponza, Igor Reshetnyak, Matteo Guzzo, Christine Giorgetti,
Matteo Gatti, Francesco Sottile, JJ Rehr, Lucia Reining,
The Journal of Chemical Physics 143, 184109 (2015).

[2] Cumulant Green's function calculations of plasmon satellites in bulk
sodium: Influence of screening and the crystal environment, Jianqiang Sky
Zhou, Matteo Gatti, JJ Kas, JJ Rehr, Lucia Reining, Phys. Rev. B 97,
035137 (2018).

If the retarded cumulant is calculated:

[3] Cumulant expansion of the retarded one-electron Green function,
J. J. Kas, J. J. Rehr, and L. Reining, Phys. Rev. B 90, 085112 (2014).
  
  
The development of this code has been made possible by financial support from
the European Research Council under the European Union’s
Seventh Framework Programme (FP/2007-2013) / **ERC grant SEED**, grant
agreement no. 320971
