#  Example for Al valence spectral function including extrinsic calcualtion

"invar.in" is the input file that will be read by CumuPy "sf.py".

"sp.out" and "spo_DS3_SIG" are GW output files from ABINIT package. Here there are three iterations for the energy self-consistency in the GW calculation, that is why SIG file are taken at "DS3" (Data Set 3).

"wtk" should be extracted from "sp.out" file which contains one column and 145 lines of data because the total number of k points is 145 in this case.

"R_1100.0.dat" contains extrinsic and interference information, which is provided by J. Kas. The extrinsic and inteference calculation is not integrated in CumuPy yet.

"wp_s.dat" contains information for the surface plasmon calculation, which is also provided by J. Kas. This is not integrated in CumuPy yet.

"pjt_s(p,d).dat" files are projections on different atomic orbitals, which is also printed from ABINIT package.

All the rest of the input files that are needed by CumuPy can be produced by excute "./make_input" that will automatically make "SIG, hartree.dat, Elda.dat, Vxc.dat, eqp_abinit.dat " .


Final you can run "python ../../sf.py > log &"  

