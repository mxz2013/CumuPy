=========================================================
 Example for sodium valence spectral function including 
=========================================================

``invar.in`` is the input file that will be read by CumuPy ``sf.py``.

``sp.out`` and ``spo_DS3_SIG`` are GW output files from ABINIT package. Here there are three iterations for the energy self-consistency in the GW calculation, that is why SIG file are taken at ``DS3`` (Data Set 3).

``wtk`` should be extracted from ``sp.out`` file which contains one column and 145 lines of data because the total number of k points is 145 in this case.

Since sodium valence only contains one band, no need to take into accout projections here.

All the rest of the input files that are needed by CumuPy can be produced by excute ``./make_input`` that will automatically make ``SIG, hartree.dat, Elda.dat, Vxc.dat, eqp_abinit.dat `` .

Finally you can run ``python ../../cumupy.py > log &``  

