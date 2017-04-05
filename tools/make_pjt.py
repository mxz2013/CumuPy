import numpy as np
import csv

def nonblank_lines(f):
   for l in f:
      line = l.rstrip()
      if line:
         yield line

nkpt = 32
nband =60
nspin = 2
ns = []
nk = []
nb = []
pj = []

lines_1 = [line.rstrip('\n') for line in
          open('projection.tex')]
for plotPair in nonblank_lines(lines_1):
     if not plotPair.startswith("#"):
         xANDy =  plotPair.split()
         ns.append(float(xANDy[0].rstrip('\r')))
         nk.append(float(xANDy[1].rstrip('\r')))
         nb.append(float(xANDy[2].rstrip('\r')))
         pj.append(float(xANDy[3].rstrip('\r')))
nk_new=[]
ns_new=[]
nb_new=[]
pj_new=[]
kptrange = np.arange(1,nkpt+1,1)

tmp_nk = 1
while tmp_nk <= nkpt:
#	print tmp_nk
	tmp_ns = 1
	while tmp_ns <= 2:
		for i in xrange(len(ns)):
			if nk[i] == tmp_nk and ns[i] == tmp_ns:
		#		print tmp_nk
		#		print tmp_ns
				nk_new.append(nk[i])
				ns_new.append(ns[i])
				nb_new.append(nb[i])
				pj_new.append(pj[i])
		tmp_ns += 1
	tmp_nk += 1


with open("projection.dat", 'w') as f:
	writer = csv.writer(f, delimiter = '\t')
	writer.writerow(['# ik','# is','# ib', '# pj'])
	writer.writerows(zip (nk_new,ns_new, nb_new, pj_new))

