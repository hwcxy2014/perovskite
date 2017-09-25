import numpy as np

def get_acceptance_set(a,b):
	# inputs:
	#	a - a vector in R^M
	#	b - a vector in R^M, components of 
	#		b are unique and sorted in ascending order
	# outputs:
	#	c - a vector in R^{M+1}, break points of the real line
	#	A - a set with |A| <= M, acceptance set
	#	Note: for more info about what c and A are, pls
	#		  refer to 'The Knowledge-Gradient Policy for
	#		  Correlated Normal Beliefs' by Peter Frazier
	M = len(a)
	c = [0 for i in range(M+1)]
	A = [0 for i in range(M)]
	infty = 1000 # To replace the infinity used in PF's paper
	c[0] = -infty 
	c[1] = infty
	A[0] = 1 # A is not the A in PF'paper, A[:Alen] is.
	Alen = 1 # Alen records the effective length of A
	# Note on indexing: in Algo 1 of PF's paper,
	#	c = (c0,c1...), a = (a1,..,aM),b = (b1,..,bM).
	#	So here the index of c stays the same, while the
	#	indices of a and b have to substract by 1.
	for i in range(1,M):
		c[i+1] = infty
		loopdone = False
		while loopdone == False:
			j = A[Alen-1]
			c[j] = float(a[j-1]-a[i])/(b[i]-b[j-1])
			if Alen > 1 and c[j] <= c[A[Alen-2]]:
				Alen = Alen - 1
			else:
				loopdone = True
		A[Alen] = i+1
		Alen += 1
	A = A[:Alen]
	return c,A

def get_sorted(a,b):
	# inputs:
	#	a - a vector in R^M
	#	b - a vector in R^M
	# outpus:
	# 	bout - b sorted in ascending order
	#		if bi = bj, and ai<aj, then bi is removed
	#	aout - elements in a corresponding to that in bout
	m = np.stack((a,b))
	m = m[:,np.argsort(m[1,:])]
	aout = list(m[0,:])
	bout = list(m[1,:])
	i = 0
	while i < len(aout)-1:
		if bout[i]==bout[i+1]:
			if aout[i]<=aout[i+1]:
				bout.pop(i)
				aout.pop(i)
			else:
				bout.pop(i)
				aout.pop(i)
		else:
			i += 1
	return aout, bout

def get_maxExp(a,b):
	# inputs:
	#	a - a vector in R^M
	#	b - a vector in R^M
	return 0
