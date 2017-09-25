# This code adopts Peter Frazier's matlab code CorrelatedKG.m for \
#	"Knowledge-Gradient Methods for
#   Ranking and Selection with Correlated Normal Beliefs." INFORMS Journal on
#   Computing. 2009.
# This code doesn't do tie breaking

import numpy as np
import math
import scipy.stats as ss
def logEI(s):
	# input:
	#	s \in R^M
	# output:
	#	logy = E[(s+Z)^+] where Z~N(0,1)
	#		for s < -10, Mill's ratio is used
	#		Refer to Pfrazier's code for more details
	# NOTE:
	#	s can't be less than -35 otherwise pdf function 
	#		returns 0 and log function no longer works

	M = len(s)
	logy = [0. for i in range(M)]
	loca = [i for i in range(M) if s[i]<-10]
	if len(loca) > 0:
		for i in loca:
			logy[i] = np.log(ss.norm.pdf(s[i])) \
						- np.log(pow(s[i],2)+1)
	loca_remain = [i for i in range(M) if s[i]>=-10]
	if len(loca_remain)>0:
		for i in loca_remain:
			logy[i] = np.log(s[i]*ss.norm.cdf(s[i])+ss.norm.pdf(s[i])) 

	return logy
			
def AffineBreakpointsPrep(a,b):
	# inputs:
	#	a,b - vectors of same length
	# outputs:
	#	a,b
	M = len(a)
	[b1,i1] = [min(b),np.argmin(b)]
	[a2,i2] = [max(a),np.argmax(a)]
	[b3,i3] = [max(b),np.argmax(b)]
	a1 = a[i1]
	b2 = b[i2]
	a3 = a[i3]
	cleft = [(a[i]-a1)/(b1-b[i]) for i in range(M)]
	cright = [(a[i]-a3)/(b3-b[i]) for i in range(M)]
	c2left = (a2 - a1)/(b1 - b2)
	c2right = (a2 - a3)/(b3-b2)
	keep = [i for i in range(M) if (b[i]==b1 or b[i]==b3 or \
			cleft[i] <= c2left or cright[i] >= c2right)]
	a = [a[i] for i in keep]
	b = [b[i] for i in keep]
	ba = np.array([b,a]) # b at first row
	ba = np.transpose(ba) # b at the first coln
	ind = np.lexsort((ba[:,1],ba[:,0])) # sort by coln 0 first
	ba = ba[ind]
	a = ba[:,1]
	b = ba[:,0] 
	keep = []
	for i in range(len(b)-1):
		if b[i] != b[i+1]:
			keep.append(i)	
	keep.append(len(b)-1)

	a = [a[i] for i in keep]
	b = [b[i] for i in keep]

	return a,b
	#TODO: test code

def AffineBreakpoints(a,b):
	# inputs:
	#	a,b -
	# outputs:
	#	c, A - refer to AffineBreakpoints.m to see what they are
	M = len(a)
	c = [0 for i in range(M)]
	A = [0 for i in range(M-1)]

	i = 0
	c[i] = float('-inf')
	c[i+1] = float('inf')
	A[0] = 1
	Alen = 1

	for i in range(1,M):
		c[i+1] = float('inf')
		while True:
			print Alen, len(A)
			j = A[Alen-1]
			c[j] = float(a[j-1] - a[i])/(b[i]-b[j-1])
			if Alen > 1 and c[j] <= c[A[Alen-2]]:
				Alen = Alen - 1
			else:
				break
		A[Alen] = i+1
		Alen += 1
	A = A[1:Alen]
	return c,A
	# TODO: test code
		
def logEmaxAffine(a,b):
	# inputs
	#	a,b - vectors in R^M
	# output:
	#	logy = log(E[max_x a_x + b_x Z]-max_x a_x), \
	#			where Z is N(0,1)
	
	#----prep-----
	if len(a) != len(b):
		raise ValueError('dimension of inputs doesn\'t match')
	if sum(map(math.isnan,a)) > 0 or \
		sum(map(math.isnan,b)) > 0:
		print 'mu or sigmatilde contains NaN'
	#------------

	[a,b] = AffineBreakpointsPrep(a,b)
	if (len(a)==1):
		logy = float('-inf')
		return
	[c,keep] = AffineBreakpoints(a,b)

	a = [a[i] for i in keep]
	b = [b[i] for i in keep]
	cposi = [keep[i]+1 for i in range(len(keep))]
	cposi = [1]+cposi
	c = [c[i] for i in cposi]
	M = len(keep)

	bdiff = [b[i+1]-b[i] for i in range(M-1)]
	fc = logEI([-abs(c[i]) for i in range(M-1)])
	logy = math.log(sum([bdiff[i]*fc[i] for i in range(M-1)]))
	return logy 

	# TODO: test code
def correlatedKG(mu,Sigma,noisevar):
	# inputs:
	#	mu - mean vector
	#	Sigma - covariance matrix
	# 	noisevar is a scalar in this code
	# 		It is computed as a hyper parameter
	# outputs:
	#	logQ - a vector of 

	M = len(mu)
	logQ = [0. for i in range(M)]
	for x in range(M):
		Sigma_x = Sigma[x]
		Sigma_xx = Sigma_x[x]
		denom = np.sqrt(Sigma_xx+noisevar)
		if denom == 0:
			logQ[x] = float('-inf')
		else:
			sigmatilde = Sigma_x / denom
			logQ[x] = logEmaxAffine(mu,sigmatilde) 
	xkg = np.argmax(logQ)
	maxLogQ = max(logQ)
	return xkg, maxLogQ




		

