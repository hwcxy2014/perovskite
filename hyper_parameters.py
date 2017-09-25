import scipy
from scipy import spatial
import numpy as np
from scipy import optimize as op


# A sample of how input data looks like
Z = np.array([[0,1,0,1,0,0,0.1,0.2,0],[0,0,1,0,0,1,0.5,0.6,1],\
	[0,1,0,0,0,1,0.1,0.2,0],[0,0,1,0,1,0,0.2,0.4,2]])
V = np.random.random(4)
data = np.hstack([Z, np.reshape(V,(4,1))])

def maternKernel_old(data, nu = 2.5, sig2 = 1., rho = 1.):
	'''
	This function computes matern kernel based on Wikipedia
	  and python scikit gaussian process kernel computation.
	Input : data is a n * 2 array
	Output: n * n matrix
	'''
	data = np.array(data)
	n = len(data)
	# compute pairwise distance between n data points in data
	d = scipy.spatial.distance.pdist(data, metric='euclidean')
	d_pairwise = np.zeros([n,n])
	c = 0 # c acts like a pointer in d
	for i in range(n):
		for j in range(n):
			if i == j:
				d_pairwise[i,j] = 0
			elif i > j: # by symmetry of distance
				d_pairwise[i,j] = d_pairwise[j,i]
			else:
				d_pairwise[i,j] = d[c]
				c+=1 
	K = np.sqrt(5) * d_pairwise / rho
	m_cov = sig2 * (1. + K + K ** 2 / 3.0) * np.exp(-K)
	return m_cov 

def maternKernel(data, l1, l2, sig_m):
	'''
	This function computes matern kernel based on 
		http://www.gaussianprocess.org/gpml/ and discussion with
		Matthias
	Input: 
		- data: a n * 2 array
		- l1, l2: the weights for each of the two dimensions
		- sig_m: the variance of a point
	Output: 
		- matern_kernel: n * n matrix  
	'''
	data = np.array(data)
	n = len(data)
	# compute distance matrix
	d_pairwise = np.zeros([n,n])
	for i in range(n):
		for j in range(n):
			if i==j:
				d_pairwise[i, j] = 0
			elif i > j: # by symmetry of distance
				d_pairwise[i, j] = d_pairwise[j, i]
			else:
				weighted = l1 * pow(data[i,0]-data[j,0],2) + \
							l2 * pow(data[i,1]-data[j,1],2)
				if weighted < 0:
					print 'INVALID'
				d_pairwise[i, j] = np.sqrt(weighted)
	# compute matern kernel
	K = np.sqrt(5) * d_pairwise
	matern_kernel = sig_m * (1. + K + K**2/3.) * np.exp(-K)
	return matern_kernel

def likelihood(data, mu_alpha, sig_alpha, mu_zeta, sig_zeta, \
				sig_beta, sig_m, l1, l2):
	'''
	This function computes the likehood of solubilities \
		given hyper parameters.
	Input: 
		data: n * 10 matrix
		mu_alpha,...,l2: hyper paramters
	Output:
		log likelihood without constant term (term involves pi)
	'''
	data = np.array(data)
	n = len(data)
	# solution info
	Z = data[:,0:9]
	# solubility data
	V = data[:,9]
	# computes matern kernel
	Z_7 = np.zeros([n,2])
	for i in range(n):
		Z_7[i,0] = Z[i][6]
		Z_7[i,1] = Z[i][7]
	# according to the model setup, Sig_0, which is the maternKernel
	#  for the solvents, should be d*d, with d being the # of solvents.
	# Here Sig_0 is n*n, with n being # of different solutions.
	# This doesn't affect the ultimate pdf, but involves some redundant
	#  calculation. For example, if solution i and j have the same
	#  solvent, then the jth and ith row of Sig_0 are the same.
	# Since the total number of solutions are small, so we can dismiss
	#  the small increase in computation effort.
	Sig_0 = maternKernel(Z_7, l1, l2, sig_m)
	mu_0 = np.zeros(n)
	# mean of solubility
	mu = [2*mu_alpha + mu_zeta for x in range(n)]
	# covariance of solubility
	Sig = np.identity(n)
	for x in range(n):
		for xx in range(n):
			if x == xx:
				Sig[x,xx] = 2*sig_alpha + sig_beta + sig_zeta + \
							Sig_0[x, xx]
			elif x > xx:
				Sig[x,xx] = Sig[xx,x]
			else:
				count_equal = 0
				for i in range(6):
					if Z[x,i]==1 and Z[xx,i]==1:
						count_equal += 1
				Sig[x,xx] = count_equal * sig_alpha + sig_zeta + \
							Sig_0[x,xx]
	# random perturbation for Sig so that we don't run into the issue of
	#  having singular matrix during numerical optimization
	rand_pert = np.random.random([n,n]) * 0.000001
	# likelihood with no constant term
	try:
		like = pow(np.linalg.det(Sig),-0.5) * \
			np.exp(-0.5 * ((V-mu).dot(np.linalg.inv(Sig+rand_pert))).dot(V-mu))
	except np.linalg.linalg.LinAlgError as err: 
		like = pow(np.linalg.det(Sig),-0.5) * \
			np.exp(-0.5 * ((V-mu).dot(np.linalg.inv(Sig+rand_pert))).dot(V-mu))
	# log likelihood
	lnlike = np.log(like)
	return lnlike

def likelihoodPact(data, theta):
	'''
	This function 
	'''
	mu_alpha, sig_alpha, mu_zeta, sig_zeta, sig_beta, sig_m, l1, l2 = theta
	neg_log_like = -likelihood(data, mu_alpha, sig_alpha, mu_zeta, sig_zeta, \
						sig_beta, sig_m, l1, l2)
	return neg_log_like 
		

def MLE(data):
	f = lambda *args: likelihoodPact(data, *args)
	# solver 1: scipy optimization
	bnds = ((0, None),(0,None),(None, None), (0, None),(0, None),\
			(0, None),(0,1),(0,1))
	n_start = 5
	mle_list = np.zeros([n_start,8])
	lkh_list = np.zeros(n_start)
	for i in range(n_start):
		init_value = np.random.random(8)
		init_value[0] = init_value[0]*2
		res = op.minimize(f, init_value, \
						bounds = bnds, method = 'L-BFGS-B')
		mle_list[i,:] = res['x']
		lkh_list[i] = res.fun

	lkh_max = np.nanmin(lkh_list)
	ind = np.nanargmin(lkh_list)
	print mle_list
	print lkh_list
	print ind
	# print lkh_max
	return mle_list[ind,:]

	

