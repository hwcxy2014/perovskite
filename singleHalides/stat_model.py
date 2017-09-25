import scipy
from scipy import spatial
import numpy as np
from scipy import optimize as op


# A sample of what input data for single halide model looks like
# Z = np.array([[0,1,0,1,0,0,0.1,0.2,0],[0,0,1,0,0,1,0.5,0.6,1],\
#	[0,1,0,0,0,1,0.1,0.2,0],[0,0,1,0,1,0,0.2,0.4,2]])
# The first three entries show which of the three halide is  
#	present in the solution, 4-6th entries show which of the
#	three cations is present in the solution
#	7,8th entries represent the UMBO and polarity respectively.
#	The last entry is in {0,1,...,no.of solvents-1}, showing
#	the numbering of the corresponding solvent. 
# Z corresponds to the domain of the function whose value 
#	we would like to optimize
# V = np.random.random(4) represent the binding energy, which
#	corresponds to the value of the the function
# data = np.hstack([Z, np.reshape(V,(4,1))])

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
	#  the small increase in the computational effort.
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
	except np.linalg.linalg.LinAlgError as erri:
		print 'singular matrix' 
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
		

def MLE(data,nPara):
	# inputs:
	#	data - samples for calculating MLE
	#	nPara - number of parameters to estimate
	# output:
		
	# -----  scipy optimization ------
	# upper bound of mu_alpha is set to be 10 * the max of 10 samples 
	mu_alpha_ub = max([data[i][-1] for i in range(len(data))])
	# upper bound of sig_alpha is set to be 10 * sample variance
	sig_alpha_ub = np.var([data[i][-1] for i in range(len(data))])
	# set upper bounds for all parameters to be estimated
	bnds = ((pow(10,-3), mu_alpha_ub), (pow(10,-3), sig_alpha_ub),\
			(pow(10,-3), mu_alpha_ub), (pow(10,-3), sig_alpha_ub),\
			(pow(10,-3),sig_alpha_ub), (pow(10,-3), sig_alpha_ub*0.1),\
			(pow(10,-3),1),(pow(10,-3),1))
	# select n_start initial points at random and
	#	replicate the 5 points with sig_m replaced by 0.1
	n_start = 5
	# choose n_start points at random
	init_value = [[np.random.random()*bnds[i][1] for i in range(nPara)]\
						for j in range(n_start)]
	mle_list = np.zeros([n_start*2,nPara])
	lkh_list = np.zeros(n_start*2)
	f = lambda *args: likelihoodPact(data, *args)
	for i in range(n_start):
		res = op.minimize(f, init_value[i], \
						bounds = bnds, method = 'L-BFGS-B')
		mle_list[i,:] = res['x']
		lkh_list[i] = res.fun
	# choose n_start points with sig_m = 0.1 (suggested by Matthias)
	for i in range(n_start,n_start*2):
		init_value_modified = init_value[i-n_start]
		init_value_modified[-3] = 0.1
		res = op.minimize(f, init_value_modified, \
						bounds = bnds, method = 'L-BFGS-B')
		mle_list[i,:] = res['x']
		lkh_list[i] = res.fun
		
	lkh_max = np.nanmin(lkh_list)
	ind = np.nanargmin(lkh_list)
	#print mle_list
	#print lkh_list
	#print ind
	# print lkh_max
	return mle_list[ind,:]

	

