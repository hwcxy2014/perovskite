import numpy as np
from scipy import optimize as op
 
def maternKernel(x,l,sig_m):
	# This function computes the covariance matrix based on\
	#	5/2 matern Kernel
	# inputs:
	#	x - each row of x represents the coordinate of a \
	#		sample. x \in R^{n*L}
	# 	l - distance scaling vector, with l_i being the \
	#		scaling factor for the ith dimension.\
	#	l \in R^L, where L is the dimension of the coordinate
	#	sig_m - scalar scaling factor
	# output:
	#	cm - n*n covariance matrix
	x = np.array(x)
	n = len(x) # no. of samples
	if n == 0:
		print 'Data cannot be empty'
		return None
	# compute distance matrix
	d_pairwise = np.zeros([n,n])
	for i in range(n):
		for j in range(n):
			if i == j:
				d_pairwise[i,j] = 0
			elif i > j: 
				d_pairwise[i,j] = d_pairwise[j,i]
			else:
				weighted = np.dot(l,pow(x[i]-x[j],2))
				d_pairwise[i,j] = np.sqrt(weighted)
				# in case weighted < 0
				if np.isnan(d_pairwise[i,j]) == True:
					aaaaa=1#print 'INVALID SCALING FACTOR'
	# compute matern kernel
	K = np.sqrt(5) * d_pairwise
	cm = sig_m * (1.+K+K**2/3.)*np.exp(-K)
	return cm
	
def mvn_pdf(y,Mu,Sig):
	# This function computes the likelihood of multivariate normal
	#	without the (2*pi)^(-n/2) term
	# It returns the usual pdf when the covariance matrix is
	#	invertible, and adds a perturbation of N(0,1e-6) to 
	#	the covariance matrix when it is not
	# inputs:
	#	y - samples. y \in R^n
	#	Mu - mean vector. Mu \in R^n
	#	Sig - covariance matrix. Sig \in R^{n*n}
	y = np.array(y)
	Mu = np.array(Mu)
	Sig = np.array(Sig)
	n = len(y)
	try:
		pdf = pow(np.linalg.det(Sig),-0.5) * np.exp(-0.5 * \
				((y-Mu).dot(np.linalg.inv(Sig))).dot(y-Mu))
	except np.linalg.linalg.LinAlgError as err:
		rand_pert = np.random.randn(n,n)*0.001
		Sig = Sig + rand_pert
		pdf = pow(np.linalg.det(Sig),-0.5) * np.exp(-0.5 * \
				((y-Mu).dot(np.linalg.inv(Sig))).dot(y-Mu))
	return pdf

def prior_umbo(x,model,mu_is0,Sig_is0,l1,sig_m1):
	# This function computes the prior parameters of binding energy\
	# 	given hyperparameters
	# inputs:
	# 	x - components of all the solutions (For singleHalide
	#		case there are 72 in total)
	#		Each row of x is in the format of \
	#		(i,i,i,i,i,i,a,b).Refer to model description\
	#		 for more details. x \in R^{n*8}  
	#	model: number of halide ('single' for single Halide model,\
	#			'multi' for multi halide model)
	#	mu_is0, Sig_is0 - prior parameters of the primary source (be)
	#	l1, sig_m1 - hypes of gp associated with the \
	#				 secondary source(umbo). l1 \in R^8 or R^14\
	#				 depending on model
	# outputs:
	#	mu, Sig (umbo ~ mvn(mu,Sig))

	mu = mu_is0[:]
	if model == 'single':
		nh = 8
	else:
		nh = 14
	if len(l1) != nh:
		raise ValueError('dimension of l1 does not match problem setup')
		return
	Sig1 = maternKernel(x,l1,sig_m1)
	Sig = Sig1 + Sig_is0
	return mu, Sig

def prior_be(x,model,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0):
	# This function computes the prior parameters of binding energy\
	# 	given hyperparameters
	# inputs:
	# 	x - components of all the solutions (For singleHalide
	#		case there are 72 in total)
	#		Each row of x is in the format of \
	#		(i,i,i,i,i,i,a,b).Refer to model description\
	#		 for more details. x \in R^{n*8}  
	#	model: number of halide ('single' for single Halide model,\
	#			'multi' for multi halide model)
	#	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
	#		hypes, \in R
	#	l0, sig_m0 - hypes of gp associated with the \
	#		solvenets in the solutions. l0 \in R^2, 	
	# ouputs:
	#	mu, Sig (be ~ mvn(mu,Sig))
	
	n = len(x)
	if model == 'single':
		nh = 6
	elif model == 'multi':
		nh = 12
	else:
		raise ValueError('invalid value for input parameter model')

	# compute matern kernel of gp of solvents
	Sig0 = maternKernel([[x[i][nh],x[i][nh+1]] for i in \
						range(n)],l0,sig_m0)
	# mean of the mvn dist of all the solutions
	Mu = [sum(x[i][:nh])*mu_alpha + mu_zeta for i in range(n)]
	# covariance of the mvn dist of all the solutions
	Sig = np.identity(n)
	for k in range(n):
		for kk in range(n):
			if k == kk:
				Sig[k,kk] = sum(x[k][:nh])*sig_alpha + sig_beta + \
					sig_zeta + Sig0[k,kk]
			elif k > kk:
				Sig[k,kk] = Sig[kk,k]
			else:
				count_equal = 0 #no. of same cation/halide
				for i in range(nh):
					if x[k][i]==x[kk][i]:
						count_equal += 1
				Sig[k,kk] = count_equal * sig_alpha + \
							sig_zeta + Sig0[k,kk]	
	return Mu, Sig

def prior_bu(x,model,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0,l1,sig_m1):
	# This function computes the prior parameters of is0 and is1\
	# 	given hyperparameters
	# inputs:
	# 	x - components of the solutions (For singleHalide
	#		case there are 72 in total)
	#		Each row of x is in the format of \
	#		(i,i,i,i,i,i,a,b).Refer to model description\
	#		 for more details. x \in R^{n*8} or R^{n*14} 
	#	model: number of halide ('single' for single Halide model,\
	#			'multi' for multi halide model)
	#	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
	#		hypes, \in R
	#	l0, sig_m0 - hypes of gp associated with the \
	#		solvenets in the solutions. l0 \in R^2, 	
	#	l1, sig_m1 - hypes of gp associated with the \
	#				 secondary source(umbo). l1 \in R^8 or R^14,
	#				 depending on model
	# outputs:
	#	mu, Sig ((is0,is1)~nvm(mu, Sig))

	# prior of is0
	mu_is0, Sig_is0 = prior_be(x,model,mu_alpha,sig_alpha,mu_zeta,\
	sig_zeta,sig_beta,l0,sig_m0)
	# prior of is1
	mu_is1, Sig_is1 = prior_umbo(x,model,mu_is0,Sig_is0,l1,sig_m1)
	# prior mean of (is0,is1)
	mu = list(mu_is0)+list(mu_is1)
	# prior cov of (is0,is1)
	upperMatrix = np.concatenate((Sig_is0,Sig_is0),axis=1)
	lowerMatrix = np.concatenate((Sig_is0,Sig_is1),axis=1)
	Sig = np.concatenate((upperMatrix, lowerMatrix),axis=0)
	return mu, Sig

def likelihood_hype_bu(y_is0,y_is1,x,model,mu_alpha,\
	sig_alpha,mu_zeta,sig_zeta,sig_beta,l0,sig_m0,l1,sig_m1):
	# This function computes the likelihood of the \
	# 	observed IS0,IS1 samples given hyperparameters
	#   without the constant involving 2*pi
	# Note it requires y_is0 and y_is1 observed on the same x
	# inputs:
	# 	y_is0,y_is1 - observed values for IS0 and IS1. 
	# 	x - corresponding components of the solutions
	#		whose be/umbo are observed;
	#		Each row of x is in the format of \
	#		(i,i,i,i,i,i,a,b).Refer to model description\
	#		 for more details. x \in R^{n*8} or R^{n*14}
	#	model - 'single' or 'multi'  
	#	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
	#		hypes, \in R
	#	l0, sig_m0 - hypes of gp associated with the \
	#		solvenets in the solutions. l0 \in R^2, 
	#	l1, sig_m1 - hypes of gp associated with the \
	#				 secondary source(umbo). l1 \in R^8 or R^14,
	#				 depending on model
	
	n0 = len(y_is0) # no. of observations
	n1 = len(y_is1)
	mu, Sig = prior_bu(x,model,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0,l1,sig_m1)
	y = list(y_is0)+list(y_is1)
	return mvn_pdf(y,mu,Sig)

def loglh_hype_bu(y_is0,y_is1,x,model,theta):
	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta,l01,l02,sig_m0,\
	l11,l12,l13,l14,l15,l16,l17,l18,sig_m1=theta
	# This function computes the log value of likelihood_hype_bu
	return -np.log(likelihood_hype_bu(y_is0,y_is1,x,model,mu_alpha,\
		sig_alpha,mu_zeta,sig_zeta,sig_beta,[l01,l02],sig_m0,\
		[l11,l12,l13,l14,l15,l16,l17,l18],sig_m1))
def hype_estimator_MLE(y_is0,y_is1,x,model):
	# inputs:
	#	y_is0, y_is1 - initial observations, both in R^n
	#	x - solution combs at which initial observations are made
	# 	model - 'single' or 'multi'
	
	# ----- scipy optimization--------
	# number of hyperparameters to estimate
	nPara = 17
	# upper bound of mu_alpha is set to be 10 the max of the initial obs
	y_is0 = list(y_is0)
	y_is1 = list(y_is1)
	mu_alpha_ub = max(y_is0 + y_is1)
	# upper bound of sig_alpha is set to be 10 * sample variance
	sig_alpha_ub = np.var(y_is0+y_is1)
	# set upper bounds for all parameters to be estimated
	bnds = ((pow(10,-3), mu_alpha_ub), (pow(10,-3), sig_alpha_ub),\
			(pow(10,-3), mu_alpha_ub), (pow(10,-3), sig_alpha_ub),\
			(pow(10,-3),sig_alpha_ub), (pow(10,-3),1),(pow(10,-3),1),\
			(pow(10,-3), sig_alpha_ub*0.1),\
			(pow(10,-3),1),(pow(10,-3),1),\
			(pow(10,-3),1),(pow(10,-3),1),\
			(pow(10,-3),1),(pow(10,-3),1),\
			(pow(10,-3),1),(pow(10,-3),1),\
			(pow(10,-3), sig_alpha_ub*0.1))
	# call optimize package from scipy	
	f = lambda *args: loglh_hype_bu(y_is0,y_is1,x,model,*args)
	if model == 'single':
		nh = 8
		# select n_start initial points at random and
		#	replicate the 5 points with sig_m replaced by 0.1
		n_start = 5
		# choose n_start points at random
		init_value = [[np.random.random()*bnds[i][1] for i in range(nPara)]\
							for j in range(n_start)]
		mle_list = np.zeros([n_start*2,nPara])
		lkh_list = np.zeros(n_start*2)

	lkh_max = np.nanmin(lkh_list)
	ind = np.nanargmin(lkh_list)
	print mle_list
	print lkh_list
	print ind
	# print lkh_max
	return mle_list[ind,:]
