import numpy as np

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
					print 'INVALID SCALING FACTOR'
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

def likelihood_enthalpy(y,x,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0):
	# This function computes the likelihood of the \
	# 	observed enthalpy values given hyperparameters
	#   without the constant involving 2*pi
	# inputs:
	# 	y - observed values. y \in R^n
	# 	x - corresponding components of the solutions
	#		whose enthalpies are observed;
	#		Each row of x is in the format of \
	#		(i,i,i,i,i,i,a,b).Refer to model description\
	#		 for more details. x \in R^{n*8}  
	#	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
	#		hypes, \in R
	#	l0, sig_m0 - hypes of gp associated with the \
	#		solvenets in the solutions. l0 \in R^2, 
	
	n = len(y) # no. of observations
	# compute matern kernel of gp of solvents
	Sig0 = maternKernel([[x[i][6],x[i][7]] for i in \
						range(n)],l0,sig_m0)
	# mean of the mvn dist of all the solutions
	Mu = [2*mu_alpha + mu_beta for i in range(n)]
	# covariance of the mvn dist of all the solutions
	Sig = np.identity(n)
	for k in range(n):
		for kk in range(n):
			if k == kk:
				Sig[k,kk] = 2*sig_alpha + sig_beta + \
					sig_zeta + Sig0[k,kk]
			elif k > kk:
				Sig[k,kk] = Sig[kk,k]
			else:
				count_equal = 0 #no. of same cation/halide
				for i in range(6):
					if x[k][i]==x[kk][i]:
						count_equal += 1
				Sig[k,kk] = count_equal * sig_alpha + \
							sig_zeta + Sig0[k,kk]	
	return mvn_pdf(y,Mu,Sig)

def likelihood_umbo(y,x,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0,l1,sig_m1):
	# This function computes the likelihood of obeserved\
	# 	umbo given hyperparameters
	# inputs:
	#	y - observed values
	# 	x - corresponding components of the solutions
	#		whose enthalpies are observed
	#	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
	#		hypes, \in R
	#	l0, sig_m0 - hypes of gp associated with the \
	#				 primal(enthalpy) source.
	#	l1, sig_m1 - hypes of gp associated with the \
	#				 secondary source. l1 \in R^8

	Mu = [2*mu_alpha + mu_beta for i in range(n)]
	Sig0 = maternKernel([[x[i][6],x[i][7]] for i in \
				range(n)],l0,sig_m0)
	Sig1 = maternKernel(x,l1,sig_m1)
	Sig = np.identity(n)
	for k in range(n):
		for kk in range(n):
			if k == kk:
				Sig[k,kk] = 2*sig_alpha + sig_beta + \
					sig_zeta + Sig0[k,kk] + Sig1[k,kk]
			elif k > kk:
				Sig[k,kk] = Sig[kk,k]
			else:
				count_equal = 0 #no. of same cation/halide
				for i in range(6):
					if x[k][i]==x[kk][i]:
						count_equal += 1
				Sig[k,kk] = count_equal * sig_alpha + \
							sig_zeta + Sig0[k,kk] + Sig1[k,kk]	
	return mvn_pdf(y,Mu,Sig)

def prior_enthalpy(x,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0):
	# This function computes the prior parameters of enthalpy\
	# 	given hyperparameters
	# inputs:
	# 	x - components of all the solutions (For singleHalide
	#		case there are 72 in total)
	#		Each row of x is in the format of \
	#		(i,i,i,i,i,i,a,b).Refer to model description\
	#		 for more details. x \in R^{n*8}  
	#	mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
	#		hypes, \in R
	#	l0, sig_m0 - hypes of gp associated with the \
	#		solvenets in the solutions. l0 \in R^2, 	
	# ouputs:
	#	mu, Sig (enthalpy ~ mvn(mu,Sig))
	
	n = len(x)
	# compute matern kernel of gp of solvents
	Sig0 = maternKernel([[x[i][6],x[i][7]] for i in \
						range(n)],l0,sig_m0)
	# mean of the mvn dist of all the solutions
	Mu = [2*mu_alpha + mu_beta for i in range(n)]
	# covariance of the mvn dist of all the solutions
	Sig = np.identity(n)
	for k in range(n):
		for kk in range(n):
			if k == kk:
				Sig[k,kk] = 2*sig_alpha + sig_beta + \
					sig_zeta + Sig0[k,kk]
			elif k > kk:
				Sig[k,kk] = Sig[kk,k]
			else:
				count_equal = 0 #no. of same cation/halide
				for i in range(6):
					if x[k][i]==x[kk][i]:
						count_equal += 1
				Sig[k,kk] = count_equal * sig_alpha + \
							sig_zeta + Sig0[k,kk]	
	return mvn_pdf(y,Mu,Sig)
