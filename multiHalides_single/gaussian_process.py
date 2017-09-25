import numpy as np
from scipy.stats import norm

import hyper_parameters

class solubilities(object):
	def __init__(self, n):
		# number of compounds, in this project n = 9*3*8 = 216
		self._dim = n
		# mean of solubility
		self._mu = np.zeros(n)
		# covariance of solubility
		self._Sig = np.identity(n)
		# error with observation 
		self._err = np.zeros(n) 
		# solubilities observed so far
		self._obs = np.array([])
		# maximum solubility observed so far
		self._max = 0.

	def setPrior(self, Z, mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,sig_m,l1,l2):
		'''
		input: 
		Z is n * 15, x^th row of Z describes the components of solution x.
		Z_x = (b1,...,b15) where b1,...,b12 \in {0,1}, b13,b14 \in R, b15 \in {0,...,7}.
		mu_alpha,...,l2 are hyper-parameters that are estimated in \
			seperated functions.
		sig_alpha, sig_beta are variance instead of standard dev.
		'''
		Z = np.array(Z)
		self._mu = [4*mu_alpha + mu_zeta for x in range(self._dim)]
		matern = hyper_parameters.maternKernel(Z[:,12:14],l1,l2,sig_m)
		for x in range(self._dim):
			for xx in range(self._dim):
				if x == xx:
					self._Sig[x,xx] = 4 * sig_alpha + sig_beta \
						+ sig_zeta + matern[x,xx]
				elif x > xx:
					self._Sig[x,xx] = self._Sig[xx,x]
				else:
					count_equal = 0
					for i in range(12):
						if Z[x,i]==1 and Z[xx,i]==1:
							count_equal += 1
					self._Sig[x,xx] = count_equal*sig_alpha + \
						sig_zeta + matern[x,xx] 
		
	def setPrior_alt(self, mu, Sig):
		self._mu = mu
		self._Sig = Sig

	def updatePosterior(self, y, x):
		'''
		This function updates the posterior distribution of solubility
		Input: 
			y : observed solubility
			x : the compound for which y is taken, x={0,..,134}
		'''
		base_vec = np.zeros(self._dim)
		base_vec[x] = 1.
		mu_new = self._mu + \
				(y - self._mu[x])/(self._err[x]+self._Sig[x,x]) * \
				(self._Sig).dot(base_vec)
		cov_vec = (self._Sig).dot(base_vec)
		Sig_new = self._Sig - np.outer(cov_vec, cov_vec) / (self._Sig[x,x]+self._err[x])
		self._mu = mu_new
		self._Sig = Sig_new
		self._max = max(self._max, y)
		self._obs = np.append(self._obs, y)


	def getNextSample(self):
		'''
		This function determines the next 
		'''
		EI_list = np.zeros(self._dim)
		for x in range(self._dim):
			u = self._mu[x]
			v = self._Sig[x,x]		
			if v <= 0:
				EI = 0.
			else:
				z = (u-self._max)/np.sqrt(v)
				EI = (u - self._max) * norm.cdf(z) + np.sqrt(v) * norm.pdf(z)
			EI_list[x] = EI
		return np.nanargmax(EI_list)
		# return np.nanargmax(EI_list), np.nanmax(EI_list) 





	
