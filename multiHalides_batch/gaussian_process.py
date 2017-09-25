import numpy as np
from scipy.stats import norm
from scipy.stats import mvn

import hyper_parameters

def computeSingleEI(mu,sig,T):
	# inputs:
	#	mu - mean of the current point
	#	sig - variance of the current point
	# outputs:
	#	EI -  EI of the current point
	
	if sig <= 0:
		EI = 0.
	else:
		z = (mu-T)/np.sqrt(sig)
		EI = (mu - T) * norm.cdf(z) + np.sqrt(sig) * norm.pdf(z)
	return EI

def computeBatchEI(qq,mu,sig,T):
	# method of computing batch EI is taken from:
	# https://hal.archives-ouvertes.fr/hal-00732512/document
	# inputs: 
	#	qq - # points in the batch X_1,...,X_qq
	#	mu - mean vector of the points in the batch
	#	sig - covariance matrix of the batch
	#	T - max value observed so far
	# output:
	#	batch_ei - EI(X) = E[max_{i\in(1,..,qq)}X_i-T] 
	
	batch_ei = 0
	for k in range(qq):
		# mean of Zk
		mu_Z = [0 for i in range(qq)]
		for i in range(qq):
			if i != k:
				mu_Z[i] = mu[i] - mu[k]
			else:
				mu_Z[i] = 0. - mu[i]
		# covariance of Zk
		s_Z = np.identity(qq)
		sk = sig[k][k]
		for i in range(qq):
			for j in range(qq):
				if i > j:
					s_Z[i,j] = s_Z[j,i]
				elif i == j:
					if i != k:
						s_Z[i,j] = sig[i][i]+sk-2*sig[i][k]
					else:
						s_Z[i,j] = sk
				else:
					if j == k:
						s_Z[i,j] = -sig[i][k] + sk
					elif i == k:
						s_Z[i,j] = -sig[k][j] + sk
					else:
						s_Z[i,j] = sig[i][j]-sig[k][j]\
									-sig[i][k]+sk
		# bk
		bk = np.zeros(qq)
		bk[k] = 0. - T
		# pk
		lower = [-50]*qq # as long as this number is\
					# relatively large compared to mu.
					# in this case the objective is in \
					# the scale of 0 to 1
		pk,ik = mvn.mvnun(lower,bk,mu_Z,s_Z)
		# one term in the sum
		summant = (mu[k] - T) * pk
		for i in range(qq):
			sai = norm.pdf(bk[i],mu_Z[i],np.sqrt(s_Z[i,i]))
			cik = np.zeros(qq-1)
			count = 0
			for j in range(qq):
				if j != i:
					cik[count] = (bk[j]-mu_Z[j])- \
						(bk[i]-mu_Z[i])*s_Z[i,j]/s_Z[i,i]
					count += 1 
			sik = np.delete(s_Z,i,0)
			sik = np.delete(sik,i,1)
			lower = [-50]*(qq-1)
			phi,iphi = mvn.mvnun(lower,cik,[0.]*(qq-1),sik)
			summant += s_Z[i,k] * sai * phi  
		batch_ei += summant
	return batch_ei			

def getMean(pts,mu):
	return [mu[i] for i in pts]

def getSig(pts,allSize,sig):
	compl = [i for i in range(allSize) if i not in pts]
	subSig = np.delete(sig,compl,0)
	subSig = np.delete(subSig,compl,1)
	return subSig

class solubilities(object):
	def __init__(self, n):
		# number of compounds, in this project n = 27*3*8 = 648
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
			x : the compound for which y is taken,x = {0,..,647}
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

	
	def getNextSample(self,q):
		# inputs:
		#	q - batch size
		# output:
		#	nextBatch: the next q points to be sampled 
		
		# compute EI for each single point	
		EI_list = np.zeros(self._dim)
		for x in range(self._dim):
			EI_list[x] = computeSingleEI(self._mu[x],\
						self._Sig[x,x],self._max)
		# the point with the largest EI
		single_max = np.nanargmax(EI_list)
		if q == 1:
			return single_max
		else:
			batch = [single_max]
			for k in range(1,q):
				ptRemain = [i for i in range(self._dim) \
							if i not in batch]
				batchEI_list = np.zeros(len(ptRemain))
				for i,x in enumerate(ptRemain):
					mu = getMean(batch+[x],self._mu)
					sig = getSig(batch+[x],self._dim,self._Sig)
					batchEI_list[i] = computeBatchEI( \
									k+1,mu,sig,self._max,)
				maxPt = ptRemain[np.nanargmax(batchEI_list)]
				batch.append(maxPt)
			return batch
					



			





	
