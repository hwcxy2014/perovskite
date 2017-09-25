import stats_model as sm
import numpy as np

class singleHalide_BU(object):
	def __init__(self,n,ns):
		# inputs:
		#	n : no. of solution combs
		#	ns: no. of information source
	
		self._dim = n*ns
		self._nSource = ns
		self._cost = [1. for i in range(self._nSource)]
		self._obs = np.array([[] for i in range(self._nSource)])
		self._model = 'single'
		self._max = [-100.,-100]

		self._mu = [0. for i in range(self._dim)]
		self._Sig = np.identity(self._dim)		

	def setCost(self,c,inso):
		# inputs:
		#	c : time cost per sample 
		#	inso : information source
		self._cost[inso] = c
	
	def setPrior(self,x,mu_alpha,sig_alpha,mu_zeta,\
				sig_zeta,sig_beta,l0,sig_m0,l1,sig_m1):
		# inputs:
		#   mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta -\
		#	hypes, \in R
		#   l0, sig_m0 - hypes of gp associated with the \
		#       solvenets in the solutions. l0 \in R^2,
		#   l1, sig_m1 - hypes of gp associated with the \
		#         secondary source(umbo). l1 \in R^8 or R^14

		self._mu,self._Sig=sm.prior_bu(x,self._model,mu_alpha,\
					sig_alpha,mu_zeta,sig_zeta,sig_beta,l0,\
					sig_m0,l1,sig_m1)

	def updatePosterior(self, y, x, sn):
		#This function updates the posterior distribution
		#Input: 
		#	y : observed value
		#	x : the compound for which y is taken, x={0,..,71}
		#	sn: info source number
		if self._model == 'single':
			x = x+sn*72
		else:
			x = x+sn*648
		base_vec = np.zeros(self._dim)
		base_vec[x] = 1.
		mu_new = self._mu + \
				(y - self._mu[x])/self._Sig[x,x] * \
				(self._Sig).dot(base_vec)
		cov_vec = (self._Sig).dot(base_vec)
		Sig_new = self._Sig - np.outer(cov_vec, cov_vec) / (self._Sig[x,x])
		self._mu = mu_new
		self._Sig = Sig_new
		self._max[sn] = max(self._max[sn], y)
		self._obs[sn] = np.append(self._obs[sn], y)
		
