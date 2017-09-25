import unittest
import numpy as np
import scipy
import scipy.stats
np.set_printoptions(threshold=np.nan)
import gaussian_process
import hyper_parameters

class test_GP(unittest.TestCase):
	def test(self):
		'''
		--- Generate samples
		'''
		# define hyperparameters
		l1 = 0.2
		l2 = 0.5
		sig_m = 1.5
		mu_alpha = 2
		sig_alpha = 1.
		mu_zeta = 0.5
		sig_zeta = 1.2
		sig_beta = 0.5
		n = 135 # total number of points
		m = 10  # number of solutions with solubilities observed
		# generate solutions
		cations = np.array([1,0,0])
		halides = np.array([1,0,0])
		solvents = np.array([np.random.random(2) for i in range(15)]) 
		data = np.zeros([n, 10])
		for i in range(n):
			# set cation
			data[i,:3] = np.random.permutation(cations)
			data[i,3:6] = np.random.permutation(halides)
			sol_n = int(np.floor(np.random.random()*15))
			data[i,6:8] = solvents[sol_n,:]
			data[i,8] = sol_n
		# calculate mean and variance based on hyperparameters
		mu = [2*mu_alpha + mu_zeta for x in range(n)]
		matern = hyper_parameters.maternKernel(data[:,6:8],l1,l2,sig_m)
		Sig = np.identity(n)
		for x in range(n):
			for xx in range(n):
				if x==xx:
					Sig[x,xx] = 2 * sig_alpha + sig_beta + sig_zeta\
								+ matern[x,xx]
				elif x > xx:
					Sig[x,xx] = Sig[xx,x]
				else:
					count_equal = 0
					for i in range(6):
						if data[x,i] == 1 and data[xx,i] == 1:
							count_equal += 1
						Sig[x,xx] = count_equal*sig_alpha + sig_zeta \
									+ matern[x,xx]
		# generate random sample for solubilities
		solubilities = scipy.stats.multivariate_normal.rvs(mu, Sig)
		data[:,9] = solubilities
	
		'''
		--- initialize gaussian process
		'''
		gp = gaussian_process.solubilities(n)
	
		'''
		--- set Prior
		'''	
		gp.setPrior(data[:,0:9],mu_alpha,sig_alpha,mu_zeta,\
					sig_zeta,sig_beta,sig_m,l1,l2)
		self.assertEquals(np.isclose(mu, gp._mu).all(), True)
		self.assertEquals(np.isclose(Sig, gp._Sig).all(),True)
		
		'''
		--- update using historical data
		--- assume the first m samples are observed
		'''
		for i in range(m):
			gp.updatePosterior(data[i,9],i)	
		'''
		--- get where to take the next sample 
		'''
		next_sample = gp.getNextSample()
		print next_sample
		print data[:,9]
class test_updatePosterior(unittest.TestCase):
	def test(self):
		mu = np.array([0.1,0.1])
		Sig = np.array([[2,0.5],[0.5,2]])
		y = 0.2
		x = 1
		gp = gaussian_process.solubilities(2)
		gp.setPrior_alt(mu, Sig)
		gp.updatePosterior(y,x)
		print gp._mu
		print gp._Sig
		true_mu = np.array([[0.125,0.2]])
		self.assertEquals(np.isclose(gp._mu, true_mu).all(),True)
		self.assertEquals(gp._Sig[0,0],1.875)
		self.assertEquals(gp._Sig[1,1],0.)
if __name__ == '__main__':
	unittest.main()
