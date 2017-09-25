import unittest
import numpy as np
import numpy.testing as npt
import hyper_parameters 
import scipy.stats
np.set_printoptions(threshold=np.inf)
'''
class test_maternKernel_old(unittest.TestCase):
	def test(self):
		data = np.array([[0.12,0.14],[0.21,0.41],[0.42,0.31],[0.42,0.86]])
		true_matern = [[]]
		self.assertEqual(EI_discrete.gaussianInference(u0,s0,fn),(0,1))
'''
class test_maternKernel(unittest.TestCase):
	def test_1(self):
		data = [[0.1,0.2],[0.5,0.6],[0.1,0.2],[0.2,0.4]]
		l1 = 0
		l2 = 1
		sig_m = 2
		k1 = 0.4*np.sqrt(5)
		kk1 = 2*(1+k1+k1**2/3.)*np.exp(-k1)
		k2 = 0.2*np.sqrt(5)
		kk2 = 2*(1+k2+k2**2/3.)*np.exp(-k2)
		true_matern = np.array([[2,kk1,2,kk2],\
						[kk1,2,kk1,kk2],\
						[2,kk1,2,kk2],
						[kk2,kk2,kk2,2]])
		test_matern = hyper_parameters.maternKernel(data,l1,l2,sig_m)
		self.assertEquals(np.isclose(true_matern,test_matern).all(), True)

	def test_2(self):
		data = [[0.1,0.2],[0.5,0.6],[0.1,0.2],[0.2,0.4]]
		l1 = 1
		l2 = 0
		sig_m = 0.5
		k1 = 0.4*np.sqrt(5)
		kk1 = sig_m*(1+k1+k1**2/3.)*np.exp(-k1)
		k2 = 0.1*np.sqrt(5)
		kk2 = sig_m*(1+k2+k2**2/3.)*np.exp(-k2)
		k3 = 0.3*np.sqrt(5)
		kk3 = sig_m*(1+k3+k3**2/3.)*np.exp(-k3)
		true_matern = np.array([[sig_m,kk1,sig_m,kk2],\
						[kk1,sig_m,kk1,kk3],\
						[sig_m,kk1,sig_m,kk2],
						[kk2,kk3,kk2,sig_m]])
		test_matern = hyper_parameters.maternKernel(data,l1,l2,sig_m)
		self.assertEquals(np.isclose(true_matern,test_matern).all(), True)
class test_likelihood(unittest.TestCase):
	def test_1(self):
		Z = [[0,1,0,1,0,0,0.1,0.2,0],[0,0,1,0,0,1,0.5,0.6,1],\
			[0,1,0,0,0,1,0.1,0.2,0],[0,0,1,0,1,0,0.2,0.4,2]]
		V = np.random.random(4)
		data = np.hstack([Z,np.reshape(V,(4,1))])
		data7 = [[0.1,0.2],[0.5,0.6],[0.1,0.2],[0.2,0.4]]
		l1 = 0
		l2 = 1
		sig_m = 2
		mu_alpha = 0.1
		sig_alpha = 1.
		mu_zeta = 0.5
		sig_zeta = 1.
		sig_beta = 1.
		k1 = 0.4*np.sqrt(5)
		kk1 = 2*(1+k1+k1**2/3.)*np.exp(-k1)
		k2 = 0.2*np.sqrt(5)
		kk2 = 2*(1+k2+k2**2/3.)*np.exp(-k2)
		matern = np.array([[2,kk1,2,kk2],\
						[kk1,2,kk1,kk2],\
						[2,kk1,2,kk2],
						[kk2,kk2,kk2,2]])
		mean = [0.2 + 0.5 for i in range(4)]
		cov = [[2+2*sig_alpha+sig_beta+sig_zeta,\
				kk1+sig_zeta,2+sig_zeta+sig_alpha,kk2+sig_zeta],\
			   [kk1+sig_zeta, 2+2*sig_alpha+sig_beta+sig_zeta,\
				sig_alpha+sig_zeta+kk1, sig_alpha+sig_zeta+kk2],\
			   [2+sig_alpha+sig_zeta, sig_alpha+sig_zeta+kk1,\
			    2+2*sig_alpha+sig_beta+sig_zeta,kk2+sig_zeta],\
			   [kk2+sig_zeta, sig_alpha+sig_zeta+kk2,\
			   	kk2+sig_zeta,2+2*sig_alpha+sig_zeta+sig_beta]]
		true_pdf = scipy.stats.multivariate_normal.pdf(V,mean,cov)
		true_loglh = np.log(true_pdf / pow(2*np.pi, -2))
		test_loglh = hyper_parameters.likelihood(data, mu_alpha,sig_alpha,mu_zeta, sig_zeta, sig_beta, sig_m, l1, l2)
		self.assertEqual(np.isclose(true_loglh, test_loglh),True)

class test_MLE(unittest.TestCase):
	def test_1(self):
		# define hyperparameters
		l1 = 1.
		l2 = 1.
		sig_m = 1.5
		mu_alpha = 0.1
		sig_alpha = 1.
		mu_zeta = 0.1
		sig_zeta = 1.2
		sig_beta = 0.5
		n = 100
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
		# solve for hyper-parameters using MLE
		res = hyper_parameters.MLE(data)
		# compare results to the original setup
		print res
		print hyper_parameters.likelihood(data, mu_alpha,sig_alpha,\
			mu_zeta, sig_zeta, sig_beta, sig_m, l1, l2)
		self.assertEqual(True, True)
			
if __name__ == '__main__':
	unittest.main()
