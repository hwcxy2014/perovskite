import unittest
import stats_model as s
import numpy as np
import scipy.stats as ss

class test_maternKernal(unittest.TestCase):
	def test(self):
		x = np.random.random([4,3])
		l = [0.1,0.3,0.8]
		sig_m = 0.7
		cov_test = s.maternKernel(x,l,sig_m)
		for i in range(len(x)):
			for j in range(len(x)):
				d = np.dot(pow((x[i]-x[j]),2),l)
				dd = np.sqrt(5*d)
				cov_ij = sig_m*(1.+dd+dd**2/3.)*np.exp(-dd)
				if not np.isclose(cov_ij, cov_test[i,j],atol = 1e-6):
					print i,j,cov_ij, cov_test[i,j]
		self.assertTrue(True)

class test_mvn_pdf(unittest.TestCase):
	def test(self):
		n = 3
		y = np.random.random(n)
		Mu = np.ones(n)#np.random.random(n)
		randm = np.random.random([n,n])
		Sig = np.identity(n)#randm.dot(randm.transpose())
		pdf_test = s.mvn_pdf(y,Mu,Sig)*pow(np.pi*2,-1.5)
		pdf_true = ss.multivariate_normal.pdf(y,Mu,Sig)
		print pdf_test,pdf_true
		self.assertEqual(pdf_test,pdf_true)
if __name__=='__main__':
	unittest.main()
