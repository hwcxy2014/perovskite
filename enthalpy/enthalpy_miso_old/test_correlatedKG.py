import unittest
import correlatedKG as ckg
import scipy.stats as ss
from math import log
class test_logEI(unittest.TestCase):
	def test(self):
		s = [-20,1,0.1,-12]
		logy_true = [0 for i in range(4)]
		logy_true[0] = log(ss.norm.pdf(s[0]))-log((pow(s[0],2)+1))
		logy_true[3] = log(ss.norm.pdf(s[3]))-log((pow(s[3],2)+1))
		logy_true[1] = log(s[1]*ss.norm.cdf(s[1])+ss.norm.pdf(s[1]))
		logy_true[2] = log(s[2]*ss.norm.cdf(s[2])+ss.norm.pdf(s[2]))
		logy_test = ckg.logEI(s)
		self.assertTrue(logy_test==logy_true)
if __name__ == '__main__':
	unittest.main()


