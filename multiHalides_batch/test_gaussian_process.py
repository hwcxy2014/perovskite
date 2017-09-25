import unittest
import gaussian_process as gp
import numpy as np

class testComputeBatchEI(unittest.TestCase):
	def test(self):
		qq = 4
		mu = [0.1,0.2,0.3,0.4]
		sig = np.identity(qq)
		print gp.computeBatchEI(4,mu,sig,0.3)
		self.assertTrue(True)

class testGetSig(unittest.TestCase):
	def test(self):
		pts = [1,3]
		allSize = 6
		sig = np.identity(allSize)
		result = gp.getSig(pts,allSize,sig)
		print result
		self.assertTrue(True)
if __name__ == '__main__':
	    unittest.main()
