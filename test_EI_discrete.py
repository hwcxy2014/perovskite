import unittest
import numpy
import EI_discrete

class test_gaussianInference(unittest.TestCase):
	def test(self):
		u0 = numpy.zeros(10)
		s0 = numpy.identity(10)
		fn = numpy.array([numpy.random.random() for x in range(9)])
		self.assertEqual(EI_discrete.gaussianInference(u0,s0,fn),(0,1))

class test_getEI(unittest.TestCase):
	def test(self):
		self.assertEqual(EI_discrete.getEI(),1)

if __name__ == '__main__':
	unittest.main()
