import numpy as np
import sample_trainingData
import unittest

class test_get_nearest(unittest.TestCase):
	def test(self):
		pt = [2.,2.]
		solvents = {'a':(1.,1.),
					'b':(2.,2.),
					'c':(3.,3.)}
		true_value = 'b'
		test_value = sample_trainingData.get_nearest(pt,solvents)
		self.assertEquals(true_value,test_value)

if __name__=='__main__':
	unittest.main()

