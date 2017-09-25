import unittest
import numpy as np
import compute_KG as t

class test_get_acceptance_set(unittest.TestCase):
	def test1(self):
		a = [1,4,5]
		b = [2,3,4]
		c,A = t.get_acceptance_set(a,b)
		print c,A
		c_true = [-1000,-3.,-1.,1000] #computed by wolfram
		A_true = [1,2,3]
		self.assertTrue(c==c_true)
		self.assertTrue(A==A_true)

class test_get_sorted(unittest.TestCase):
	def test1(self):
		a = [2,1,6,0,4,1,2]
		b = [1,3,4,5,3,2,5]
		aout,bout = t.get_sorted(a,b)
		print aout,bout
		a_true = [2,1,4,6,2]
		b_true = [1,2,3,4,5]
		self.assertTrue(a_true,aout)
		self.assertTrue(b_true,bout)
	def test2(self):
		a = [2,2,1,2,2]
		b = [1,4,1,1,1]
		aout,bout = t.get_sorted(a,b)
		print aout,bout
		a_true = [2,2]
		b_true = [1,4]
		self.assertTrue(a_true,aout)
		self.assertTrue(b_true,bout)

if __name__ == "__main__":
	unittest.main()
