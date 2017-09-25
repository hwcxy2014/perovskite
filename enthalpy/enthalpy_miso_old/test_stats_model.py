import unittest
import stats_model as sm
import numpy as np
class testMaternKernel(unittest.TestCase):
	def test(self):
		l = [0.2,0.2,0.4]
		x = [[1,1,1],[1,0,1],[0,0,0],[0,1,0]]
		sig_m = 0.5
		test_mk = sm.maternKernel(x,l,sig_m)
		true_mk = [[0. for j in range(4)] for i in range(4)]
		true_mk[0][1] = np.sqrt(0.2)
		true_mk[0][2] = np.sqrt(0.8)
		true_mk[0][3] = np.sqrt(0.6)
		true_mk[1][2] = np.sqrt(0.6)
		true_mk[1][3] = np.sqrt(0.8)
		true_mk[2][3] = np.sqrt(0.2)
		for i in range(4):
			for j in range(4):
				if i > j:
					true_mk[i][j] = true_mk[j][i]
		true_mk = [[sig_m*(1+np.sqrt(5)*true_mk[i][j]+5./3.*\
					pow(true_mk[i][j],2))*np.exp(-np.sqrt(5)*\
					true_mk[i][j]) for j in range(4)] for i \
					in range(4)]
		is_equal = True
		for i in range(4):
			for j in range(4):
				if true_mk[i][j] != test_mk[i][j]:
					print true_mk[i][j], test_mk[i][j]
					is_equal = False
		self.assertTrue(is_equal)
if __name__ == '__main__':
	unittest.main()
 

