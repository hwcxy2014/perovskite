import unittest

import sys
sys.path.append("../")

from misoLogRegMnistUsps.logreg_mnistusps import LogRegMNISTUSPS, LogRegMNISTUSPS2, LogRegMNISTUSPS3, LogRegMNISTUSPS4

__author__ = 'matthiaspoloczek'

'''
Testclass for the Logistic Regression
'''

class LogRegMNISTUSPSTest(unittest.TestCase):
    def test(self):

        lr = LogRegMNISTUSPS( )
        self.assertEqual('lrMU', lr.getFuncName())
        x=[0.1435,0.0,206,685] # best point from SSA'13
        # self.assertAlmostEqual(0.071905, lr.evaluate(0,x), delta=0.001) # true valid error
        self.assertAlmostEqual(-1.4872, lr.evaluate(0,x), delta=0.01) # centered logit of testerror 7.847896

        x_usps = [0.0002, 0.0032, 560, 161]
        # self.assertAlmostEqual(0.067261, lr.evaluate(1,x_usps), delta=0.001) # true valid error
        self.assertAlmostEqual(-1.6535, lr.evaluate(1,x_usps), delta=0.01) # centered logit of testerror

        lr = LogRegMNISTUSPS2( )
        self.assertEqual('lrMU2', lr.getFuncName())
        self.assertAlmostEqual(-1.4872, lr.evaluate(0,x), delta=0.01) # centered logit of testerror 7.847896
        self.assertAlmostEqual(-1.6535 -0.97599 - 0.01502, lr.evaluate(1,x_usps), delta=0.01) # centered logit of testerror

        lr = LogRegMNISTUSPS3( )
        self.assertEqual('lrMU3', lr.getFuncName())
        self.assertAlmostEqual(-1.6535 -0.97599 - 0.01502, lr.evaluate(1,x_usps), delta=0.01) # centered logit of testerror
        self.assertAlmostEqual(43.69, lr.noise_and_cost_func(0,x_usps)[1], delta=0.001)
        self.assertAlmostEqual(2.18, lr.noise_and_cost_func(1,x_usps)[1], delta=0.001)

        lr = LogRegMNISTUSPS4( )
        self.assertEqual('lrMU4', lr.getFuncName())
        self.assertAlmostEqual(-1.6535 -0.97599 - 0.01502, lr.evaluate(1,x_usps), delta=0.01) # centered logit of testerror
        self.assertAlmostEqual(43.69, lr.noise_and_cost_func(0,x_usps)[1], delta=0.001)
        self.assertAlmostEqual(4.5, lr.noise_and_cost_func(1,x_usps)[1], delta=0.001)

if __name__ == '__main__':
    unittest.main()