import unittest
import numpy

import sys
sys.path.append("../dragAndLift/")
sys.path.append("../")
from dragAndLift import DragAndLift

__author__ = 'matthiaspoloczek'

'''
Testclass for the AeroFoil Setup DragAndLift.
'''



class DragAndLiftTest(unittest.TestCase):
    def test(self):
        draglift = DragAndLift( mult = -1.0 )
        self.assertEqual(draglift.getBestInitialObjValue(), +numpy.inf )
        self.assertEqual(draglift.getDim(), 2)
        self.assertEqual(draglift.getNumIS(), 4)


        self.assertAlmostEqual( draglift._search_domain[0][0], 0.1, delta=0.00001 )
        self.assertAlmostEqual( draglift._search_domain[0][1], 0.5, delta=0.00001 )
        self.assertAlmostEqual( draglift._search_domain[1][0], 0.01, delta=0.00001 )
        self.assertAlmostEqual( draglift._search_domain[1][1], 8.0, delta=0.00001 )

        randompoint = draglift.drawRandomPoint()
        self.assertEqual(len(randompoint), 2)
        self.assertGreaterEqual(randompoint[0], 0.1)
        self.assertLessEqual(randompoint[0], 0.5)
        self.assertGreaterEqual(randompoint[1], 0.01)
        self.assertLessEqual(randompoint[1], 8.0)
        # print randompoint

        # cost and noise are uniform for this scenario
        self.assertEqual(draglift.noise_and_cost_func(0, randompoint) , (-1., -1.) )
        self.assertEqual(draglift.noise_and_cost_func(1, randompoint) , (10., 1.) )
        self.assertEqual(draglift.noise_and_cost_func(2, randompoint) , (10., 1.) )
        self.assertEqual(draglift.noise_and_cost_func(3, randompoint) , (1., 3600.) )
        self.assertEqual(draglift.noise_and_cost_func(4, randompoint) , (1., 3600.) )

        # test of evaluate()
        draglift._previousQueriesXFOIL[ (0.1,1.) ] = [ [1.1, 1.2, 1.3], [0.4, 0.3, 0.2] ]
        draglift._previousQueriesSU2[ (0.1,1.) ] = [ [0.1, 0.2, 0.3], [0.41, 0.31, 0.21] ]

        self.assertEqual(draglift.evaluate(1, (0.1, 1.0) ), (draglift._mult) * 1.1)
        self.assertEqual(draglift.evaluate(2, (0.1, 1.0) ), (draglift._mult) * 0.4)
        self.assertEqual(draglift.evaluate(1, (0.1, 1.0) ), (draglift._mult) * 1.2)
        self.assertEqual(draglift.evaluate(2, (0.1, 1.0) ), (draglift._mult) * 0.3)
        self.assertEqual(draglift.evaluate(1, (0.1, 1.0) ), (draglift._mult) * 1.3)
        # print draglift.evaluate(1, (0.1, 1.0) )
        # print draglift.evaluate(1, (0.1, 1.0) )
        self.assertAlmostEqual(draglift.evaluate(1, (0.1, 1.0) ), (draglift._mult) * 15.74, delta=0.001)
        self.assertAlmostEqual(draglift.evaluate(1, (0.1, 1.0) ), (draglift._mult) * 15.74, delta=0.001)
        self.assertEqual(draglift.evaluate(2, (0.1, 1.0) ), (draglift._mult) * 0.2)
        #print draglift.evaluate(2, (0.1, 1.0) )
        #print draglift.evaluate(2, (0.1, 1.0) )
        self.assertAlmostEqual(draglift.evaluate(2, (0.1, 1.0) ), (draglift._mult) * 0.2639, delta=0.001)
        self.assertAlmostEqual(draglift.evaluate(2, (0.1, 1.0) ), (draglift._mult) * 0.2639, delta=0.001)

        self.assertEqual(draglift.evaluate(3, (0.1, 1.0) ), (draglift._mult) * 0.1)
        self.assertEqual(draglift.evaluate(4, (0.1, 1.0) ), (draglift._mult) * 0.41)
        self.assertEqual(draglift.evaluate(3, (0.1, 1.0) ), (draglift._mult) * 0.2)
        self.assertEqual(draglift.evaluate(4, (0.1, 1.0) ), (draglift._mult) * 0.31)

        #
        # for i in xrange(draglift.getDim()):
        #    self.assertAlmostEqual(draglift.getNumInitPtsAllIS()[i],20.,delta=0.1)
        #
        # x = draglift.drawRandomPoint()
        # for i in xrange(draglift.getDim()):
        #     self.assertGreaterEqual(x[i],0.0)
        #     self.assertLessEqual(x[i],20.0)
        #
        # xa =  numpy.zeros(8)
        # xb =  numpy.zeros( (8, 5) )
        # for i in xrange(8):
        #     xa[i] = i
        #     xb[i][1] = i
        # self.assertAlmostEqual(numpy.mean(xa),3.5, delta=0.1)
        # self.assertAlmostEqual(numpy.mean(xb,axis=0)[1], 3.5, delta=0.1)
        # self.assertAlmostEqual(numpy.mean(xb,axis=0)[0], 0., delta=0.1)



if __name__ == '__main__':
    unittest.main()