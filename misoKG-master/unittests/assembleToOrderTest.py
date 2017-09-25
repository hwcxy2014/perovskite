import unittest
import numpy

import sys
sys.path.append("../")
from assembleToOrder.assembleToOrder import AssembleToOrder

'''
Testclass for the ATO Simulator AssembleToOrder.
'''

__author__ = 'matthiaspoloczek'

class assembleToOrderTest(unittest.TestCase):
    def test(self):
        ato = AssembleToOrder(numIS=4)
        self.assertEqual(ato.getRunlength(0),-1)

        self.assertEqual(ato.getBestInitialObjValue(), -numpy.inf )

        x = numpy.array([19.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        self.assertEqual(ato.noise_and_cost_func(0,x),(-1., -1.))

        xstring = "[19.0 17.0 14.0 20.0 16.0 13.0 17.0 15.0 ]"
        self.assertEqual(ato.convertXtoString(x),xstring)

        self.assertAlmostEqual(ato.evaluate(2,x), 13.0, delta=3.0)

        for i in xrange(ato.getDim()):
           self.assertAlmostEqual(ato.getNumInitPtsAllIS()[i],20.,delta=0.1)

        x = ato.drawRandomPoint()
        for i in xrange(ato.getDim()):
            self.assertGreaterEqual(x[i],0.0)
            self.assertLessEqual(x[i],20.0)

        xa =  numpy.zeros(8)
        xb =  numpy.zeros( (8, 5) )
        for i in xrange(8):
            xa[i] = i
            xb[i][1] = i
        self.assertAlmostEqual(numpy.mean(xa),3.5, delta=0.1)
        self.assertAlmostEqual(numpy.mean(xb,axis=0)[1], 3.5, delta=0.1)
        self.assertAlmostEqual(numpy.mean(xb,axis=0)[0], 0., delta=0.1)

        # ato2 = AssembleToOrder(numIS=4)
        # ato2.estimateVarianceAndCost(2,10)


if __name__ == '__main__':
    unittest.main()

