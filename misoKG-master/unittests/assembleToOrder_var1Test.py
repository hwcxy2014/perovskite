import unittest
import numpy

import sys

sys.path.append("../")
from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var1 import AssembleToOrderVar1

'''
Testclass for the ATO Simulator AssembleToOrderVar1.
'''

__author__ = 'matthiaspoloczek'

class assembleToOrderVar1Test(unittest.TestCase):
    def test(self):
        ato = AssembleToOrderVar1( )

        x = numpy.array([0, 0, 0, 0, 0, 0, 0, 0])
        #print ato.evaluate(0,x)
        self.assertAlmostEqual(ato.evaluate(0,x), 0.0, delta=3.0)

        x = numpy.array([19.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        self.assertEqual(ato.noise_and_cost_func(0,x),(0.332, (11.422 - 7.5)))

        xstring = "[19.0 17.0 14.0 20.0 16.0 13.0 17.0 15.0 ]"
        self.assertEqual(ato.convertXtoString(x),xstring)

        #print ato.evaluate(0,x)
        self.assertAlmostEqual(ato.evaluate(0,x), 18.7, delta=3.0)

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

        x = numpy.array([12.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        print ato.evaluate(0,x)
        #self.assertAlmostEqual(ato.evaluate(0,x), 26.0, delta=4.0)


        # ato2 = AssembleToOrder(numIS=4)
        # ato2.estimateVarianceAndCost(2,10)


if __name__ == '__main__':
    unittest.main()

