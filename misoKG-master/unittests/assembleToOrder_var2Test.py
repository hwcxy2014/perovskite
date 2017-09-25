import unittest
import numpy

import sys

sys.path.append("../")
from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from assembleToOrder.assembleToOrder import AssembleToOrder
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4


'''
Testclass for the ATO Simulator AssembleToOrder_vanilla.
'''

__author__ = 'matthiaspoloczek'

class assembleToOrder_vanillaTest(unittest.TestCase):
    def test(self):
        ato = AssembleToOrderVar2( )
        print ato.getFuncName()
        meanObjValue = ato._meanValue
        self.assertAlmostEqual(meanObjValue, 14.996, 0.00001)
        self.assertAlmostEqual(ato.estimateMeanFromPickles(), 0.0, delta=3.0)
        #ato.estimateVariance( )

        x = numpy.array([0, 0, 0, 0, 0, 0, 0, 0])
        #print ato.evaluate(0,x)
        self.assertAlmostEqual(ato.evaluate(0,x), 0.0 - meanObjValue, delta=3.0)
        self.assertEqual(ato.noise_and_cost_func(0,x),(0.85, (11.422 - 7.5)))

        x = numpy.array([19.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        self.assertEqual(ato.noise_and_cost_func(0,x),(0.85, (11.422 - 7.5)))

        xstring = "[19.0 17.0 14.0 20.0 16.0 13.0 17.0 15.0 ]"
        self.assertEqual(ato.convertXtoString(x),xstring)

        # print ato.evaluate(0,x)
        # self.assertAlmostEqual(ato.evaluate(0,x), 9.6, delta=3.0)

        x = numpy.array([12.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        print ato.evaluate(0,x)
        self.assertAlmostEqual(ato.evaluate(0,x), 19.6 - meanObjValue, delta=5.0)

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


if __name__ == '__main__':
    unittest.main()

