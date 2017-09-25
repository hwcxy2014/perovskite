import unittest
import numpy

import sys

sys.path.append("../")
from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from assembleToOrder.assembleToOrder import AssembleToOrder

'''
Testclass for the ATO Simulator AssembleToOrder_vanilla.
'''

__author__ = 'matthiaspoloczek'

class assembleToOrder_vanillaTest(unittest.TestCase):
    def test(self):
        ato = AssembleToOrderVanilla( )
        meanObjValue = ato._meanValue
        self.assertAlmostEqual(meanObjValue, 17.049, 0.00001)
        self.assertAlmostEqual(ato.estimateMeanFromPickles(), 0.0, delta=3.0)

        #ato.estimateVariance( )

        x = numpy.array([0, 0, 0, 0, 0, 0, 0, 0])
        #print ato.evaluate(0,x)
        self.assertAlmostEqual(ato.evaluate(0,x), 0.0 - meanObjValue, delta=3.0)

        x = numpy.array([19.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        self.assertEqual(ato.noise_and_cost_func(0,x),(0.332, (11.422 - 7.5)))

        xstring = "[19.0 17.0 14.0 20.0 16.0 13.0 17.0 15.0 ]"
        self.assertEqual(ato.convertXtoString(x),xstring)

        self.assertAlmostEqual(ato.evaluate(0,x), 13.0 - meanObjValue, delta=3.0)

        # for i in xrange(ato.getDim()):
        #    self.assertAlmostEqual(ato.getNumInitPtsAllIS()[i],20.,delta=0.1)

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
        self.assertAlmostEqual(ato.evaluate(0,x), 26.0 - meanObjValue, delta=4.0)

        ato_original = AssembleToOrder(numIS=4)
        self.assertAlmostEqual(ato_original.evaluate(2,x), 26.0, delta=4.0)

        # taken from the kg_data object to test the sign of the value and the correct storage
        x = numpy.array([4.56878149,  14.2741282 ,   7.1434222 ,
        13.71855152,  10.24729441,   4.76192534,   2.10440107,  10.76843903])
        self.assertAlmostEqual(ato.evaluate(0, x), 82.40377 - meanObjValue, delta=4.0)
        x = numpy.array([1.55342433,   8.60075366,  11.34155703, 19.13172139,
                         14.37334116,  15.85730744,  11.81194939,   9.24025991])
        self.assertAlmostEqual(ato.evaluate(0, x), 0.17 - meanObjValue, delta=5.0)


        # ato2 = AssembleToOrder(numIS=4)
        # ato2.estimateVarianceAndCost(2,10)


if __name__ == '__main__':
    unittest.main()

