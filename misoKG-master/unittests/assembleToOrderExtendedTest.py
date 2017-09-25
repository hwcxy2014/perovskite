import sys
import unittest

import numpy

sys.path.append("../")
from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended

'''
Testclass for the two ATO Simulators in AssembleToOrderExtended.
'''

__author__ = 'matthiaspoloczek'

class assembleToOrderExtendedTest(unittest.TestCase):
    def test(self):
        ato = AssembleToOrderExtended( )

        self.assertEqual(ato.getRunlength(0),500)
        self.assertEqual(ato.getRunlength(1),10)
        self.assertEqual(ato.getRunlength(2),100)
        self.assertEqual(ato.getBestInitialObjValue(), -numpy.inf )
        listIS = [0,1,2]
        self.assertTrue((ato.getList_IS_to_query() == listIS))
        self.assertEqual('atoext', ato.getFuncName())
        # for IS in ato.getList_IS_to_query():
        #     self.assertEqual(ato.getMeanValue(IS), 17.049)

        self.assertEqual(8, ato.getDim())

        x = numpy.array([19.0, 17.0, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])
        self.assertEqual(ato.noise_and_cost_func(0,x), (0.056, (24.633 - 7.5)))
        self.assertEqual(ato.noise_and_cost_func(1,x), (2.944, (8.064 - 7.5)))
        self.assertEqual(ato.noise_and_cost_func(2,x), (0.332, (11.422 - 7.5)))

        xstring = "[19.0 17.0 14.0 20.0 16.0 13.0 17.0 15.0 ]"
        self.assertEqual(ato.convertXtoString(x),xstring)

        # print 'ato.evaluate(1,x)='+str(ato.evaluate(1,x))
        # print 'ato.evaluate(1,x)='+str(ato.evaluate(1,x))
        # print 'ato.evaluate(1,x)='+str(ato.evaluate(1,x))
        # print 'ato.evaluate(1,x)='+str(ato.evaluate(1,x))
        # print 'ato.evaluate(1,x)='+str(ato.evaluate(1,x))
        # exit(0)
        self.assertAlmostEqual(ato.evaluate(2,x), 13.0 - ato.getMeanValue(2), delta=3.0)
        self.assertAlmostEqual(ato.evaluate(1,x), -37.0 - ato.getMeanValue(1), delta=3.0)
        self.assertAlmostEqual(ato.evaluate(0,x), 13.0 - ato.getMeanValue(0), delta=3.0)

        x3 = numpy.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
        # print '(20,...,20) = ' + str(ato.evaluate(1,x3))
        self.assertAlmostEqual(-94.0, ato.evaluate(1,x3) + ato.getMeanValue(1), delta=10.0)
        self.assertAlmostEqual(-36.0, ato.evaluate(2,x3) + ato.getMeanValue(2), delta=10.0)

        x4 = numpy.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        self.assertAlmostEqual(65.0, ato.evaluate(1,x4) + ato.getMeanValue(1), delta=10.0)
        self.assertAlmostEqual(78.0, ato.evaluate(2,x4) + ato.getMeanValue(2), delta=10.0)

        atoneg = AssembleToOrderExtended( mult = -1.0 )
        self.assertAlmostEqual(-78.0 + ato.getMeanValue(2), atoneg.evaluate(2,x4), delta=10.0)

        for i in xrange(ato.getDim()):
           self.assertAlmostEqual(ato.getNumInitPtsAllIS()[i],26.,delta=0.1)

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


        ### Test of ensureBoundaries()
        x2 = numpy.array([-0.000010, 20.0000000001, 14.0, 20.0, 16.0, 13.0, 17.0, 15.0])

        ato.ensureBoundaries(x2)
        for i in xrange(ato.getDim()):
            self.assertGreaterEqual(x2[i],0.0)
            self.assertLessEqual(x2[i],20.0)

if __name__ == '__main__':
    unittest.main()

