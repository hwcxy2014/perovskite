import unittest
import numpy

import sys

sys.path.append("../")
from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_shifted import RosenbrockShifted
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted

'''
Testclass for the Rosenbrock Family.
'''

__author__ = 'matthiaspoloczek'

class rosenbrockFamilyTest(unittest.TestCase):
    def test(self):

        # all outputs are noisy:
        allowed_delta=2.5

        '''
        Test of RosenbrockVanilla
        '''
        for mult in [1.0, -1.0]:
            rb_vanilla = RosenbrockVanilla(mult=mult)
            self.assertEqual(rb_vanilla.getFuncName(), 'rbC_vanN')

            self.assertAlmostEqual(rb_vanilla.estimateMeanFromPickles(), 0.0, delta=100.0)

            x = numpy.array([0.0,0.0])
            self.assertEqual(rb_vanilla.noise_and_cost_func(0,x),(0.25, 1000.))
            self.assertAlmostEqual(rb_vanilla.evaluate(0, x), mult * (-9.0 - rb_vanilla.getMeanValue(0) ), delta=allowed_delta)

            x = numpy.array([1.0,2.0])
            self.assertAlmostEqual(rb_vanilla.evaluate(0, x), mult * (90.0 - rb_vanilla.getMeanValue(0) ), delta=allowed_delta)

            self.assertEqual(rb_vanilla.getNumIS(), 1)
            self.assertEqual(rb_vanilla.getTruthIS(), 0)
            self.assertEqual(rb_vanilla.getDim(), 2)
            x = rb_vanilla.drawRandomPoint()
            for i in xrange(rb_vanilla.getDim()):
                self.assertGreaterEqual(x[i], -2.0)
                self.assertLessEqual(x[i], 2.0)

            x = numpy.array([-2.0000001, 2.0])
            self.assertTrue(x[0] < -2.0)
            rb_vanilla.ensureBoundaries(x)
            self.assertTrue(x[0] >= -2.0)
            self.assertTrue(x[0] <= 2.0)
            self.assertTrue(x[1] >= -2.0)
            self.assertTrue(x[1] <= 2.0)

        '''
        Test of RosenbrockSinus
        '''
        for mult in [1.0, -1.0]:
            rb_sinus = RosenbrockSinus(mult=mult)

            self.assertEqual(rb_sinus.getFuncName(), 'rbC_sinN')

            # meanstr = str(rb_sinus.estimateMeanFromPickles())
            # print "mean=" + meanstr
            self.assertAlmostEqual(rb_sinus.estimateMeanFromPickles(), 0.0, delta=100.0)

            x = numpy.array([0.0,0.0])
            self.assertEqual(rb_sinus.noise_and_cost_func(0,x),(0.25, 1000.0))
            self.assertAlmostEqual(rb_sinus.evaluate(0, x), mult * (-9.0 - rb_sinus.getMeanValue(0) ), delta=allowed_delta)

            x = numpy.array([1.0,2.0])
            self.assertAlmostEqual(rb_sinus.evaluate(0, x), mult * ( (90.0 + numpy.sin(20)) - rb_sinus.getMeanValue(0)), delta=allowed_delta)

            self.assertEqual(rb_sinus.getNumIS(), 1)
            self.assertEqual(rb_sinus.getTruthIS(), 0)
            self.assertEqual(rb_sinus.getDim(), 2)
            x = rb_sinus.drawRandomPoint()
            for i in xrange(rb_sinus.getDim()):
                self.assertGreaterEqual(x[i], -2.0)
                self.assertLessEqual(x[i], 2.0)

        # the following test might fail occasionally due to random noise

        '''
        Test of RosenbrockBiased
        '''

        for mult in [1.0, -1.0]:
            rb_biased = RosenbrockBiased(mult=mult)
            self.assertEqual(rb_biased.getFuncName(), 'rbC_biasN')

            # meanstr = str(rb_biased.estimateMeanFromPickles())
            # print "mean=" + meanstr
            self.assertAlmostEqual(rb_biased.estimateMeanFromPickles(), 0.0, delta=100.0)


            x = numpy.array([0.0,0.0])
            self.assertEqual(rb_biased.noise_and_cost_func(0,x),(0.25, 1000.0))
            self.assertAlmostEqual(rb_biased.evaluate(0, x), mult* ( (-9.0 + 0.01 * x[0]) - rb_biased.getMeanValue(0) ), delta=allowed_delta)

            x = numpy.array([1.0,2.0])
            self.assertAlmostEqual(rb_biased.evaluate(0, x),
                                   mult* ( (90.0 + numpy.sin(20) + 0.01 * x[0]) - rb_biased.getMeanValue(0) ),
                                   delta=allowed_delta)

            self.assertEqual(rb_biased.getNumIS(), 1)
            self.assertEqual(rb_biased.getTruthIS(), 0)
            self.assertEqual(rb_biased.getDim(), 2)
            x = rb_biased.drawRandomPoint()
            for i in xrange(rb_biased.getDim()):
                self.assertGreaterEqual(x[i], -2.0)
                self.assertLessEqual(x[i], 2.0)

        '''
        Test of RosenbrockShifted
        '''
        for mult in [1.0, -1.0]:
            rb_shifted = RosenbrockShifted(mult=mult)

            self.assertEqual(rb_shifted.getFuncName(), 'rbC_shiftedN')

            # meanstr = str(rb_shifted.estimateMeanFromPickles())
            # print "mean=" + meanstr
            self.assertAlmostEqual(rb_shifted.estimateMeanFromPickles(), 0.0, delta=100.0)

            x = numpy.array([-0.1,-0.05])
            self.assertEqual(rb_shifted.noise_and_cost_func(0,x),(0.25, 1000.0))
            self.assertAlmostEqual(rb_shifted.evaluate(0, x),
                                   mult* ( (-9.0 + 0.01 * (x[0]+0.1)) - rb_shifted.getMeanValue(0) ),
                                   delta=allowed_delta)

            x = numpy.array([0.9,1.95])
            self.assertAlmostEqual(rb_shifted.evaluate(0, x), mult* ( (90.0 + numpy.sin(20) + 0.01 * (x[0]+0.1)) - rb_shifted.getMeanValue(0) ), delta=allowed_delta)

            self.assertEqual(rb_shifted.getNumIS(), 1)
            self.assertEqual(rb_shifted.getTruthIS(), 0)
            self.assertEqual(rb_shifted.getDim(), 2)
            x = rb_shifted.drawRandomPoint()
            for i in xrange(rb_shifted.getDim()):
                self.assertGreaterEqual(x[i], -2.0)
                self.assertLessEqual(x[i], 2.0)


        '''
        Test of RosenbrockSlightShifted
        '''
        for mult in [1.0, -1.0]:
            rb_slightshifted = RosenbrockSlightShifted(mult=mult)
            self.assertEqual(rb_slightshifted.getFuncName(), 'rbC_slshN')

            # meanstr = str(rb_slightshifted.estimateMeanFromPickles())
            # print "mean=" + meanstr
            self.assertAlmostEqual(rb_slightshifted.estimateMeanFromPickles(), 0.0, delta=100.0)

            x = numpy.array([0.0 - 0.01, 0.0 + 0.005])
            self.assertEqual(rb_slightshifted.noise_and_cost_func(0,x),(0.25, 1000.0))
            self.assertAlmostEqual(rb_slightshifted.evaluate(0, x), mult * (-9.0 - rb_slightshifted.getMeanValue(0) ), delta=allowed_delta)

            x = numpy.array([1.0 - 0.01, 2.0 + 0.005])
            self.assertAlmostEqual(rb_slightshifted.evaluate(0, x), mult * (90.0 - rb_slightshifted.getMeanValue(0) ), delta=allowed_delta)

            self.assertEqual(rb_slightshifted.getNumIS(), 1)
            self.assertEqual(rb_slightshifted.getTruthIS(), 0)
            self.assertEqual(rb_slightshifted.getDim(), 2)
            x = rb_slightshifted.drawRandomPoint()
            for i in xrange(rb_slightshifted.getDim()):
                self.assertGreaterEqual(x[i], -2.0)
                self.assertLessEqual(x[i], 2.0)


        # for i in xrange(rb_vanilla.getDim()):
        #    self.assertAlmostEqual(rb_vanilla.getNumInitPtsAllIS()[i],20.,delta=0.1)
        #
        # x = rb_vanilla.drawRandomPoint()
        # for i in xrange(rb_vanilla.getDim()):
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