import unittest
import numpy

import sys
sys.path.append("../")
from mothers_little_helpers import *


__author__ = 'matthiaspoloczek'


'''
Testclass for the Mother's Little Helpers.
'''

class mothersLittleHelpersTest(unittest.TestCase):
    def test(self):

        func_name = 'pickleTest'
        directory = "picklesTest"
        numIS = 4

        init_points_for_all_IS = numpy.ones((2,3))
        pickle_init_points_for_all_IS(directory, func_name, numIS, init_points_for_all_IS)

        init_points_for_all_IS2 = load_init_points_for_all_IS(directory, func_name,numIS)

        self.assertTrue( (init_points_for_all_IS == init_points_for_all_IS2).all )
        # https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise

        '''
        Test overwriting existing files
        '''
        init_points_for_all_IS = numpy.ones((4,3,2))
        pickle_init_points_for_all_IS(directory, func_name, numIS, init_points_for_all_IS)
        init_points_for_all_IS2 = load_init_points_for_all_IS(directory, func_name,numIS)
        self.assertTrue( (init_points_for_all_IS == init_points_for_all_IS2).all )

        best_initial_value = 2.3
        pickle_best_initial_value(directory, func_name, numIS, best_initial_value)
        best_initial_value2 = load_best_initial_value(directory, func_name,numIS)
        self.assertEqual(best_initial_value,best_initial_value2)

        vals = numpy.ones((4,3,2))
        pickle_vals(directory, func_name,numIS,vals)
        vals2 = load_vals(directory, func_name,numIS)
        self.assertTrue( (vals == vals2).all )

        svals = numpy.ones((4,3,2))
        pickle_sample_vars(directory, func_name,numIS,svals)
        svals2 = load_sample_vars(directory, func_name,numIS)
        self.assertTrue( (svals == svals2).all )

        # array1 = numpy.array([250, 250, 250, 250])
        # num_pts_to_gen = numpy.repeat( 250, 4)
        # self.assertTrue( (array1 == num_pts_to_gen).all )




if __name__ == '__main__':
    unittest.main()