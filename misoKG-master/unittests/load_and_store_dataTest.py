import unittest
import numpy

import sys

sys.path.append("../")
from load_and_store_data import *

from moe.optimal_learning.python.data_containers import HistoricalData
from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended


__author__ = 'matthiaspoloczek'


'''
Testclass for load_and_store_data
'''

class load_and_store_dataTest(unittest.TestCase):
    def test(self):

        rb = RosenbrockVanilla( )
        func_name = rb.getFuncName()

        pathToPickles = 'picklesTest'



        ### Test load_data_from_a_min_problem()
        name_testfile = 'load_and_store_Test'
        samples = numpy.array( [ [[1,1], [1,2]] ] )
        #print samples
        values = [ [1.0 , 2.0] ]
        data = { "points": samples, "vals": values }

        with open("{0}/{1}.pickle".format(pathToPickles, name_testfile), "wb") as output_file:
            pickle.dump(data, output_file)
        loaded_pts, loaded_vals = load_data_from_a_min_problem(pathToPickles, name_testfile)
        for index in range(len(samples)):
            self.assertTrue( (samples[index] == loaded_pts[index]).all )
        for index in range(len(values)):
            self.assertTrue( (values[index] == loaded_vals[index]) )

        # test overwriting
        samples = numpy.array( [ [[1,4], [1,2]] ] )
        with open("{0}/{1}.pickle".format(pathToPickles, name_testfile), "wb") as output_file:
            pickle.dump(data, output_file)
        loaded_pts, loaded_vals = load_data_from_a_min_problem(pathToPickles, name_testfile)
        for index in range(len(samples)):
            self.assertTrue( (samples[index] == loaded_pts[index]).all )
        for index in range(len(values)):
            self.assertTrue( (values[index] == loaded_vals[index]) )



        ### Test obtainHistoricalDataForEGO()

        #TODO come up with tests for these  functions

        list_IS_to_query = [0]
        num_init_pts_each_IS = 10

        name_testfile = rb.getFuncName() + '_' + 'IS_' + '_'.join(
            str(element) for element in list_IS_to_query) + '_' + str(num_init_pts_each_IS) + "_points_each"

        with open("{0}/{1}.pickle".format(pathToPickles, name_testfile), "wb") as output_file:
            pickle.dump(data, output_file)

        # testHistoricalData = obtainHistoricalDataForEGO(True, rb, pathToPickles, list_IS_to_query, num_init_pts_each_IS)
        # print testHistoricalData
        #
        # testHistoricalDataRandom = obtainHistoricalDataForEGO(False, rb, pathToPickles, list_IS_to_query, num_init_pts_each_IS)
        # print testHistoricalDataRandom



        ### Test createHistoricalDataForKG()
        listPrevData = []

        samples = [[1,1], [1,2]]
        values = [1.0 , 2.0]
        list_noise_variance_at_sample = [0.1, 0.3]
        listPrevData.append( (samples, values, list_noise_variance_at_sample) )

        hist_kg = createHistoricalDataForKG(rb._dim, listPrevData)
        #print hist_kg
        IS_samples = [[0,1,1], [0,1,2]]
        for index in range(len(hist_kg.points_sampled)):
            self.assertTrue( (IS_samples[index] == hist_kg.points_sampled[index]).all )
        for index in range(len(hist_kg.points_sampled_value)):
            self.assertTrue( (values[index] == hist_kg.points_sampled_value[index]).all )

        samples = [ [0,0], [4,3] ]
        for index in range(len(hist_kg.points_sampled)):
            self.assertTrue( (IS_samples[index] == hist_kg.points_sampled[index]).all )

        listPrevData = [ (samples, values, list_noise_variance_at_sample) ]
        bestpt, bestval, best_truth = findBestSampledValue(rb, listPrevData, 0)
        # print findBestSampledValue(rb, listPrevData, 0)
        self.assertAlmostEqual(bestval, 1.0, delta=.0001)
        self.assertAlmostEqual(bestval, 1.0, delta=0.0001)
        # self.assertAlmostEqual(bestval, 1.0, delta=0.0001)
        self.assertAlmostEqual(best_truth, numpy.float64(-9.0), delta=1.0)
        self.assertTrue( (bestpt == [0.0, 0.0]) )

        list_sampled_IS = [0,0]
        gathered_data_from_all_replications = []
        gathered_data_from_all_replications.append( {"points": samples, "vals": values,
                                                     "noise_variance": list_noise_variance_at_sample,
                                                     "sampledIS": list_sampled_IS } )

        for indexList in range(len(gathered_data_from_all_replications)):
            for indexElem in range(len(gathered_data_from_all_replications[indexList]['vals'])):
                self.assertAlmostEqual( values[indexElem], gathered_data_from_all_replications[indexList]['vals'][indexElem], delta=0.0001 )

            for indexElem in range(len(gathered_data_from_all_replications[indexList]['points'])):
                self.assertTrue(samples[indexElem] == gathered_data_from_all_replications[indexList]['points'][indexElem])

            for indexElem in range(len(gathered_data_from_all_replications[indexList]['sampledIS'])):
                self.assertTrue(list_sampled_IS[indexElem] == gathered_data_from_all_replications[indexList]['sampledIS'][indexElem])

        gathered_data_from_all_replications.append( {"points": samples, "vals": values,
                                                     "noise_variance": list_noise_variance_at_sample,
                                                     "sampledIS": list_sampled_IS } )
        for indexList in range(len(gathered_data_from_all_replications)):
            for indexElem in range(len(gathered_data_from_all_replications[indexList]['vals'])):
                self.assertAlmostEqual( values[indexElem], gathered_data_from_all_replications[indexList]['vals'][indexElem], delta=0.0001 )

            for indexElem in range(len(gathered_data_from_all_replications[indexList]['points'])):
                self.assertTrue(samples[indexElem] == gathered_data_from_all_replications[indexList]['points'][indexElem])

            for indexElem in range(len(gathered_data_from_all_replications[indexList]['sampledIS'])):
                self.assertTrue(list_sampled_IS[indexElem] == gathered_data_from_all_replications[indexList]['sampledIS'][indexElem])

        samples = [[-1.,0], [0.1, -2.0]]
        values = [0.2, 1.5]
        list_sampled_IS = [3,3]
        gathered_data_from_all_replications.append( {"points": samples, "vals": values,
                                                     "noise_variance": list_noise_variance_at_sample,
                                                     "sampledIS": list_sampled_IS } )
        for indexElem in range(len(gathered_data_from_all_replications[2]['vals'])):
            self.assertAlmostEqual( values[indexElem], gathered_data_from_all_replications[2]['vals'][indexElem], delta=0.0001 )

        for indexElem in range(len(gathered_data_from_all_replications[2]['points'])):
            self.assertTrue(samples[indexElem] == gathered_data_from_all_replications[2]['points'][indexElem])

        for indexElem in range(len(gathered_data_from_all_replications[2]['sampledIS'])):
            self.assertTrue(list_sampled_IS[indexElem] == gathered_data_from_all_replications[2]['sampledIS'][indexElem])

        listPrevData.append( (gathered_data_from_all_replications[2]['points'],
                              gathered_data_from_all_replications[2]['vals'],
                             gathered_data_from_all_replications[2]['noise_variance'])
                             )

        hist_kg = createHistoricalDataForKG(rb._dim, listPrevData)
        #print hist_kg
        self.assertTrue( (hist_kg.points_sampled[0] == [0,0,0]).all)
        self.assertTrue( (hist_kg.points_sampled[1] == [0,4,3]).all)
        self.assertTrue( (hist_kg.points_sampled[2] == [1,-1.0,0]).all)
        self.assertTrue( (hist_kg.points_sampled[3] == [1,.1,-2]).all)

        self.assertAlmostEqual(values[0],-1.0 * hist_kg.points_sampled_value[2], delta=0.0001)
        self.assertAlmostEqual(values[1],-1.0 * hist_kg.points_sampled_value[3], delta=0.0001)

        self.assertAlmostEqual(list_noise_variance_at_sample[0], hist_kg.points_sampled_noise_variance[2], delta=0.0001)
        self.assertAlmostEqual(list_noise_variance_at_sample[1], hist_kg.points_sampled_noise_variance[3], delta=0.0001)



        ### Test for findBestSampledValueFromHistoricalData()
        atoext = AssembleToOrderExtended( mult= -1.0 )
        hd = HistoricalData(atoext.getDim())
        pts = numpy.array( [ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ], [1.0, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4] ] )
        self.assertTrue(len(pts) == 2)
        self.assertTrue(len(pts[0]) == atoext.getDim())
        self.assertTrue(len(pts[1]) == atoext.getDim())
        vals = [-1.0, 0.2]
        noises = [0.1, 0.2]
        hd.append_historical_data(pts, vals, noises)
        # print hd.to_list_of_sample_points()

        bestpt, best_val, best_truth = findBestSampledValueFromHistoricalData(atoext, hd)
        # print bestpt
        # print best_val
        # print best_truth
        self.assertTrue( (pts[0] == bestpt).all )
        self.assertTrue(best_val == -1.0)
        self.assertAlmostEqual(best_truth, atoext.evaluate(2, bestpt), delta=10.0)

        pts = numpy.array( [ [1.3, 1.4, 10.0, 11.0, 19.0, 1.0, 1.0, 1.0 ], [13.0, 10.2, 10.3, 10.4, 10.5, 0.2, 10.3, 0.4] ] )
        vals = [-11.0, 10.2]
        noises = [10.1, 1000.2]
        hd.append_historical_data(pts, vals, noises)
        bestpt, best_val, best_truth = findBestSampledValueFromHistoricalData(atoext, hd)
        self.assertTrue( (pts[0] == bestpt).all )
        self.assertTrue(best_val == -11.0)

        pts2 = numpy.array( [ [10.3, 10.4, 10.0, 11.0, 19.0, 1.0, 1.0, 1.0 ], [13.0, 10.2, 10.3, 10.4, 10.5, 0.2, 10.3, 0.4] ] )
        vals = [11.0, 10.2]
        hd.append_historical_data(pts, vals, noises)
        bestpt, best_val, best_truth = findBestSampledValueFromHistoricalData(atoext, hd)
        self.assertTrue( (pts[0] == bestpt).all )
        self.assertTrue(best_val == -11.0)





if __name__ == '__main__':
    unittest.main()