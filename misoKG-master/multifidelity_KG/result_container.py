import numpy
import pandas

import sql_util

__author__ = 'jialeiwang'

class BenchmarkResult(object):

    def __init__(self, num_iterations, func_dim, table_name):
        self.idx = 0
        self.table = pandas.DataFrame()
        self.table['idx'] = numpy.zeros(1)
        self.table['IS'] = numpy.zeros(1),
        self.table['total_cost'] = numpy.zeros(1),
        self.table['voi'] = numpy.zeros(1),
        self.table['x_mean'] = numpy.zeros(1)
        self.table['x_var'] = numpy.zeros(1),
        self.table['x_val'] = numpy.zeros(1),
        self.table['best_sampled_val'] = numpy.zeros(1),
        self.table['best_sampled_truth'] = numpy.zeros(1),
        self.table['mu_star_truth'] = numpy.zeros(1),
        self.table['mu_star'] = numpy.zeros(1),
        self.table['mu_star_var'] = numpy.zeros(1),
        for k in range(func_dim):
            self.table['mu_star_pt{0}'.format(k)] = numpy.zeros(1)
        for k in range(func_dim):
            self.table['sample_pt{0}'.format(k)] = numpy.zeros(1)
        self._total_cost = 0.0
        self._func_dim = func_dim
        self._table_name = table_name

    def add_entry(self, sample_point, IS, sample_val, best_sampled_val, best_sampled_truth, predict_mean, predict_var, cost, voi, mu_star=0.0, mu_star_var=0.0, mu_star_truth=0.0, mu_star_point=None):
        self._total_cost += cost
        self.table.ix[0, 'idx'] = self.idx
        self.table.ix[0, 'IS'] = IS
        self.table.ix[0, 'total_cost'] = self._total_cost
        self.table.ix[0, 'voi'] = voi
        self.table.ix[0, 'x_mean'] = predict_mean
        self.table.ix[0, 'x_var'] = predict_var
        self.table.ix[0, 'x_val'] = sample_val
        self.table.ix[0, 'best_sampled_val'] = best_sampled_val
        self.table.ix[0, 'best_sampled_truth'] = best_sampled_truth
        self.table.ix[0, 'mu_star_truth'] = mu_star_truth
        self.table.ix[0, 'mu_star'] = mu_star
        self.table.ix[0, 'mu_star_var'] = mu_star_var
        for k in range(self._func_dim):
            self.table.ix[0, 'sample_pt{0}'.format(k)] = sample_point[k]
            if mu_star_point is not None:
                self.table.ix[0, 'mu_star_pt{0}'.format(k)] = mu_star_point[k]
        self.table.to_sql('benchmark_{0}'.format(self._table_name), sql_util.sql_engine, if_exists='append')
        self.idx += 1
