import sys
import numpy as np
from joblib import Parallel, delayed

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

import boto
from boto.s3.connection import S3Connection
from data_io import send_data_to_s3

from logreg_mnistusps import LogRegMNISTUSPS

__author__ = 'matthiaspoloczek'

'''
Evaluate points add a Latin Hypercube design and store them in Matthias' S3

Main difference to logreg_gen_benchmark_data: all IS are evaluated at the same pts.
Moreover, there is only 1 replication.
'''


conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket('poloczeks3', validate=True)

num_pts = 1000
estimateMean = False # change to False once the mean is adapted in class
allows_parallelization = True  # set to True if each simulator/IS can be queried multiple times simultaneously
# is True for lrMU, rosenbrock and ATO
# is False for dragAndLift
### end

directory = '/miso/data'

func_dict = {
    "lrMU": LogRegMNISTUSPS(mult=1.0),
}

argv = sys.argv[1:]
func_name = argv[0]
func = func_dict[func_name]

search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in func._search_domain])

num_parallel_jobs = num_pts
if(('ato' in func_name) and (num_parallel_jobs > 16)): # do not start too many MATLAB instances
    num_parallel_jobs = 16
if(('lrMU' in func_name) and (num_parallel_jobs > 8)): # do not start too many theano instances
    num_parallel_jobs = 8
if(not allows_parallelization):
    num_parallel_jobs = 1

def parallel_func(IS, pt):
    return func.evaluate(IS, pt)

### Main difference to logreg_gen_benchmark_data: all IS are evaluated at the same pts
points = search_domain.generate_uniform_random_points_in_domain(num_pts)
meanIS = [0.0, 0.0]

with Parallel(n_jobs=num_parallel_jobs) as parallel:
    for index_IS in func.getList_IS_to_query():
        vals = parallel(delayed(parallel_func)(index_IS, pt) for pt in points)

        if estimateMean:
            # print np.array(vals)
            mean = np.mean(np.array(vals))
            vals = [(obsval - mean) for obsval in vals]
            print 'Computed mean '+str(mean)+' for IS '+str(index_IS)
            meanIS[index_IS] = mean

        noise = [func.noise_and_cost_func(index_IS, pt)[0] for pt in points] #func.noise_and_cost_func(index_IS, None)[0] * np.ones(num_pts)
        data = {"points": np.array(points), "vals": np.array(vals), "noise": np.array(noise)}

        # write to S3
        key = directory+'/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, func_name, num_pts)
        send_data_to_s3(bucket, key, data)

    if estimateMean:
        print meanIS


