import sys
import numpy as np

import boto
from boto.s3.connection import S3Connection
from data_io import get_data_from_s3

__author__ = 'matthiaspoloczek'

'''
Inspect the data used to generate hypers. The command line arg could be lrMU.
'''


conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket('poloczeks3', validate=True)

argv = sys.argv[1:]
func_name = argv[0]

directory = '/miso/data'
num_pts = 1000
key = directory+'/hyper_{1}_IS_{0}_{2}_points'.format(0, func_name, num_pts)
data0 = get_data_from_s3(bucket, key)
key = directory+'/hyper_{1}_IS_{0}_{2}_points'.format(1, func_name, num_pts)
data1 = get_data_from_s3(bucket, key)

### To inspect initial data
# num_pts = 10
# key = directory+'/{1}_IS_{0}_{2}_points_repl_0'.format(0, func_name, num_pts)
# data0 = get_data_from_s3(bucket, key)
# key = directory+'/{1}_IS_{0}_{2}_points_repl_0'.format(1, func_name, num_pts)
# data1 = get_data_from_s3(bucket, key)

print np.mean(data0["vals"])
print np.mean(data1["vals"])
#
# print np.mean(data0["noise"])
# print np.mean(data1["noise"])

print np.var(data0["vals"])
print np.var(data1["vals"])
print np.var(data1["vals"] - data0["vals"])

