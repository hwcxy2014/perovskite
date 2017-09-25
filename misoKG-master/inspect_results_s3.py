import boto
from boto.s3.connection import S3Connection
import sys
from constants import s3_bucket_name
from data_io import get_data_from_s3
from problems.identifier import identify_problem

__author__ = 'matthiaspoloczek'

'''
Script to inspect data stored at S3

invoke as : python inspect_results_s3.py miso_lrMU_benchmark_mkgcandpts 0

where 0 is a natural integer determining a replication

Optional parameters for lrMU:
    "miso_lrMU_benchmark_ego":
    "miso_lrMU_benchmark_mkg":
    "miso_lrMU_benchmark_mkgcandpts":
    "miso_lrMU_benchmark_pes":
    "miso_lrMU_benchmark_mei":
'''

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)
# construct problem instance given CMD args
argv = sys.argv[1:]
if argv[0].find("ego") < 0 and argv[0].find("kg") < 0 and argv[0].find("mei") < 0 and argv[0].find("mkg") < 0\
        and argv[0].find("pes") < 0:
    raise ValueError("No correct algo selected!")
problem = identify_problem(argv, bucket)

data = get_data_from_s3(bucket, problem.result_path)

# print data['sampled_is']
print data['raw_voi']
