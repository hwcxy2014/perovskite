import boto
from boto.s3.connection import S3Connection
import sys
from constants import s3_bucket_name
from data_io import get_data_from_s3
from problems.identifier import identify_problem

__author__ = 'matthiaspoloczek'

'''
Script to inspect hypers stored at S3

invoke as : python inspect_hypers.py miso_lrMU_hyper_ego

Optional parameters for lrMU:
    "miso_lrMU_hyper_ego":
    "miso_lrMU_hyper_mkg":
    "miso_lrMU_hyper_pes":
    "miso_lrMU_hyper_mei":
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

data = get_data_from_s3(bucket, problem.hyper_path)
print "prior_mean = " + str(data["prior_mean"])
print "prior_sig = " + str(data["prior_sig"])
if argv[0].find("pes") >=0:
    print "hyperparam = " + str(data["hyperparam"])
    print "hyperparam_mat = " + str(data["hyperparam_mat"])
else:
    print "hyper_bounds = " + str(data["hyper_bounds"])
    print "hyperparam = " + str(data["hyperparam"])
    print "loglikelihood = " + str(data["loglikelihood"])