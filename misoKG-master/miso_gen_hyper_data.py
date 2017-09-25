import numpy as np
import boto
from boto.s3.connection import S3Connection
from joblib import Parallel, delayed

from misoRosenbrock.rosenbrock import RosenbrockNew, RosenbrockRemi
from data_io import send_data_to_s3
from constants import s3_bucket_name

__author__ = 'jialeiwang'

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)

num_pts = 1000

##### Rosenbrock
# func = RosenbrockRemi(mult=1.0)
func = RosenbrockNew(mult=1.0)
list_IS = [0, 1]
search_domain = func.get_moe_domain()
points = search_domain.generate_uniform_random_points_in_domain(num_pts)
for IS in list_IS:
    key = "miso/data/hyper_{0}_IS_{1}_1000_points".format(func.getFuncName(), IS)
    print key
    vals = np.array([func.evaluate(IS, pt) for pt in points])
    noise = np.ones(len(points)) * func.noise_and_cost_func(IS, None)[0]
    data = {'points': points, 'vals': vals, 'noise': noise}
    send_data_to_s3(bucket, key, data)
