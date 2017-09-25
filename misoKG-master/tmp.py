import numpy as np
import boto
from boto.s3.connection import S3Connection
import pickle
from data_io import send_data_to_s3
from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended
from misoRosenbrock.rosenbrock import RosenbrockNew, RosenbrockRemi
import sys
from constants import s3_bucket_name
import pandas as pd
import os

# argv = sys.argv[1:]
# repl = int(argv[0])

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)
# func = AssembleToOrderExtended(mult=-1.0)
# func = RosenbrockNew(mult=1.0)
# func = RosenbrockRemi(mult=1.0)
# for repl in range(100):
#     name = "rbpes_IS_0_1_200_points_each_repl_0.pickle"
#     with open('/fs/europa/g_pf/pickles/miso/rbpes_IS_0_1_5_points_each_repl_{0}.pickle'.format(repl), 'rb') as f:
#         d=pickle.load(f)
#         for IS in range(2):
#             pts = d['points'][IS]
#             vals = np.array([func.evaluate(IS, pt) for pt in pts])
#             noise = func.noise_and_cost_func(IS, None)[0] * np.ones(len(pts))
#             key = 'miso/data/rbRemi_IS_{0}_5_points_repl_{1}'.format(IS, repl)
#             print key
#             data = {'points': pts, 'vals': vals, 'noise': noise}
#             send_data_to_s3(bucket, key, data)

