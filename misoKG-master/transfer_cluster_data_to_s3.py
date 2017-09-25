import boto
from boto.s3.connection import S3Connection
import pickle
from data_io import send_data_to_s3, get_data_from_s3
import os

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket('poloczeks3', validate=True)
# bucket = conn.get_bucket('frazier-research', validate=True)
# cluster_dir = '/fs/europa/g_pf/pickles/miso/data/'
# s3_dir = 'miso/data/'
# for filename in os.listdir(cluster_dir):
#     if '.pickle' in filename:
#         print filename.split('.')[0]
#         with open(cluster_dir+filename, 'rb') as f:
#             d=pickle.load(f)
#             send_data_to_s3(bucket, s3_dir+filename.split('.')[0], d)

cluster_dir = '/fs/europa/g_pf/pickles/miso/data/'
s3_dir = 'miso/data/'
for i in range(100):
    for IS in range(3):
        name = "atoext_IS_{0}_20_points_repl_{1}".format(IS, i)
        print name
        with open(cluster_dir+name, 'rb') as f:
            data = pickle.load(f)
        send_data_to_s3(bucket, s3_dir+name, data)


# cluster_dir = '/fs/europa/g_pf/pickles/miso/hyper/'
# s3_dir = 'miso/hyper/'
# # for i in range(100):
# #     for IS in range(3):
# # name = "atoext_IS_{0}_20_points_repl_{1}".format(IS, i)
# name = "mkg_atoext"
# print name
# data = get_data_from_s3(bucket, s3_dir+name)
# with open(cluster_dir+name, 'wb') as f:
#     pickle.dump(data, f)
