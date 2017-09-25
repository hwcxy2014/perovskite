import os
import pickle

data_dir = "/fs/europa/g_pf/pickles/coldstart/data"
result_dir = "/fs/europa/g_pf/pickles/coldstart/result"
all_files = os.listdir(result_dir)
for filename in all_files:
    if filename.find("seqpes") >= 0:
        with open("{0}/{1}".format(result_dir, filename), 'rb') as f:
            d = pickle.load(f)
        data_to_pickle = {"points": d['sampled_points'],
                          "vals": d['sampled_vals'],
                          "noise": d['sampled_noise_var'],
                          }
        with open("{0}/{1}".format(data_dir, filename), 'wb') as ff:
            pickle.dump(data_to_pickle, ff)