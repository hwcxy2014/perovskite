import numpy as np
import scipy
import scipy.stats
np.set_printoptions(threshold=np.nan)
import gaussian_process
import hyper_parameters
import csv
import matplotlib.pyplot as plt
def readData(filename):
	'''
	--- output: python list, not numpy array
	'''
	f = open(filename,'r')
	f_reader = csv.reader(f)
	data = []
	for row in f_reader:
		values = [float(i) for i in row]
		values[:6] = [int(values[i]) for i in range(6)]
		values[8] = int(values[8])
		values[9] = -values[9]/1000.
		data.append(values)
	return data

# read data from file
data = readData('data_solution_parsed.csv')
n_data = len(data)
# number of observed sample
n_obs = 20
# randomly pick n_obs points from data
rnd_ay = np.random.permutation([i for i in range(n_data)])
observed_posi = rnd_ay[:n_obs]
# confidence interval list with true obs
ci_list = np.zeros([n_obs,3])
count = 0
for i in observed_posi:
	train_posi = [j for j in observed_posi if j != i] 
	train_data = [data[j] for j in train_posi]
	print train_data
	loo = data[i]
	bo = gaussian_process.solubilities(n_data)
	theta = hyper_parameters.MLE(train_data)
	[x1,x2,x3,x4,x5,x6,x7,x8] = theta
	bo.setPrior([data[j][:9] for j in range(n_data)], \
				x1,x2,x3,x4,x5,x6,x7,x8)
	for j in train_posi:
		bo.updatePosterior(data[j][9],j)
	mu_pred = bo._mu[i]
	va_pred = bo._Sig[i,i]
	CI_error = 1.96*np.sqrt(va_pred)
	ci_list[count,:] = [mu_pred, CI_error, loo[9]]
	count += 1
plt.errorbar([j for j in range(n_obs)], ci_list[:,0],yerr = ci_list[:,2])
plt.plot([j for j in range(n_obs)],ci_list[:,1],'sb')
plt.show()
