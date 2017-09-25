import numpy as np
import scipy
import scipy.stats
np.set_printoptions(threshold=np.nan)
import gaussian_process
import hyper_parameters
import parse_data
import csv
import matplotlib.pyplot as plt

def format_converter(data):
	# input: data = [halide, cation, solvent, solubility]
	n = len(data)
	# get solvent info
	solvent_info = parse_data.solvent_parser('data_solvent.csv')
	data_cl = [[0 for i in range(10)] for j in range(n)]
	for i in range(n):
		# halide
		if data[i][0] == 'Br':
			data_cl[i][0] = 1
		elif data[i][0] == 'Cl':
			data_cl[i][1] = 1
		elif data[i][0] == 'I':
			data_cl[i][2] = 1
		# cation
		if data[i][1] == 'MA':
			data_cl[i][3] = 1
		elif data[i][1] == 'FA':
			data_cl[i][4] = 1
		elif data[i][1] == 'Cs':
			data_cl[i][5] = 1
		# solvent
		name = data[i][2]
		info = solvent_info[name]
		data_cl[i][6],data_cl[i][7],data_cl[i][8] = \
				info[0],info[1],info[2]
		data_cl[i][9] = data[i][3]
	return data_cl

def readData(filename):
	'''
	--- output: python list, not numpy array
	'''
	f = open(filename,'Ur')
	f_reader = csv.reader(f,dialect=csv.excel_tab)
	data = []
	for row in f_reader:
		row = row[0].split(',')
		row[-1] = float(row[-1])
		data.append(row)
	return data

# read data from file
data = readData('full_sample.csv')
data = format_converter(data)
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
plt.errorbar([j for j in range(n_obs)], ci_list[:,0],yerr = ci_list[:,1])
plt.plot([j for j in range(n_obs)],ci_list[:,2],'sb')
plt.xlabel('Iterations')
plt.ylabel('UMBO')
plt.savefig('plot_LOOCV_72.png')
plt.show()
