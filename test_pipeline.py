import numpy as np
import scipy
import scipy.stats
np.set_printoptions(threshold=np.nan)
import gaussian_process
import hyper_parameters
import csv
import matplotlib.pyplot as plt

'''
This script shall
- Instantiate the hyper-parameters randomly to fix a model.
- Then draw random observations from this model for say 20 points.
- estimate the hyper-parameters of the model via MLE. Let this estimate by theta.
- Run the BO algorithm with these 20 points as initial data and the above estimated hypers theta for a couple of steps, say 50, and plot what obj values EI achieves.
'''

# define hyperparameters
l1 = 0.1
l2 = 0.8
sig_m = 2
mu_alpha = 4
sig_alpha = 2
mu_zeta = 0.5
sig_zeta = 0.2
sig_beta = 1.
n = 135 
m = 20

# generate solutions
cations = np.identity(3)
halides = np.identity(3)
solvents = np.array([np.random.random(2)*1000 for i in range(15)])
data = np.zeros([n,10])
rowCount = 0
for i in range(3):
	for j in range(3):
		for k in range(15):
			# set cation
			data[rowCount,:3] = cations[i,:]
			# set halide
			data[rowCount,3:6] = halides[j,:]
			# set solvent
			data[rowCount,6:8] = solvents[k,:]
			data[rowCount,8] = k
			rowCount += 1

# calculate mean and variance based on hyperparameters
mu = [2*mu_alpha + mu_zeta for x in range(n)]
matern = hyper_parameters.maternKernel(data[:,6:8],l1,l2,sig_m)
Sig = np.identity(n)
for x in range(n):
	for xx in range(n):
		if x==xx:
			Sig[x,xx] = 2 * sig_alpha + sig_beta + sig_zeta\
						+ matern[x,xx]
		elif x > xx:
			Sig[x,xx] = Sig[xx,x]
		else:
			count_equal = 0
			for i in range(6):
				if data[x,i] == 1 and data[xx,i] == 1:
					count_equal += 1
				Sig[x,xx] = count_equal*sig_alpha + sig_zeta \
							+ matern[x,xx]
# generate random sample for solubilities
solubilities = scipy.stats.multivariate_normal.rvs(mu, Sig)
data[:,9] = solubilities

'''
--- initialize BO process
'''
bo = gaussian_process.solubilities(n)

'''
--- estimate hyperparameters
--- Assume 20 out of 135 points are observed
'''
observed_loca = np.random.permutation([i for i in range(n)])
observed_loca = observed_loca[:m]
observed_data = data[observed_loca,:]
prior_prm = hyper_parameters.MLE(observed_data)

'''
--- set prior
'''
[x1,x2,x3,x4,x5,x6,x7,x8] = prior_prm
bo.setPrior(data[:,0:9],x1,x2,x3,x4,x5,x6,x7,x8)

'''
--- update posterior
'''
for i in range(m):
	loca = observed_loca[i]
	bo.updatePosterior(data[loca,9],loca)

'''
--- get next 50 samples
'''
ei_list = np.zeros(50)
max_list = np.zeros(50)
for i in range(50):
	next_sample, ei = bo.getNextSample()
	ei_list[i] = ei
	max_list[i] = bo._max
	observed = data[next_sample, 9]
	bo.updatePosterior(data[next_sample, 9], next_sample)

'''
--- write output to a file
'''
'''
f = open('test_pipeline_output.csv','wb')
f_writer = csv.writer(f)
f_writer.writerow(ei_list)
f.close()
'''

'''
--- plot result
'''
plt.plot([i for i in range(50)],np.array(ei_list)*40,'r',[i for i in range(50)], max_list,'b')
plt.show()  
