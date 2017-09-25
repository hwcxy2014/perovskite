import lhsmdu
import parse_data
import numpy as np

def get_nearest(pt, solvents):
	'''
	input:  pt = [x1,x2]
			sovlents is a dictionary	
	'''
	x1, x2 = pt[0], pt[1]
	dis_dict = {}
	dis_list = np.zeros(len(solvents.keys()))
	count = 0
	for key in solvents:
		s1,s2 = solvents[key][0],solvents[key][1]
		distance = (x1-s1)**2 + (x2-s2)**2
		dis_dict[distance] = key
		dis_list[count] = distance
		count += 1
	min_dist = min(dis_list)
	return dis_dict[min_dist]

# read in solvents and their polarity and mbo
sol = parse_data.solvent_parser('data_solvent.csv')
# number of initial samples
n_sample = 10
# sample uniformly from [0,1]^4
samples = lhsmdu.sample(4,n_sample)
# first two dimensions are {1,2,3}
samples[0] = np.floor(samples[0] * 3)
samples[1] = np.floor(samples[1] * 3)
# third dim is (0.6,1.4)
samples[2] = [0.6 for i in range(n_sample)] + \
				samples[2] * (1.4-0.6)
# fourth dim is (5, 60)
samples[3] = [5. for i in range(n_sample)] + \
				samples[3] * (60 - 5.)
# map samples to solution combo
solutions = [[0 for j in range(3)] for i in range(n_sample)]
for i in range(n_sample):
	# halides
	if samples[0,i] == 0:
		solutions[i][0] = 'I'
	elif samples[0,i] == 1:
		solutions[i][0] = 'Br'
	else:
		solutions[i][0] = 'Cl'
	# cations
	if samples[1,i] == 0:
		solutions[i][1] = 'MA'
	elif samples[1,i] == 1:
		solutions[i][1] = 'Cs'
	else:
		solutions[i][1] = 'FA'
	# solvents
	# choose the one with the shortest distance
	solutions[i][2] = get_nearest([samples[2,i],samples[3,i]],sol)
print solutions
