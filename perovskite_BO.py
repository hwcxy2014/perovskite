import unittest
import numpy as np
import scipy
import scipy.stats
import csv
np.set_printoptions(threshold=np.nan)
import gaussian_process
import hyper_parameters
import parse_data
import read_numbering_solutions

def read_iniSample(filename):
	fin = open(filename,'rU')
	f_reader = csv.reader(fin, dialect=csv.excel_tab) 
	data = []
	for item in f_reader:
		row = item[0].split(',')
		row[-1] = float(row[-1])
		data.append(row)
	fin.close()
	return data
def read_allData(filename):
	fin = open(filename,'r')
	f_reader = csv.reader(fin)
	data = []
	for item in f_reader:
		item[0] = int(item[0])
		data.append(item)
	fin.close()
	return data
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

def get_full_sample(filename):
	# read in the the solubilities of all samples
	# output: {('hal','cat','solv'):solubility}
	f = open(filename, 'rU')
	f_reader = csv.reader(f,dialect=csv.excel_tab)
	solus = {}
	for item in f_reader:
		item = item[0].split(',')
		hal,cat,solv = item[0],item[1],item[2]
		solus[(hal,cat,solv)] = float(item[3])
	f.close()
	return solus
'''
--- initialize gaussian process
'''
n = 72
gp = gaussian_process.solubilities(n)
all_points = read_allData('numbering_solutions.csv')
all_points = [[all_points[i][j] for j in range(1,4)] for i in range(n)]
all_points_num = read_numbering_solutions.alphaToNum(all_points)
print all_points_num
'''
--- set Prior
'''	
initial_rawData = read_iniSample('data_iniSample.csv')
initial_data = format_converter(initial_rawData)
print initial_data
#hyper_para = hyper_parameters.MLE(initial_data)
hyper_para = [0.39522832,0.34814395,0.0581192,0.30348731,0.03480635,\
				0.02713183,0.33849561,0.72345582]
#hyper_para = [ 0.35009371, 0.40118288, 0.81866189, 0.9547815, \
#			  0.88751619, 0.17602205, 0.07244465, 0.64099571]
[mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta,sig_m,l1,l2] = hyper_para
gp.setPrior(all_points_num,mu_alpha, sig_alpha, \
				mu_zeta, sig_zeta,sig_beta,sig_m,l1,l2)
'''
--- update using historical data
--- assume the first m samples are observed
'''

for sample in initial_rawData:
	sample_num = read_numbering_solutions.get_number(sample[0],\
					sample[1],sample[2])
	gp.updatePosterior(sample[3],sample_num-1)	
'''
--- get where to take the next sample 
'''
nIter = 50
soluList = get_full_sample('full_sample.csv')
BOList = [0 for i in range(nIter)]
for iter in range(nIter):
	next_sample = gp.getNextSample()
	next_soln = read_numbering_solutions.get_solution(next_sample+1)
	hal,cat,solv = next_soln[0],next_soln[1],next_soln[2]
	solu = soluList[(hal,cat,solv)]
	BOList[iter] = solu
	gp.updatePosterior(solu,next_sample)

# write output
ff = open('perovskite_BO_output.csv','w')
ff_writer = csv.writer(ff)
ff_writer.writerow(BOList)
ff.close()

