import unittest
import numpy as np
import scipy
import scipy.stats
import csv
np.set_printoptions(threshold=np.nan)
import parse_data
import read_numbering_solutions
import singleHalide_BU as s
import correlatedKG as ckg
import stats_model as sm

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
bu = s.singleHalide_BU(n,2)
bu.setCost(10.,0)
bu.setCost(1.,1)

all_points = read_allData('numbering_solutions.csv')
all_points = [[all_points[i][j] for j in range(1,4)] for i in range(n)]
all_points_num = read_numbering_solutions.alphaToNum(all_points)
'''
--- set Prior
'''	
# the samples in initial umbo and be data have to be the same
initial_rawData_umbo = read_iniSample('data_iniSample_umbo.csv')
initial_data_umbo = format_converter(initial_rawData_umbo)
initial_rawData_be = read_iniSample('data_iniSample_be.csv')
initial_data_be = format_converter(initial_rawData_be)
x = [initial_data_umbo[i][:-2] for i in range(len(initial_data_umbo))]
y_be = [initial_data_be[i][-1] for i in range(len(initial_data_umbo))]
y_umbo = [initial_data_umbo[i][-1] for i in range(len(initial_data_umbo))]
#hyper_para = sm.hype_estimator_MLE(y_be,y_umbo,x,bu._model)
#[mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta,l01,l02,sig_m0,\
#	l11,l12,l13,l14,l15,l16,l17,l18,sig_m1] = hyper_para
[mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta,l01,l02,sig_m0,\
l11,l12,l13,l14,l15,l16,l17,l18,sig_m1]=[0.95721727,0.24586293,\
 0.98511899,0.641596,3.1176544,0.57863563,0.89340642,4.48012875,\
 0.65830175,0.39213137,0.30354928,0.05105964,0.65333231,0.2361337,\
 0.44377858,0.1758838,0.44754939]
Z = [all_points_num[i][:-1] for i in range(len(all_points_num))]
bu.setPrior(Z,mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta,[l01,l02],\
	sig_m0,[l11,l12,l13,l14,l15,l16,l17,l18],sig_m1)


'''
--- update using historical data
--- assume the first m samples are observed
'''

for sample in initial_rawData_be:
	sample_num = read_numbering_solutions.get_number(sample[0],\
					sample[1],sample[2])
	bu.updatePosterior(sample[3],sample_num-1,0)	
for sample in initial_rawData_umbo:
	sample_num = read_numbering_solutions.get_number(sample[0],\
					sample[1],sample[2])
	bu.updatePosterior(sample[3],sample_num-1,1)	
'''
--- get where to take the next sample 
'''
nIter = 100
soluList_be = get_full_sample('full_sample_be_neg_norm.csv')
soluList_umbo = get_full_sample('full_sample_norm.csv')
BOList = [0 for i in range(nIter)]
for iter in range(nIter):
	voi_be,arg_be = ckg.correlatedKG(bu._mu[:72],bu._Sig[:72,:72],0.)/bu._cost[0]
	voi_umbo,arg_umbo = ckg(bu_mu[72:],bu._Sig[72:,72:],0.)/bu._cost[1]
	if voi_be > voi_umbo:
		next_sample = [arg_be, 0]
	else:
		next_sample = [arg_umbo, 1]
	next_soln = read_numbering_solutions.get_solution(next_sample+1)
	hal,cat,solv = next_soln[0],next_soln[1],next_soln[2]
	if next_sample[1]==0:
		solu = soluList_be[(hal,cat,solv)]
	else:
		solu = soluList_umbo[(ahl,cat,solv)]
	BOList[iter] = solu
	bu.updatePosterior(solu,next_sample,next_sample[1])
		

# write output
ff = open('perovskite_BO_output.csv','w')
ff_writer = csv.writer(ff)
ff_writer.writerow(BOList)
ff.close()

