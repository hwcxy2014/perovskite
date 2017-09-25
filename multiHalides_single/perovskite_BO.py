import unittest
import fpl_auto
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
    # input: data format = [halide1, halide2, halide3, \
    #                   cation, solvent, solubility]
    # output: data format = [0,0,1,0,0,1,0,0,1,0,0,1,\
    #               mvee,polarity,solvent#,solubility]
    n = len(data)
    # get solvent info
    solvent_info = parse_data.solvent_parser('data_solvent.csv')
    data_num = [[0 for i in range(16)] for j in range(n)]
    for i in range(n):
        # halide1
        if data[i][0] == 'Br':
            data_num[i][0] = 1
        elif data[i][0] == 'Cl':
            data_num[i][1] = 1
        elif data[i][0] == 'I':
            data_num[i][2] = 1
        # halide2
        if data[i][1] == 'Br':
            data_num[i][3] = 1
        elif data[i][1] == 'Cl':
            data_num[i][4] = 1
        elif data[i][1] == 'I':
            data_num[i][5] = 1
        # halide3
        if data[i][2] == 'Br':
            data_num[i][6] = 1
        elif data[i][2] == 'Cl':
            data_num[i][7] = 1
        elif data[i][2] == 'I':
            data_num[i][8] = 1
        # cation
        if data[i][3] == 'MA':
            data_num[i][9] = 1
        elif data[i][3] == 'FA':
            data_num[i][10] = 1
        elif data[i][3] == 'Cs':
            data_num[i][11] = 1
        # solvent
        name = data[i][4]
        info = solvent_info[name]
        data_num[i][12],data_num[i][13],data_num[i][14] = \
                info[0],info[1],info[2]
        data_num[i][15] = data[i][5]
    return data_num

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
n = 648
gp = gaussian_process.solubilities(n)
all_points = read_allData('numbering_solutions.csv')
all_points = [[all_points[i][j] for j in range(1,6)] for i in range(n)]
all_points_num = read_numbering_solutions.alphaToNum(all_points)
'''
--- set Prior
''' 
initial_rawData = read_iniSample('data_iniSample.csv')
initial_data = format_converter(initial_rawData)
print initial_data
#hyper_para = hyper_parameters.MLE(initial_data)
hyper_para = [1.08153853e-01,3.93812462e-01,8.91991501e-01,\
              3.30380611e-03,3.90495942e-01,1.04133289e-04,\
              4.91940389e-01,2.57158861e-01]
[mu_alpha,sig_alpha,mu_zeta,sig_zeta,sig_beta,sig_m,l1,l2] = hyper_para
gp.setPrior(all_points_num,mu_alpha, sig_alpha, \
                mu_zeta, sig_zeta,sig_beta,sig_m,l1,l2)
'''
--- update using historical data
--- assume the first m samples are observed
'''

for sample in initial_rawData: 
    sample_num = read_numbering_solutions.get_number(sample[0],\
                    sample[1],sample[2],sample[3],sample[4])
    gp.updatePosterior(sample[5],sample_num-1)  
'''
--- get where to take the next sample 
'''
nIter = 100
BOList = [[] for i in range(nIter)]
for iter in range(nIter):
    next_sample = gp.getNextSample()
    next_soln = read_numbering_solutions.get_solution(next_sample+1)
    hal1,hal2,hal3,cat,solv = next_soln[0],next_soln[1],\
            next_soln[2],next_soln[3],next_soln[4]
    solu = fpl_auto.get_UMBO([hal1,hal2,hal3],cat,solv)
    print solu,next_soln
    with open(r'BO_result.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(next_soln+[solu]) 
    BOList[iter] = next_soln+[solu]
    gp.updatePosterior(solu,next_sample)

# write output
ff = open('perovskite_BO_output.csv','w')
ff_writer = csv.writer(ff)
ff_writer.writerows(BOList)
ff.close()

