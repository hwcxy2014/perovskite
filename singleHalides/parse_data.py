import csv
import re

def solvent_parser(filename):
	# output:
	#	a dictionary with solvent name as key
	f = open(filename, 'rU')
	f_reader = csv.reader(f, dialect=csv.excel_tab)
	solvents = {}
	count = 0
	for row in f_reader:
		solv, v1, v2 = row[0].split(',')
		solvents[solv] = [float(v1),float(v2),count]
		count += 1
	return solvents

def data_parser(filename_solvents, filename_ma, filename_cs, \
				filename_output):
	# parse solvents
	solvent_list = solvent_parser(filename_solvents)
	# halide
	halides = {'I':[1,0,0],'B':[0,1,0],'C':[0,0,1]}
	# parse solutions with MA cation
	# MA is represented by [1,0,0]
	f = open(filename_ma,'rU')
	f_reader =csv.reader(f, dialect=csv.excel_tab)
	data = []
	for row in f_reader:
		ions, solv, eng = row[0].split(',')
		# halide is represented by the second upper case letter
		up_letters = re.findall('([A-Z])',ions)
		halide = up_letters[1]
		halide_rep = halides[halide]
		solv_rep = solvent_list[solv]
		eng = float(eng)
		solution = halide_rep + [1,0,0] + solv_rep + [eng]
		data.append(solution)
	f.close()
	# parse solution with Cs cation
	# Cs is represented by [0,1,0]
	f = open(filename_cs,'rU')
	f_reader = csv.reader(f,dialect=csv.excel_tab)
	for row in f_reader:
		ions, solv, eng = row[0].split(',')
		# halide is represented by the second upper case letter
		up_letters = re.findall('([A-Z])',ions)
		halide = up_letters[1]
		halide_rep = halides[halide]
		solv_rep = solvent_list[solv]
		eng = float(eng)
		solution = halide_rep + [0,1,0] + solv_rep + [eng]
		data.append(solution)
	f.close()
	# write data into csv
	fout = open(filename_output, 'w')
	f_writer = csv.writer(fout)
	f_writer.writerows(data)
	fout.close()
	return 0

#--------- main -----------

#data_parser('data_solvent_mbo.csv','data_MA_raw.csv',\
#					'data_Cs_raw.csv','data_solution_parsed_1.csv')
