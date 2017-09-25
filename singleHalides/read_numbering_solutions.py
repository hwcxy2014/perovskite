import csv
import parse_data

def get_number_dict(version):
	# version = 1: key = solution, value = numbering
	# version = 2: reverse of version 1
	f = open('numbering_solutions.csv','r')
	f_reader = csv.reader(f)
	solutions = {}
	if version == 1:
		for row in f_reader:
			[num,hal,cat,sol] = row
			solutions[(hal,cat,sol)] = int(num)
	else:
		for row in f_reader:
			[num,hal,cat,sol] = row
			solutions[int(num)] = [hal,cat,sol]
	f.close()
	return solutions

def alphaToNum(data):
	# input is in (hal, cat, solv) format
	# output is in (0,0,1,0,1,0,density,polarity,solvent number,
	#					solution number) format
	n = len(data)
	# get solvent info
	solvent_info = parse_data.solvent_parser('data_solvent.csv')
	data_num = [[0 for i in range(9)] for j in range(n)]
	for i in range(n):
		# halide
		if data[i][0] == 'Br':
			data_num[i][0] = 1
		elif data[i][0] == 'Cl':
			data_num[i][1] = 1
		elif data[i][0] == 'I':
			data_num[i][2] = 1
		# cation
		if data[i][1] == 'MA':
			data_num[i][3] = 1
		elif data[i][1] == 'FA':
			data_num[i][4] = 1
		elif data[i][1] == 'Cs':
			data_num[i][5] = 1
		# solvent
		name = data[i][2]
		info = solvent_info[name]
		data_num[i][6],data_num[i][7],data_num[i][8] = \
				info[0],info[1],info[2]
	return data_num

def numToAlpha():
	# output is in (hal, cat, solv) format
	# input is in (0,0,1,0,1,0,density,polarity,solvent number) format
	return 'function not available for now'

def get_solution(number):
	# input: numbering of the solution
	# output: components of the solution
	solutions = get_number_dict(2)
	return solutions[number]

def get_number(hal,cat,sol):
	# input: components of the solution
	# output: numbering of the solution
	solutions = get_number_dict(1)
	return solutions[(hal,cat,sol)]



