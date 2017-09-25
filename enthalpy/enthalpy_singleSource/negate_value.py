import csv

def negate_value(filename):
	f = open(filename)
	f_r = csv.reader(f)
	ff = open(filename[:-4]+'_neg'+'.csv','w')
	f_w = csv.writer(ff)
	for item in f_r:
		row = item[:-1]+[-float(item[-1])]
		f_w.writerow(row)
	f.close()
	ff.close()

negate_value('data_iniSample.csv')
