import csv
import numpy as np
def normalize_value(filename):
	f = open(filename)
	f_r = csv.reader(f)
	rows = []
	l = []
	for item in f_r:
		row = item[:-1]+[float(item[-1])]
		l.append(row[-1])
		rows.append(row)
	m = np.mean(l)
	st = np.std(l)
	ff = open(filename[:-4]+'_norm'+'.csv','w')
	f_w = csv.writer(ff)
	for row in rows:
		f_w.writerow(row[:-1]+[(row[-1]-m)/st])
	f.close()
	ff.close()

normalize_value('full_sample_be_neg.csv')
