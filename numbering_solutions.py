import csv

f = open('numbering_solutions.csv','w')
f_writer = csv.writer(f)

halides = ['Br','Cl','I']
cations = ['MA','FA','Cs']
solvents = ['dmso','DMF','nmp','gbl','acetone','methacrolein',\
			'THTO','nitromethane']
count = 1
for i in halides:
	for j in cations:
		for k in solvents:
			f_writer.writerow([count,i,j,k])
			count += 1
f.close()			
