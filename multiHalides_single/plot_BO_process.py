import csv
import read_numbering_solutions
import matplotlib.pyplot as plt

# import results for single sample
fr = open('BO_result.csv','rU')
fr_reader = csv.reader(fr)
bo_sin = []
for item in fr_reader:
	bo_sin.append(float(item[5]))
fr.close()
# add initial data
# bo_sin = [0.5604] + bo_sin
'''
# import results for batch sample
fb = open('../multihalides_batch/BO_result.csv','rU')
fb_reader = csv.reader(fb)
bo_bat = []
bo_bat = [0.5604] + bo_bat
for item in fb_reader:
	item = map(float,item)
	# take only the largest value in a batch
	bo_bat.append(max(item[:5]))
fb.close()
'''
bo_sin_max = [max(bo_sin[:i]) for i in range(1,len(bo_sin)+1)]
#bo_bat_max = [max(bo_bat[:i]) for i in range(1,len(bo_bat)+1)]
plt.plot([i for i in range(len(bo_sin))], bo_sin_max, 'b')
#plt.plot([i for i in range(len(bo_bat))], bo_bat_max, 'r')
plt.show()

'''
# plot
data_num = [data[i][4] for i in range(len(data))]
data_sol = [data[i][3] for i in range(len(data))]

max_sol = max([data[i][3] for i in range(len(data))])
f = open('perovskite_BO_output.csv','rU')
f_reader = csv.reader(f)
bo_sol = []
for item in f_reader:
	bo_sol = map(float,item)
f.close()

n_obs = len(bo_sol)
print n_obs
bo_sol_max = [max(bo_sol[0:i]) for i in range(1,len(bo_sol)+1)]
plt.plot([i for i in range(1,n_obs+1)],bo_sol_max,'b')
plt.plot([i for i in range(1,n_obs+1)],[max_sol for i in range(n_obs)],'r')
plt.ylabel('UMBO')
plt.xlabel('Iterations')
plt.savefig('plot_BO_process.png')
plt.show()

'''
