import csv
import read_numbering_solutions
import matplotlib.pyplot as plt

# read real solubilities
sol_number = read_numbering_solutions.get_number_dict(1)
number_sol = read_numbering_solutions.get_number_dict(2)
f = open('full_sample_be.csv','r')
f_reader = csv.reader(f)
data = []
data_dict = {}
for row in f_reader:
	sol = [row[0],row[1],row[2]]
	sol_tuple = (row[0],row[1],row[2])
	val = float(row[3])
	num = sol_number[sol_tuple]
	data.append(sol + [val] + [num])
	data_dict[sol_tuple] = val
f.close()

# read initial sampels


# plot
data_num = [data[i][4] for i in range(len(data))]
data_sol = [data[i][3] for i in range(len(data))]
'''
plt.plot(data_num, data_sol,'bo')
bo_num = [32,24,48,27,35]
bo_solution = [number_sol[i+1] for i in bo_num]
print bo_solution
bo_sol = [data_dict[(r[0],r[1],r[2])] for r in bo_solution]
plt.plot(bo_num,bo_sol,'ro',ms = 10)
for i in range(len(bo_num)):
	txt = i+1
	plt.annotate(str(txt),(bo_num[i],bo_sol[i]))
plt.show()
'''
max_sol = min([data[i][3] for i in range(len(data))])
f = open('perovskite_BO_output.csv','r')
f_reader = csv.reader(f)
bo_sol = []
for item in f_reader:
	print item
	bo_sol = map(float,item)
f.close()

n_obs = len(bo_sol)
print n_obs
bo_sol_max = [min(bo_sol[0:i]) for i in range(1,len(bo_sol)+1)]
plt.plot([i for i in range(1,n_obs+1)],bo_sol_max,'b')
plt.plot([i for i in range(1,n_obs+1)],[max_sol for i in range(n_obs)],'r')
plt.ylabel('binding energy')
plt.xlabel('Iterations')
plt.savefig('plot_BO_process.png')
plt.show()


