import numpy as np
import scipy as sp
from scipy import linalg as lin
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib
from matplotlib import cm
matplotlib.use('PDF')
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import call_cfd as cfd
import cPickle as pickle
import os
from multiprocessing import Pool
import time

# rc('text', usetex=True)
# rc('font', family='serif')
# rc('font', serif='Computer Modern Roman')
# rc('font', size='12')

def compute_covariance_matrix(X,hyp):


	N_dim  = X[0,:].size
	N_set  = X[:,0].size
	lambd  = np.abs(hyp[0])
	s2_max = np.abs(hyp[1])
	L      = np.abs(hyp[2:])

	d2 = np.zeros((N_set, N_set))
	
	for i in range(0,N_dim):
		x  = np.tile(X[:,i],(N_set,1))
		d2 += (x - x.transpose())**2 / (L[i])**2
	K_exp = np.exp(-d2)

	K = s2_max*K_exp + np.eye(N_set)*lambd

	return K

def compute_kstar(X,hyp,xtest):
	N_dim  = X[0,:].size
	N_set  = X[:,0].size
	N_test = xtest[:,0].size
	lambd  = np.abs(hyp[0])
	s2_max = np.abs(hyp[1])
	L      = np.abs(hyp[2:])

	N = X.shape[1]
	d2_star = np.zeros((N_test,N_set ))
	for i in range(0,N_dim):
		tmp  = np.ones((N_test,1))
		tmp2 = np.ones((N_set,1))
		d2_star += (np.kron(tmp,X[:,i])- np.kron(tmp2,xtest[:,i]).transpose())**2/(L[i])**2
	k_star  = s2_max*np.exp(-d2_star)



	return k_star.transpose()




# def GP_hyperparameter(xtrain,ytrain,hyp):
	
# 	solution = opt.fmin_slsqp(negative_MarginalLikelihood,hyp, args=(xtrain,ytrain) ,\
# 		bounds = [(0.001, 0.005),(0.00001, 1000),(0.5, 2),(0.5, 2)] ,\
# 		iprint = 0, full_output = 1)
# 	hyp   = solution[0]
# 	value = solution[1]

# 	return hyp

def GP_hyperparameter(xtrain,ytrain,hyp):
	
	solution = opt.fmin_slsqp(negative_MarginalLikelihood,hyp, args=(xtrain,ytrain) ,\
		bounds = [(0.001, 0.005),(10.0, 1000.0),(0.5, 2.0),(0.5, 2.0)] ,\
		iprint = 0, full_output = 1)
	hyp   = solution[0]
	value = solution[1]

	if hyp[0]>0.0005:
		hyp[0]=0.0005
	if hyp[0]<0.0001:
		hyp[0]=0.0001

	if hyp[1]>1000.0:
		hyp[1] = 1000.0
	if hyp[1]<10.0:
		hyp[1] = 10.0
	
	if hyp[2]>2.0:
		hyp[2] = 2.0
	if hyp[2]<0.5:
		hyp[2] = 0.5
	
	if hyp[3]>2.0:
		hyp[3] = 2.0
	if hyp[3]<0.5:
		hyp[3] = 0.5

	return hyp

def negative_MarginalLikelihood(hyp,xtrain,ytrain):
	
	K = compute_covariance_matrix(xtrain,hyp)
	condition = np.linalg.cond(K)
	if math.isnan(condition) :
		print 'condition number of K ', condition
		print '----'
		print 'hyp ',hyp
		print 'K',K
	L = np.linalg.cholesky(K)

	alpha = lin.solve_triangular(L,ytrain,lower = True)
	alpha = lin.solve_triangular(L.transpose(),alpha,lower = False)
	N = alpha.size
	if lin.det(L) == 0.0 :
		ML = 0.0
	else:
		ML = -0.5*N*np.log10(2*math.pi) - 0.5*np.dot(np.transpose(ytrain),alpha) - np.log10(lin.det(L))

	return - ML

def GP_precompute(xtrain,ytrain, s2_max,length_scale,lambd):

	hyp = np.append([lambd , s2_max], length_scale)
	K = compute_covariance_matrix(xtrain,hyp)
	L = np.linalg.cholesky(K)

	alpha = lin.solve_triangular(L,ytrain,lower = True)
	alpha = lin.solve_triangular(L.transpose(),alpha,lower = False)

	return alpha, L



def GP_predict(x,xtrain,alpha, L, s2_max, length_scale, N_test):
	hyp = np.append([0.0 , s2_max], length_scale)

	k_star = compute_kstar(xtrain,hyp,x)
	y = np.dot(np.transpose(k_star),alpha)

	v_tmp    = lin.solve_triangular(L,k_star,lower = True)
	v_tmp2 = s2_max - np.sum(pow(v_tmp,2), axis = 0)
	if (np.all(v_tmp2>0))!= True:
		print 's2_max ',s2_max
		print 'second ',np.sum(pow(v_tmp,2), axis = 0)
		print 'diagonal L', np.diag(L)
		print 'condition number ', np.linalg.cond(L*L.transpose())
	v_tmp2 = np.resize(v_tmp2,(v_tmp2.size,1))

	return y, v_tmp2


class GP:
	'Class of Gaussian Processes'

	def __init__(self,xtrain,ytrain,hyp):
		self.xtrain  = xtrain
		self.ytrain  = ytrain
		self.hyp     = hyp
		self.N_train = xtrain.size

	def display_gp(self):
		print 'xtrain', self.xtrain
		print 'ytrain', self.ytrain
		print 'hyp   ', self.hyp

	def add_training_set(self,x,y):
		self.xtrain = np.vstack((self.xtrain, x))
		self.ytrain = np.append(self.ytrain, y)


	def update_hyperparameter(self):
		self.hyp = GP_hyperparameter(self.xtrain,self.ytrain,self.hyp )

	def update_gp(self):
		self.update_hyperparameter()
		self.alpha, self.L = GP_precompute(self.xtrain,self.ytrain, self.hyp[1],self.hyp[2:],self.hyp[0])


	def evaluate_gp(self,x_test):
		# print 'noise ', self.hyp[0]
		y_test , var_test  = GP_predict(x_test,self.xtrain,self.alpha, self.L, self.hyp[1],self.hyp[2:], self.hyp.size-2)
		y_test = np.reshape(y_test,(y_test.size,1))

		return y_test , var_test


class MF:
	'Class of Multi Fidelity problem'

	def __init__(self,gp_list):
		self.gplist = gp_list

	# def compute_MF posterior mean and variance
	# 


def f1(x):
	return pow(1-x[:,0],2)+ 100*pow(x[:,1]-x[:,0]**2,2)

# def f1(x):
# 	n = x.shape[0]
# 	qoi = np.zeros(n)
# 	for i in range(0,n):
# 		# print x[i,0],x[i,1]
# 		qoi[i] = cfd.call_SU2(x[i,0],x[i,1])
# 		# print 'qoi value ',qoi[i]
# 	return qoi

def c1(x):
	return -x[:,0]**2 - (x[:,1]-1.0)/2.0

def fidelity_variance_1(x):
	# return 10*np.exp(0.7*x[:,0])*(np.sin(x[:,1])**2+0.01)
	return 0.0*x[:,0]+0.001


def f2(x):
	return pow(1-x[:,0],2)+ 100*pow(x[:,1]-x[:,0]**2,2)+ 0.1*np.sin(10*x[:,0]+5*x[:,1])

# def f2(x):
# 	n = x.shape[0]
# 	qoi = np.zeros(n)
# 	for i in range(0,n):
# 		# print x[i,0],x[i,1]
# 		qoi[i] = cfd.call_XFOIL(x[i,0],x[i,1])
# 		# print 'qoi value ',qoi[i]
# 	return qoi

def c2(x):
	return -x[:,0]**2 - (x[:,1]-1.0)/2.0 + 0.1*np.sin(10*x[:,0]+5*x[:,1])

def fidelity_variance_2(x):
	# return 10*np.exp(0.7*x[:,0])*(np.sin(x[:,1])**2+0.001)+10
	return 0.0*x[:,0]+0.01


def f_true_fused(x):
	return (f1(x)/fidelity_variance_1(x) + f2(x)/fidelity_variance_2(x))/(1/fidelity_variance_1(x) + 1/fidelity_variance_2(x))

def c_true_fused(x):
	return (c1(x)/fidelity_variance_1(x) + c2(x)/fidelity_variance_2(x))/(1/fidelity_variance_1(x) + 1/fidelity_variance_2(x))



def MF_surrogate(xtest,gp1,gp2):

	y1, var1 = gp1.evaluate_gp(xtest)
	y2, var2 = gp2.evaluate_gp(xtest)

	varfid1 = fidelity_variance_1(xtest)
	varfid1 = np.reshape(varfid1,(varfid1.shape[0],1))
	varfid2 = fidelity_variance_2(xtest)
	varfid2 = np.reshape(varfid2,(varfid2.shape[0],1))

	vartot1 = var1+varfid1
	vartot2 = var2+varfid2
	var = 1/(1/vartot1 + 1/vartot2)


	y = var *(y1/vartot1+y2/vartot2)
	

	return y, var

def MF_surrogate_for_optim(xtest,gp1,gp2,gpc1,gpc2):
	xtest = np.reshape(xtest,(1,len(xtest)))
	y,var = MF_surrogate(xtest,gp1,gp2)

	fused_constraints = fuse_constraints(xtest,gpc1,gpc2,std_factor = 0.0)

	return y-min(0,fused_constraints)

def MF_surrogate_for_optim_helper(args):
	return MF_surrogate_for_optim(*args)

def parallel_minimize_MF_surrogate(x,gp1,gp2,gpc1,gpc2):
	# spark given number of processes
	p = Pool(4)
	# set each matching item into a tuple
	job_args = [(item_x,gp1,gp2,gpc1,gpc2) for i, item_x in enumerate(x)]
	# map to pool
	result = p.map(MF_surrogate_for_optim_helper, job_args)
	p.close()
	return result

def minimize_MF_surrogate(gp1,gp2,gpc1= None, gpc2 = None, Nstart = 50, bruteforce = 0):
	Ndim = 2
	

	if bruteforce == 0:
		# multi start pool
		x_multistart = (np.random.rand(Nstart,Ndim)-0.5)*4


		# run an optimization for each of them
		for i in range(0,Nstart):
			#optimize
			x = x_multistart[i]
			solution = opt.fmin_slsqp(MF_surrogate_for_optim,x, args=(gp1,gp2,gpc1,gpc2) ,\
			bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)
			x =  solution[0]
			y = solution[1]
	else:
		n_discret = bruteforce
		x = np.linspace(-2.0,2.0,n_discret)
		xx,yy = np.meshgrid(x,x)

		zz = list()	
		for xxx,yyy in zip(np.ravel(xx),np.ravel(yy)):
			x = xxx,yyy
			zz.append(x)
		result = parallel_minimize_MF_surrogate(zz,gp1,gp2,gpc1,gpc2)
		index_min = result.index(min(result))
		x = zz[index_min]
		y = result[index_min]

		solution = opt.fmin_slsqp(MF_surrogate_for_optim,x, args=(gp1,gp2,gpc1,gpc2) ,\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)

		x = solution[0]
		y = solution[1]

	return x,y






def gp_for_optim(xtest,gp,gpc):
	xtest = np.reshape(xtest,(1,len(xtest)))
	y,var = gp.evaluate_gp(xtest)
	yc,varc = gpc.evaluate_gp(xtest)

	return y-min(0,yc)

def gp_for_optim_no_cons(xtest,gp,gpc):
	xtest = np.reshape(xtest,(1,len(xtest)))
	y,var = gp.evaluate_gp(xtest)

	return y

def constraints_for_optim(xtest,gp,gpc):
	xtest = np.reshape(xtest,(1,len(xtest)))
	yc,varc = gpc.evaluate_gp(xtest)

	return yc



def gp_for_optim_helper(args):
	return gp_for_optim(*args)

def parallel_minimize_gp(x,gp,gpc):
	# spark given number of processes
	p = Pool(4)
	# set each matching item into a tuple
	job_args = [(item_x,gp,gpc) for i, item_x in enumerate(x)]
	# map to pool
	result = p.map(gp_for_optim_helper, job_args)
	p.close()
	return result

def minimize_gp(gp,gpc= None, Nstart = 50, bruteforce = 0):
	Ndim = 2
	

	if bruteforce == 0:
		# multi start pool
		x_multistart = (np.random.rand(Nstart,Ndim)-0.5)*4


		# run an optimization for each of them
		for i in range(0,Nstart):
			#optimize
			x = x_multistart[i]
			solution = opt.fmin_slsqp(gp_for_optim,x, args=(gp,gpc) ,\
			bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)
			x =  solution[0]
			y = solution[1]
	else:
		n_discret = bruteforce
		x = np.linspace(-2.0,2.0,n_discret)
		xx,yy = np.meshgrid(x,x)

		zz = list()	
		for xxx,yyy in zip(np.ravel(xx),np.ravel(yy)):
			x = xxx,yyy
			zz.append(x)
		result = parallel_minimize_gp(zz,gp,gpc)
		index_min = result.index(min(result))
		x = zz[index_min]
		y = result[index_min]

		solution = opt.fmin_slsqp(gp_for_optim_no_cons,x, ieqcons = [constraints_for_optim], args=(gp,gpc) ,\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)

		if np.any(solution[0]<-2):
			pass
		elif np.any(solution[0]>2):
			pass
		else:
			x = solution[0]
			y = solution[1]

	return x,y



def fuse_constraints(xtest,gpc1,gpc2, std_factor = 3):
	c1, var1 = gpc1.evaluate_gp(xtest)
	c2, var2 = gpc2.evaluate_gp(xtest)

	varfid1 = fidelity_variance_1(xtest)
	varfid1 = np.reshape(varfid1,(varfid1.shape[0],1))
	varfid2 = fidelity_variance_2(xtest)
	varfid2 = np.reshape(varfid2,(varfid2.shape[0],1))

	vartot1 = var1+varfid1
	vartot2 = var2+varfid2
	var = 1/(1/vartot1 + 1/vartot2)

	c = var *(c1/vartot1+c2/vartot2)
	c = c + std_factor/np.sqrt(1/var1 + 1/var2)
	return 100*c[0,0]

def minimize_cost_variance_criterion(gp1,gp2, cost1, cost2, var_fid_1, var_fid_2, Nstart = 50):
	
	Ndim = 2
	x_multistart = (np.random.rand(Nstart,Ndim)-0.5)*4

	next_x = x_multistart[0]

	min_crit = np.inf
	# run an optimization for each of them
	for i in range(0,Nstart):
		#optimize
		x = x_multistart[i]
		solution = opt.fmin_slsqp(compute_criterion,x, args=(gp1,cost1,var_fid_1) ,\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)
		x =  solution[0]
		criterion = solution[1]
		x = np.reshape(x,(1,Ndim))

		# keep the best EI
		if min_crit > criterion:
			if np.any(x<-2):
				pass
			elif np.any(x>2):
				pass
			else:
				min_crit = criterion
				next_x = x

	for i in range(0,Nstart):
		#optimize
		x = x_multistart[i]
		solution = opt.fmin_slsqp(compute_criterion,x, args=(gp2,cost2,var_fid_2) ,\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)
		x =  solution[0]
		criterion = solution[1]
		x = np.reshape(x,(1,Ndim))

		# keep the best EI
		if min_crit > criterion:
			if np.any(x<-2):
				pass
			elif np.any(x>2):
				pass
			else:
				min_crit = criterion
				next_x = x



	return next_x


def compute_criterion(x,gp, cost, var_fid):
	x = np.reshape(x,(1,len(x)))
	y,varf = gp.evaluate_gp(x)
	criterion = var_fid(x)* cost / varf
	return criterion


def compute_MF_expected_improvement_helper(args):
	return - compute_MF_expected_improvement(*args)

def parallel_compute_MF_expected_improvement(x,gp1,gp2,ymin,gpc1,gpc2):
	# spark given number of processes
	p = Pool(4)
	# set each matching item into a tuple
	job_args = [(item_x,gp1,gp2,ymin,gpc1,gpc2) for i, item_x in enumerate(x)]
	# map to pool
	result = p.map(compute_MF_expected_improvement_helper, job_args)
	p.close()
	return result

def maximize_EI(gp1,gp2,gpc1,gpc2, bruteforce = 0, Nstart = 50): 
# needs to have a consistent use of bound constrains
	Ndim = 2
	
	#compute the ymin
	ytrain1, _ = MF_surrogate(gp1.xtrain, gp1,gp2)
	ytrain2, _ = MF_surrogate(gp2.xtrain, gp1,gp2)
	ytrain_stacked = np.vstack((ytrain1, ytrain2))
	index_min = np.argmin(ytrain_stacked)
	ymin = ytrain_stacked[index_min,0]
	
	max_ei = -1000000 

	if bruteforce == 0: 
		# multi start pool
		x_multistart = (np.random.rand(Nstart,Ndim)-0.5)*4

		next_x = x_multistart[0]

		# run an optimization for each of them
		for i in range(0,Nstart):
			#optimize
			x = x_multistart[i]
			solution = opt.fmin_slsqp(compute_MF_expected_improvement,x, args=(gp1,gp2,ymin,gpc1,gpc2) ,\
			bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)
			x =  solution[0]
			ei = - solution[1]

			# keep the best EI
			if max_ei < ei:
				if np.any(x<-2):
					pass
				elif np.any(x>2):
					pass
				else:
					max_ei = ei
					next_x = x
	else:
		# Discretize domain and find max EI
		# then optimize using it as an initial guess
		n_discret = bruteforce
		x = np.linspace(-2.0,2.0,n_discret)
		xx,yy = np.meshgrid(x,x)

		zz = list()	
		for xxx,yyy in zip(np.ravel(xx),np.ravel(yy)):
			x = xxx,yyy
			zz.append(x)
		result = parallel_compute_MF_expected_improvement(zz,gp1,gp2,ymin,gpc1,gpc2)
		index_max = result.index(max(result))
		next_x = zz[index_max]
		max_ei = result[index_max]

		solution = opt.fmin_slsqp(compute_MF_expected_improvement,next_x, args=(gp1,gp2,ymin,gpc1,gpc2) ,\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)

		if np.any(solution[0]<-2):
			pass
		elif np.any(solution[0]>2):
			pass
		else:
			max_ei = -solution[1]
			next_x = solution[0]

	next_x = np.reshape(next_x,(1,len(next_x)))
	
	return next_x, max_ei, ymin


def compute_MF_expected_improvement(xtest, gp1, gp2, ymin , gpc1 = None, gpc2 = None):
	# Compute the (opposite) of the equivalent of EI for the multifidelity 
	# surrogate. Note that the ymin is not the min of the evaluated design (to avoid)
	# being stuck. It is in fact the min of the MF surrogate at previously evaluated 
	# designs. This allows 'consistency' in the optimization and get rid of 
	# pathological cases

	# y    = performance of the design
	# ymin = minimum of training set
	# s    = standard deviation of the GP (?)
	xtest = np.reshape(xtest,(1,len(xtest)))
	y,_  = MF_surrogate(xtest, gp1, gp2)

	s2 = compute_fused_gp_variance(gp1, gp2, xtest)
	s = np.sqrt(s2)

	# Equation (15) of Jones et al. 1998
	EI = (ymin-y)*stats.norm.cdf((ymin-y)/s) + s*stats.norm.pdf((ymin-y)/s)
	fused_constraints = fuse_constraints(xtest,gpc1,gpc2)

	# Returns the opposite of the expected improvement
	return - EI[0,0]-min(0,fused_constraints)



def compute_expected_improvement_helper(args):
	return - compute_expected_improvement(*args)

def parallel_compute_expected_improvement(x,gp,gpc):
	# spark given number of processes
	p = Pool(4)
	# set each matching item into a tuple
	job_args = [(item_x,gp,gpc) for i, item_x in enumerate(x)]
	# map to pool
	result = p.map(compute_expected_improvement_helper, job_args)
	p.close()
	return result

def compute_expected_improvement(xtest, gp, gpc):
	# Computes the (opposite) of the EI of a GP

	# y    = performance of the design
	# ymin = minimum of training set
	# s    = standard deviation of the GP (?)
	xtest = np.reshape(xtest,(1,len(xtest)))
	ymin = np.min(gp.ytrain)
	y,s2 = gp.evaluate_gp(xtest)
	s = np.sqrt(s2)

	# evaluate the constraint gp
	yc,sc2 = gpc.evaluate_gp(xtest)

	# Equation (15) of Jones et al. 1998
	EI = (ymin-y)*stats.norm.cdf((ymin-y)/s) + s*stats.norm.pdf((ymin-y)/s)
	
	# Returns the opposite of the expected improvement
	return - EI[0,0]-min(0,100*(yc + 3.0*sc2))

def maximize_EI_EGO(gp,gpc, bruteforce = 0, Nstart = 50): 
# needs to have a consistent use of bound constrains
	Ndim = 2
	
	if bruteforce == 0: 
		# multi start pool
		x_multistart = (np.random.rand(Nstart,Ndim)-0.5)*4

		next_x = x_multistart[0]

		# run an optimization for each of them
		for i in range(0,Nstart):
			#optimize
			x = x_multistart[i]
			solution = opt.fmin_slsqp(compute_expected_improvement,x, args=(gp,gpc) ,\
			bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)
			x =  solution[0]
			ei = - solution[1]

			# keep the best EI
			if max_ei < ei:
				if np.any(x<-2):
					pass
				elif np.any(x>2):
					pass
				else:
					max_ei = ei
					next_x = x
	else:
		# Discretize domain and find max EI
		# then optimize using it as an initial guess
		n_discret = bruteforce
		x = np.linspace(-2.0,2.0,n_discret)
		xx,yy = np.meshgrid(x,x)

		zz = list()	
		for xxx,yyy in zip(np.ravel(xx),np.ravel(yy)):
			x = xxx,yyy
			zz.append(x)
		result = parallel_compute_expected_improvement(zz,gp,gpc)
		index_max = result.index(max(result))
		next_x = zz[index_max]
		max_ei = result[index_max]
		solution = opt.fmin_slsqp(compute_expected_improvement,next_x, args=(gp,gpc) ,\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1)

		if np.any(solution[0]<-2):
			pass
		elif np.any(solution[0]>2):
			pass
		else:
			max_ei = -solution[1]
			next_x = solution[0]

	next_x = np.reshape(next_x,(1,len(next_x)))

	return next_x, max_ei



def compute_fused_gp_variance(gp1,gp2,xtest):
	_, v2_gp1 = gp1.evaluate_gp(xtest)
	_, v2_gp2 = gp2.evaluate_gp(xtest)

	v2_fused_gp = 1/(1/v2_gp1+1/v2_gp2)

	return v2_fused_gp

###################################################################

if __name__ == "__main__":

	####################################
	# Main part
	####################################
	N0_train = 5
	n_iter   = 300
	max_call_f1 = 300

	np.random.seed(101) #---------------------for reproductibility
	# np.random.seed(100) #---------------------for reproductibility
	# np.random.seed(103) #---------------------for reproductibility
	cost1   = 3600
	cost2   = 1

	initial_data_set_exist = os.path.isfile('initial_data')

	if initial_data_set_exist:
		print '-----------------------------'
		print 'Loading initial training sets'
		print '-----------------------------'
	 	with open('initial_data', 'r') as filehandler:
		 	gp1,gp2,gpc1,gpc2 = pickle.load(filehandler)

		# This over writes the previous GP1 by adding all the xtrain in one and recomputing the true fused value
		# and the true fused constraints. The name gp1 and gpc1 are kept but there is only one GP being computed here
		xtrain = np.vstack((gp1.xtrain, gp2.xtrain))
		ytrain = f_true_fused(xtrain)
		ctrain = c_true_fused(xtrain)
		hyp    = np.array([0.0001 , 10.0 , .5, .5])

		gp1 = GP(xtrain,ytrain,hyp)
		gp1.update_gp()

		gpc1 = GP(xtrain,ctrain,hyp)
		gpc1.update_gp()


	else :
		print 'Error the initial data set is not available'


	number_call = np.array([0.0, 0.0])
	history_list = list()

	print 	'Starting Optimization'
	print 	'-----------------------------\n'
	print '{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format('Iteration','IS','x_1','x_2','y','ymin','max_ei', 'x1_minMF','x2_minMF','min y_MF')

	i = 0
	dy = 1.0
	while (i < n_iter)& (number_call[0]< max_call_f1 ):
		# What is the next design to sample ?
		xnext, max_ei   = maximize_EI_EGO(gp1,gpc1, bruteforce = 100)
		xmin_MF,ymin_MF = minimize_MF_surrogate(gp1,gp2,gpc1,gpc2, bruteforce = 100)
		if max_ei < 0.00001:
			xnext = np.reshape(xmin_MF,(1,2))
			# xnext =	minimize_cost_variance_criterion(gp1,gp2, cost1, cost2, fidelity_variance_1, fidelity_variance_2)

		try:
			ynext          = f_true_fused(xnext)
			constrainsnext = c_true_fused(xnext)
			gp1.add_training_set(xnext,ynext)
			gp1.update_gp()

			gpc1.add_training_set(xnext, constrainsnext)
			gpc1.update_gp()

			model_used = 1
			number_call[0]+=1
		except IOError, ValueError:
			print 'SU2 FAILED ', xnext


		i = i+1
		
		history_list.append((model_used, xnext))
		

		ymin = 10000 # useless value
	 	with open('history/ego/saved_iter_EGO_'+str(i) , 'w') as filehandler:
			pickle.dump((gp1,gp2,gpc1,gpc2,ymin, history_list,xmin_MF,ymin_MF), filehandler)


		# print '{:3}{:10}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}'.format(i, model_used,xnext[0,0],xnext[0,1], ynext[0], 1.0, max_ei, xmin_MF[0], xmin_MF[1], ymin_MF[0,0])
		print i


	print 'IS 1 calls: ',number_call[0]
	print 'IS 2 calls: ',number_call[1]
	print 'Ratio ', number_call[0]/i

	print '------------------------------'
	print 'Writing data to file'
	print '------------------------------'
	filehandler = open('final_data_ego', 'w') 
	pickle.dump((gp1,gp2,gpc1,gpc2), filehandler)
	