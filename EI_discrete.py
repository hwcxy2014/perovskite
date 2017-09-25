import numpy
from scipy.stats import norm

def gaussianInference(u0, s0, fn):
	'''
	u0: prior mean, dimension of u0 = n+1
   	s0: prior variance, s0 = [[],[],...,[]]
   	fn: observations, dimension of fn = n
   	assume the point to be inferred correspond to the last row/col
	'''
	u0 = numpy.array(u0)
	s0 = numpy.array(s0)
	dim = len(u0)
	#means of points with obs
	u0n = u0[:-1]
	#cov of points with obs
	s0n = numpy.delete(numpy.delete(s0,-1,0),-1,1)
	s0n_inv = numpy.linalg.inv(s0n)
	#cov between the pt to be inferred with pts with obs
	cov0 = s0[-1,:-1]
	u = u0[dim-1] + (cov0.dot(s0n_inv)).dot(fn-u0n)
	v = s0[-1,-1] - (cov0.dot(s0n_inv)).dot(cov0)
	return u, v
	
def getEI(fn,u,v):
	'''
	fn: observations
	u, v: posteriror mean and variance computed by gaussianInference
	'''
	f_max = max(fn)
	z = (u-f_max)/numpy.sqrt(v)
	EI = (u - f_max) * norm.cdf(z) + numpy.sqrt(v) * norm.pdf(z)

	return EI
	
