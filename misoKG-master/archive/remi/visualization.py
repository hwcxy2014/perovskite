import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import cPickle as pickle
from class_definition_2d import f_true_fused, c_true_fused
import scipy as sp
import scipy.optimize as opt


if __name__ == "__main__":

	# x = [0.5824,0.5828]
	# solution = opt.fmin_slsqp(f_true_fused ,x,
	# bounds = [(0.5824,0.5828),(0.3375,0.3390)] ,\
	# iprint = 0, full_output = 1)
	# hyp   = solution[0]


	n_points = 100
	# x = np.linspace(0.58260,0.58265,n_points)
	# y = np.linspace(0.33824,0.33828,n_points)

	x = np.linspace(0.5828,0.5830,n_points)
	y = np.linspace(0.3385,0.3386,n_points)




	xx,yy = np.meshgrid(x,y)
	f_fused = np.zeros((n_points,n_points))
	c_fused = np.zeros((n_points,n_points))
	for ii in range(0,n_points):
		for jj in range(0,n_points):
			design = np.zeros((1,2))
			design[0,:] = [xx[ii,jj], yy[ii,jj]]
			f_fused[ii,jj] = f_true_fused(design)
			c_fused[ii,jj] = c_true_fused(design)

	fig= plt.figure()
	cp  = plt.contour(xx,yy,f_fused, 50)
	cp2 = plt.contour(xx,yy,c_fused, 1)
	plt.grid()
	plt.show()

