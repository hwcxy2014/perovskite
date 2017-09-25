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
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import call_cfd as cfd
import cPickle as pickle
import os
from class_definition_2d import *
from multiprocessing import Pool

rc('text', usetex=True)
rc('font', family='serif')
rc('font', serif='Computer Modern Roman')
rc('font', size='20')

def produce_plot(filename, i):

	with open(filename, 'r') as filehandler:
		 	gp1,gp2,gpc1,gpc2, ymin, history_list, xmin_MF,ymin_MF = pickle.load(filehandler)

	N_visualisation = 50
	x = np.linspace(-2.0,2.0,N_visualisation)
	xx,yy = np.meshgrid(x,x)
	zz1   = np.zeros((N_visualisation,N_visualisation))
	zzv1  = np.zeros((N_visualisation,N_visualisation))
	ei_is1= np.zeros((N_visualisation,N_visualisation))
	ei_is2= np.zeros((N_visualisation,N_visualisation))
	zzei  = np.zeros((N_visualisation,N_visualisation))
	zz2   = np.zeros((N_visualisation,N_visualisation))
	zzv2  = np.zeros((N_visualisation,N_visualisation))
	zzf   = np.zeros((N_visualisation,N_visualisation))
	zzvf  = np.zeros((N_visualisation,N_visualisation))
	for ii in range(0,N_visualisation):
		for jj in range(0,N_visualisation):
			design = np.zeros((1,2))
			design[0,:] = [xx[ii,jj], yy[ii,jj]]
			zz1[ii,jj],zzv1[ii,jj] = gp1.evaluate_gp(design)
			zzei[ii,jj] =  -compute_MF_expected_improvement(design[0],gp1,gp2,ymin,gpc1,gpc2)
			zz2[ii,jj],zzv2[ii,jj] = gp2.evaluate_gp(design)
			zzf[ii,jj],zzvf[ii,jj] = MF_surrogate(design,gp1,gp2)
			ei_is1[ii,jj] = -compute_expected_improvement(design,gp1)
			ei_is2[ii,jj] = -compute_expected_improvement(design,gp2)
	print '\t value for plot computed'


	fig= plt.figure()
	fig.set_size_inches(30,15)

	ax1 = fig.add_subplot(331, projection='3d')
	ax1.plot_wireframe(xx, yy, zz1)
	ax1.scatter(gp1.xtrain[:,0],gp1.xtrain[:,1] , list(gp1.ytrain), color = 'red')
	ax1.set_xlabel(r'$x_1$')
	ax1.set_ylabel(r'$x_2$')
	ax1.set_zlabel(r'$\mu_{1}$')


	ax2 = fig.add_subplot(332, projection='3d')
	ax2.plot_wireframe(xx, yy, zz2)
	ax2.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1] , list(gp2.ytrain), color = 'black')
	ax2.set_xlabel(r'$x_1$')
	ax2.set_ylabel(r'$x_2$')
	ax2.set_zlabel(r'$\mu_{2}$')

	ax3 = fig.add_subplot(333, projection='3d')
	ax3.plot_wireframe(xx, yy, zzf)
	ax3.plot(gp1.xtrain[:,0],gp1.xtrain[:,1] , list(gp1.ytrain), marker = 'o',ls = '*', color = 'red', markeredgecolor = 'black')
	ax3.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1] , list(gp2.ytrain), color = 'black')
	ax3.set_xlabel(r'$x_1$')
	ax3.set_ylabel(r'$x_2$')
	ax3.set_zlabel(r'$\overline{\mu}$')

	ax4 = fig.add_subplot(334, projection='3d')
	ax4.plot_wireframe(xx, yy, zzv1)
	ax4.set_xlabel(r'$x_1$')
	ax4.set_ylabel(r'$x_2$')
	ax4.set_zlabel(r'$\sigma_{GP,1}^{2}$')

	ax5 = fig.add_subplot(335, projection='3d')
	ax5.plot_wireframe(xx, yy, zzv2)
	ax5.set_xlabel(r'$x_1$')
	ax5.set_ylabel(r'$x_2$')
	ax5.set_zlabel(r'$\sigma_{GP,2}^{2}$')

	ax6 = fig.add_subplot(336, projection='3d')
	ax6.plot_wireframe(xx, yy, zzvf)
	ax6.set_xlabel(r'$x_1$')
	ax6.set_ylabel(r'$x_2$')
	ax6.set_zlabel(r'$\tilde{\sigma}_{GP}^{2}$')


	ax7 = fig.add_subplot(337, projection='3d')
	ax7.plot_wireframe(xx, yy, ei_is1)
	ax7.set_xlabel(r'$x_1$')
	ax7.set_ylabel(r'$x_2$')
	ax7.set_zlabel('EI')

	ax8 = fig.add_subplot(338)#, projection='3d')
	cp = plt.contourf(xx,yy,ei_is2, 20)
	C = plt.contour(xx, yy, ei_is2, 20, colors='black', linewidth=.5)
	plt.xlabel(r'$x_1$', fontsize = 20)
	plt.ylabel(r'$x_2$', fontsize = 20)
	plt.colorbar(cp)
	# ax8.plot_wireframe(xx, yy, ei_is2)
	# ax8.set_xlabel(r'$x_1$')
	# ax8.set_ylabel(r'$x_2$')
	# ax8.set_zlabel('EI')

	ax9 = fig.add_subplot(339)#, projection='3d')
	# ax9.plot_wireframe(xx, yy, zzei)
	cp = plt.contourf(xx,yy,zzei, 20)
	C = plt.contour(xx, yy, zzei, 20, colors='black', linewidth=.5)
	plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	plt.xlabel(r'$x_1$', fontsize = 20)
	plt.ylabel(r'$x_2$', fontsize = 20)
	plt.colorbar(cp)
	# ax9.set_xlabel(r'$x_1$')
	# ax9.set_ylabel(r'$x_2$')
	# # ax9.set_zlabel('EI')
	# ax7.set_zlim([0,0.1])

	plt.savefig('plots_2d/iteration_'+ str(i) +'.pdf', bbox_inches="tight")
	plt.close(fig)

	fig= plt.figure()
	cp = plt.contourf(xx,yy,zzf, 20)
	C = plt.contour(xx, yy, zzf, 20, colors='black', linewidth=.5)
	plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	plt.xlabel(r'$x_1$', fontsize = 20)
	plt.ylabel(r'$x_2$', fontsize = 20)
	plt.colorbar(cp)
	plt.savefig('plots_2d/MF_'+ str(i) +'.pdf', bbox_inches="tight")
	plt.close(fig)

	fig= plt.figure()
	cp = plt.contourf(xx,yy,zzei, 20)
	C = plt.contour(xx, yy, zzei, 20, colors='black', linewidth=.5)
	plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	plt.xlabel(r'$x_1$', fontsize = 20)
	plt.ylabel(r'$x_2$', fontsize = 20)
	plt.colorbar(cp)
	plt.savefig('plots_2d/EI_'+ str(i) +'.pdf', bbox_inches="tight")
	plt.close(fig)

	print '\t plots computed'



def produce_contour_plot(filename, i):

	with open(filename, 'r') as filehandler:
		 	gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)

	xbot = -2.0
	xtop =  2.0
	ybot = -2.0
	ytop =  2.0

	# xbot = 0.5
	# xtop = 0.7
	# ybot = 0.3
	# ytop = 0.4

	# xbot = 0.58260
	# xtop = 0.58265
	# ybot = 0.3382
	# ytop = 0.3383

	# xbot = 0.577
	# xtop = 0.578
	# ybot = 0.3315
	# ytop = 0.3325


	N_visualisation = 50
	x = np.linspace(xbot,xtop,N_visualisation)
	y = np.linspace(ybot,ytop,N_visualisation)
	xx,yy = np.meshgrid(x,y)
	zz1   = np.zeros((N_visualisation,N_visualisation))
	zzv1  = np.zeros((N_visualisation,N_visualisation))
	ei_is1= np.zeros((N_visualisation,N_visualisation))
	ei_is2= np.zeros((N_visualisation,N_visualisation))
	zzei  = np.zeros((N_visualisation,N_visualisation))
	zzeic = np.zeros((N_visualisation,N_visualisation))
	zz2   = np.zeros((N_visualisation,N_visualisation))
	zzv2  = np.zeros((N_visualisation,N_visualisation))
	zzf   = np.zeros((N_visualisation,N_visualisation))
	zzvf  = np.zeros((N_visualisation,N_visualisation))
	true_obj= np.zeros((N_visualisation,N_visualisation))
	true_con= np.zeros((N_visualisation,N_visualisation))
	approx_con = np.zeros((N_visualisation,N_visualisation))
	approx_con_plus = np.zeros((N_visualisation,N_visualisation))

	for ii in range(0,N_visualisation):
		for jj in range(0,N_visualisation):
			design = np.zeros((1,2))
			design[0,:] = [xx[ii,jj], yy[ii,jj]]
			# zz1[ii,jj],zzv1[ii,jj] = gp1.evaluate_gp(design)
			# zzeic[ii,jj] =  -compute_MF_expected_improvement(design[0],gp1,gp2,ymin,gpc1,gpc2)
			# zzei[ii,jj] =  -compute_MF_expected_improvement(design[0],gp1,gp2,ymin,gpc1 = None ,gpc2 = None)
			# zz2[ii,jj],zzv2[ii,jj] = gp2.evaluate_gp(design)
			zzf[ii,jj],zzvf[ii,jj] = MF_surrogate(design,gp1,gp2)
			# ei_is1[ii,jj] = -compute_expected_improvement(design,gp1)
			# ei_is2[ii,jj] = -compute_expected_improvement(design,gp2)
			true_obj[ii,jj] = (f1(design)/ fidelity_variance_1(design) + f2(design)/ fidelity_variance_2(design))/(1/fidelity_variance_1(design) + fidelity_variance_2(design)) 
			true_con[ii,jj] = (c1(design)/ fidelity_variance_1(design) + c2(design)/ fidelity_variance_2(design))/(1/fidelity_variance_1(design) + fidelity_variance_2(design)) 
			approx_con[ii,jj]       = fuse_constraints(design,gpc1,gpc2, std_factor =  0)
			approx_con_plus[ii,jj]  = fuse_constraints(design,gpc1,gpc2, std_factor =  3)

	# fig= plt.figure()
	# fig.set_size_inches(30,20)

	# ax1 = fig.add_subplot(331)
	# plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o',ls = '*', color = 'red',   markeredgecolor = 'black')
	# cp = plt.contourf(xx,yy,zz1, 20)
	# C = plt.contour(xx, yy, zz1, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))

	# ax2 = fig.add_subplot(332)
	# ax2.plot(gp2.xtrain[:,0],gp2.xtrain[:,1], marker = 'o',ls = '*', color = 'black', markeredgecolor = 'black')
	# cp = plt.contourf(xx,yy,zz2, 20)
	# C = plt.contour(xx, yy, zz2, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))

	# ax3 = fig.add_subplot(333)
	# ax3.plot(gp1.xtrain[:,0],gp1.xtrain[:,1] , marker = 'o',ls = '*', color = 'red',   markeredgecolor = 'black')
	# ax3.plot(gp2.xtrain[:,0],gp2.xtrain[:,1] , marker = 'o',ls = '*', color = 'black', markeredgecolor = 'black')
	# ax3.plot(xmin_MF[0]     ,     xmin_MF[1] , marker = 'o',ls = '*', color = 'yellow', markeredgecolor = 'black')
	# # cp = plt.contourf(xx,yy,zzf, 20)
	# # C = plt.contour(xx, yy, zzf, 20, colors='black', linewidth=.5)
	# cp = plt.contourf(xx,yy,approx_con, 20)
	# C = plt.contour(xx, yy, approx_con, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((xbot,xtop))
	# plt.ylim((ybot,ytop))

	# ax4 = fig.add_subplot(334)
	# cp = plt.contourf(xx,yy,zzv1, 20)
	# C = plt.contour(xx, yy, zzv1, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))

	# ax5 = fig.add_subplot(335)
	# cp = plt.contourf(xx,yy,zzv2, 20)
	# C = plt.contour(xx, yy, zzv2, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))


	# ax6 = fig.add_subplot(336)
	# # cp = plt.contourf(xx,yy,zzvf, 20)
	# # C = plt.contour(xx, yy, zzvf, 20, colors='black', linewidth=.5)
	# # plt.xlabel(r'$x_1$', fontsize = 20)
	# # plt.ylabel(r'$x_2$', fontsize = 20)
	# # plt.colorbar(cp)
	# # plt.xlim((-2.0, 2.0))
	# # plt.ylim((-2.0, 2.0))
	# ax6.plot(gp1.xtrain[:,0],gp1.xtrain[:,1] , marker = 'o',ls = '*', color = 'red',   markeredgecolor = 'black', zorder = -1)
	# ax6.plot(gp2.xtrain[:,0],gp2.xtrain[:,1] , marker = 'o',ls = '*', color = 'black', markeredgecolor = 'black', zorder = -2)
	# C6 = plt.contour(xx, yy, approx_con, 1, colors='red', linewidth=.5)
	# cp6 = plt.contourf(xx,yy,zzf, 20)
	# C61 = plt.contour(xx, yy, zzf, 20, colors='black', linewidth=.5)
	# ax6.plot(xmin_MF[0]     ,     xmin_MF[1] , marker = 'o',ls = '*', color = 'yellow', markeredgecolor = 'black', zorder = 1)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((xbot,xtop))
	# plt.ylim((ybot,ytop))



	# ax7 = fig.add_subplot(337)
	# cp = plt.contourf(xx,yy,ei_is1, 20)
	# C = plt.contour(xx, yy, ei_is1, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))

	# ax8 = fig.add_subplot(338)
	# cp = plt.contourf(xx,yy,ei_is2, 20)
	# C = plt.contour(xx, yy, ei_is2, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))

	# ax9 = fig.add_subplot(339)
	# cp = plt.contourf(xx,yy,zzei, 20)
	# C = plt.contour(xx, yy, zzei, 20, colors='black', linewidth=.5)
	# plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red',   markeredgecolor = 'black')
	# plt.plot(gp2.xtrain[:,0],gp2.xtrain[:,1], marker = 'o', ls = '*', color = 'black', markeredgecolor = 'black')
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))


	# plt.savefig('plots_2d/iteration_'+ str(i) +'.pdf', bbox_inches="tight")
	# plt.close(fig)





	# fig= plt.figure()
	# fig.set_size_inches(30,15)

	# ax1 = fig.add_subplot(331, projection='3d')
	# ax1.plot_wireframe(xx, yy, zz1)
	# ax1.scatter(gp1.xtrain[:,0],gp1.xtrain[:,1] , list(gp1.ytrain), color = 'red')
	# ax1.set_xlabel(r'$x_1$')
	# ax1.set_ylabel(r'$x_2$')
	# ax1.set_zlabel(r'$\mu_{1}$')


	# ax2 = fig.add_subplot(332, projection='3d')
	# ax2.plot_wireframe(xx, yy, zz2)
	# ax2.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1] , list(gp2.ytrain), color = 'black')
	# ax2.set_xlabel(r'$x_1$')
	# ax2.set_ylabel(r'$x_2$')
	# ax2.set_zlabel(r'$\mu_{2}$')
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))
	# # ax2.set_zlim((-2.0, 200.0))


	# ax3 = fig.add_subplot(333, projection='3d')
	# ax3.plot_wireframe(xx, yy, zzf)
	# ax3.plot(gp1.xtrain[:,0],gp1.xtrain[:,1] , list(gp1.ytrain), marker = 'o',ls = '*', color = 'red', markeredgecolor = 'black')
	# ax3.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1] , list(gp2.ytrain), color = 'black')
	# ax3.set_xlabel(r'$x_1$')
	# ax3.set_ylabel(r'$x_2$')
	# ax3.set_zlabel(r'$\overline{\mu}$')
	# plt.xlim((xbot, xtop))
	# plt.ylim((ybot, ytop))
	# ax3.set_zlim(0.1, 0.2)

	# ax4 = fig.add_subplot(334, projection='3d')
	# ax4.plot_wireframe(xx, yy, zzv1)
	# ax4.set_xlabel(r'$x_1$')
	# ax4.set_ylabel(r'$x_2$')
	# ax4.set_zlabel(r'$\sigma_{GP,1}^{2}$')

	# ax5 = fig.add_subplot(335, projection='3d')
	# ax5.plot_wireframe(xx, yy, zzv2)
	# ax5.set_xlabel(r'$x_1$')
	# ax5.set_ylabel(r'$x_2$')
	# ax5.set_zlabel(r'$\sigma_{GP,2}^{2}$')

	# ax6 = fig.add_subplot(336, projection='3d')
	# ax6.plot_wireframe(xx, yy, zzvf)
	# ax6.set_xlabel(r'$x_1$')
	# ax6.set_ylabel(r'$x_2$')
	# ax6.set_zlabel(r'$\tilde{\sigma}_{GP}^{2}$')


	# ax7 = fig.add_subplot(337, projection='3d')
	# ax7.plot_wireframe(xx, yy, ei_is1)
	# ax7.set_xlabel(r'$x_1$')
	# ax7.set_ylabel(r'$x_2$')
	# ax7.set_zlabel('EI')

	# ax8 = fig.add_subplot(338)#, projection='3d')
	# cp = plt.contourf(xx,yy,zzeic, 20)
	# C = plt.contour(xx, yy, zzeic, 20, colors='black', linewidth=.5)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# # ax8.plot_wireframe(xx, yy, ei_is2)
	# # ax8.set_xlabel(r'$x_1$')
	# # ax8.set_ylabel(r'$x_2$')
	# # ax8.set_zlabel('EI')

	# ax9 = fig.add_subplot(339)#, projection='3d')
	# # ax9.plot_wireframe(xx, yy, zzei)
	# cp = plt.contourf(xx,yy,zzei, 20)
	# C = plt.contour(xx, yy, zzei, 20, colors='black', linewidth=.5)
	# plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	# plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	# plt.xlim((xbot, xtop))
	# plt.ylim((ybot, ytop))
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# # ax9.set_xlabel(r'$x_1$')
	# # ax9.set_ylabel(r'$x_2$')
	# # # ax9.set_zlabel('EI')
	# # ax7.set_zlim([0,0.1])
	# plt.savefig('plots_2d/iteration_3D_'+ str(i) +'.pdf', bbox_inches="tight")
	# # plt.savefig('plots_2d/iteration_3D.pdf', bbox_inches="tight")
	# plt.close(fig)


	fig= plt.figure()
	fig.set_size_inches(6,5)
	ax3 = fig.add_subplot(111, projection='3d')
	ax3.plot_wireframe(xx, yy, zzf)
	ax3.plot(gp1.xtrain[:,0],gp1.xtrain[:,1] , list(gp1.ytrain), marker = 'o',ls = '*', color = 'red', markeredgecolor = 'black')
	ax3.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1] , list(gp2.ytrain), color = 'black')
	ax3.scatter(xmin_MF[0],xmin_MF[1],ymin_MF , color = 'yellow')
	ax3.set_xlabel(r'$x_1$')
	ax3.set_ylabel(r'$x_2$')
	ax3.set_zlabel(r'$\overline{\mu}$')
	plt.xlim((xbot, xtop))
	plt.ylim((ybot, ytop))
	# ax3.set_zlim(0.1, 0.2)
	# plt.show()
	plt.savefig('plots_2d/MF_3D.pdf', bbox_inches="tight")




	# fig= plt.figure()
	# cp = plt.contourf(xx,yy,np.log(zzf+20), 40)
	# C = plt.contour(xx, yy, np.log(zzf+20), 40, colors='black', linewidth=.5)
	# # np.log(true_obj+2)
	# plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	# plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))
	# plt.savefig('plots_2d/MF_'+ str(i) +'.pdf', bbox_inches="tight")
	# plt.close(fig)

	# fig= plt.figure()
	# cp = plt.contourf(xx,yy,zzei, 20)
	# C = plt.contour(xx, yy, np.log10(zzei), 1, colors='black', linewidth=.5)
	# # plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	# # plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))
	# plt.savefig('plots_2d/EI_'+ str(i) +'.pdf', bbox_inches="tight")
	# plt.close(fig)

	# fig= plt.figure()
	# cp = plt.contourf(xx,yy,zzeic, 20)
	# C = plt.contour(xx, yy, zzeic, 1, colors='black', linewidth=.5)
	# plt.plot(gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	# plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))
	# plt.savefig('plots_2d/EIc_'+ str(i) +'.pdf', bbox_inches="tight")
	# plt.close(fig)



	fig = plt.figure()
	fig.set_size_inches(6,5)
	cp_plus = plt.contourf(xx,yy,approx_con_plus, 0, colors = ['white','blue'], alpha = 0.5, zorder = -1)
	cp      = plt.contourf(xx,yy,approx_con     , 0, colors = ['white','blue'], alpha = 0.2, zorder = -1)
	cp_line = plt.contour(xx,yy,approx_con     , 0, zorder = -1)
	blk  = plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black')
	# rd   = plt.plot(	gp1.xtrain[:,0],gp1.xtrain[:,1], marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black')
	rd   = plt.scatter(gp1.xtrain[:,0],gp1.xtrain[:,1], color = 'red', edgecolors = 'black')
	star = plt.scatter(xmin_MF[0],xmin_MF[1], c = 'yellow', s = 80, marker = "*", zorder = 1)
	F = plt.contour(xx, yy, true_con, 1, colors='black', linewidth=.5, zorder = 0)
	C = plt.contour(xx, yy, np.log(true_obj+2), 40, colors='black', linewidths=.1, zorder = 0)
	# C = plt.contour(xx, yy, np.log(zzf+2)     , 40, colors='red'  , linewidths=.1, zorder = 0)
	p1 = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha = 0.2 , lw = 0.0)
	p2 = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha = 0.5 , lw = 0.0)
	p3,= plt.plot(x,x-10000,'k')
	str_nb_train1 = ' (' + str(gp1.xtrain.shape[0]) + ')'
	str_nb_train2 = ' (' + str(gp2.xtrain.shape[0]) + ')'
	
	plt.legend([p1,p2,p3,star, rd,blk], [r'Feasible $\overline{\mu}_{c} - 3 \overline{\sigma}_{c} \leq 0$', r'Feasible $\overline{\mu}_{c}\leq 0$' \
		 ,r'Real process constraint $\mu_{c}=0$' ,r'MF surrogate optimizer $\overline{\mathbf{x}}^{*}$', 'IS 1 samples' + str_nb_train1 , 'IS 2 samples' + str_nb_train2]\
		 , loc = 2,scatterpoints = 1,prop={'size':10})
	plt.xlabel(r'$x_1$', fontsize = 20)
	plt.ylabel(r'$x_2$', fontsize = 20)
	plt.xlim((xbot, xtop))
	plt.ylim((ybot, ytop))
	# plt.savefig('plots_2d/shade_constrains_'+ str(i) +'.pdf', bbox_inches="tight")
	plt.savefig('plots_2d/shade_constrains.pdf', bbox_inches="tight")
	plt.close(fig)





	fig= plt.figure()
	fig.set_size_inches(10,7)

	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(xx, yy, zzf)
	p2, = ax1.plot(gp2.xtrain[:,0],gp2.xtrain[:,1] , list(gp2.ytrain), marker = 's',ls = '*', color = 'black' , markeredgecolor = 'black')
	p1, = ax1.plot(gp1.xtrain[:,0],gp1.xtrain[:,1] , list(gp1.ytrain), marker = 'o',ls = '*', color = 'red'   , markeredgecolor = 'black')
	p3, = ax1.plot(np.array([xmin_MF[0]]),np.array([xmin_MF[1]]), ymin_MF[0], marker = '*',ls = '*', color = 'yellow', markeredgecolor = 'black', ms = 15.0)
	# ax1.scatter(xmin_MF[0],xmin_MF[1],ymin_MF , color = 'yellow', zorder =1)
	ax1.set_xlabel(r'$x_1$', fontsize = 20)
	ax1.set_ylabel(r'$x_2$', fontsize = 20)
	ax1.set_zlabel(r'$\overline{\mu}$')
	plt.xlim((xbot, xtop))
	plt.ylim((ybot, ytop))
	plt.xticks(np.array((-2.0, -1.0, 0.0, 1.0, 2.0)))
	plt.yticks(np.array((-2.0, -1.0, 0.0, 1.0, 2.0)))
	ax1.set_zticks(np.array((0.0, 1000, 2000, 3000, 4000)))
	plt.legend([p1,p2,p3], ['IS 1 samples' + str_nb_train1 , 'IS 2 samples' + str_nb_train2,r'MF surrogate optimizer $\overline{\mathbf{x}}^{*}$']\
		 , loc = 5,scatterpoints = 1, numpoints = 1 , prop={'size':15}, bbox_to_anchor =(1.5,.5))
	plt.savefig('plots_2d/biplot1.pdf', bbox_inches="tight")


	fig= plt.figure()
	fig.set_size_inches(10,7)
	ax2 = fig.add_subplot(111)
	cp_plus      = plt.contourf(xx,yy,approx_con_plus, 0, colors = ['white','blue'], alpha = 0.4, zorder = -1)
	cp           = plt.contourf(xx,yy,approx_con     , 0, colors = ['white','blue'], alpha = 0.4, zorder = -1)
	cp_plus_line = plt.contour(xx,yy,approx_con_plus, 0, zorder = -1,linewidths=2.0, linestyles = 'dotted' , colors ='black')
	cp_line      = plt.contour(xx,yy,approx_con     , 0, zorder = -1,linewidths=2.0, linestyles = 'dashed' , colors ='black')
	blk  = plt.scatter(gp2.xtrain[:,0],gp2.xtrain[:,1], color = 'black', marker = u's', s = 50)
	rd   = plt.scatter(gp1.xtrain[:,0],gp1.xtrain[:,1], color = 'red', edgecolors = 'black', s = 50)
	star = plt.scatter(xmin_MF[0],xmin_MF[1], c = 'yellow', s = 180, marker = "*", zorder = 1)
	F = plt.contour(xx, yy, true_con, 1, colors='black', linewidth=.5, zorder = 0)
	C = plt.contour(xx, yy, np.log(true_obj+2), 40, colors='black', linewidths=.1, zorder = 0)
	p1 = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha = 0.4 , lw = 0.0)
	p2 = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha = 0.8 , lw = 0.0)
	p3,= plt.plot(x,x-10000,'k')
	p4,= plt.plot(x,x-10000,'k--')
	p5,= plt.plot(x,x-10000,'k', ls =':',lw = 2.0)
	str_nb_train1 = ' (' + str(gp1.xtrain.shape[0]) + ')'
	str_nb_train2 = ' (' + str(gp2.xtrain.shape[0]) + ')'
	
	plt.legend([rd, blk, p2,p1,p3,p4,p5,star], ['IS 1 samples' + str_nb_train1 , 'IS 2 samples' + str_nb_train2,r'Feasible $\overline{\mu}_{c} \leq 0$', r'Feasible $\overline{\mu}_{c}- 3 \overline{\sigma}_{c}\leq 0$' \
		 ,r'Real process constraint $\mu_{c}=0$' ,r'Strict constraint $\overline{\mu}_{c}=0$' ,r'Loose constraint $\overline{\mu}_{c} - 3 \overline{\sigma}_{c}=0$' ,r'MF surrogate optimizer $\overline{\mathbf{x}}^{*}$']\
		 , loc = 5,scatterpoints = 1,prop={'size':15}, bbox_to_anchor =(1.6,.5))
	plt.xlabel(r'$x_1$', fontsize = 20)
	plt.ylabel(r'$x_2$', fontsize = 20)
	plt.xlim((xbot, xtop))
	plt.ylim((ybot, ytop))
	plt.savefig('plots_2d/biplot2.pdf', bbox_inches="tight")



	# colors = ['r','b']
	# IS = [colors[element[0]-1] for element in history_list]
	
	# cmap = cm.jet
	# x =  [element[1][0,0] for element in history_list]
	# y =  [element[1][0,1] for element in history_list]

	# fig= plt.figure()
	# cp = plt.contourf(xx,yy,np.log(true_obj+2), 40, zorder = -1)
	# C = plt.contour(xx, yy, np.log(true_obj+2), 40, colors='black', linewidth=.5, zorder = -1)
	# D = plt.contour(xx, yy, true_con, 1, colors='red', linewidth=.5, zorder = -1)
	# plt.plot(x,y,  marker = 'None', ls = '-', color = 'black', markeredgecolor = 'black', zorder = -1)
	# plt.scatter(x,y, c = IS, zorder = 1)
	# star = plt.scatter(xmin_MF[0],xmin_MF[1], c = 'yellow', s = 80, marker = "*", zorder = 1)
	# plt.xlabel(r'$x_1$', fontsize = 20)
	# plt.ylabel(r'$x_2$', fontsize = 20)
	# plt.colorbar(cp)
	# plt.xlim((-2.0, 2.0))
	# plt.ylim((-2.0, 2.0))
	# plt.savefig('plots_2d/convergence_history_'+ str(i) +'.pdf', bbox_inches="tight")
	# plt.close(fig)


	# print '\t plots computed'
	print 'Finished plot ', i




def plot_all(n_plots,begin = 1):
	for i in range(begin,n_plots):
		print 'Plotting iteration '+ str(i)
		filename = 'history/saved_iter_' + str(i)
		produce_contour_plot(filename, i)
	return 1	

 

# auxiliary funciton to make it work
def produce_contour_plot_helper(args):
	return produce_contour_plot(*args)
 
def parallel_plot_all(n_plots, begin = 1):
	# spark given number of processes
	p = Pool(4)
	# set each matching item into a tuple
	job_args = [('history/saved_iter_' + str(i), i) for i in range(begin,n_plots)]
	# print job_args
	# map to pool
	p.map(produce_contour_plot_helper, job_args)
	p.close()
 	return 1



def convergence_plot(file_number, n_file_EGO = 1):
	
	n_file_MF = file_number

	x_optim = [0.5828, 0.3385]
	x_optim = [0.5775, 0.3325]


	x_optim = [0.57, 0.33]
	x = np.reshape(x_optim, (1,2))

	f = lambda x:   f_true_fused(np.reshape(x, (1,2)))[0]
	c = lambda x:   c_true_fused(np.reshape(x, (1,2)))[0]


	solution = opt.fmin_slsqp(f,x, ieqcons = [c],\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1, iter = 1000, acc = 10e-10)

	x_optim = solution[0]
	print x_optim

	y_optim = f_true_fused(np.reshape(x_optim, (1,2)))
	
	distance = []
	distance_hi = []
	index_hi    = []
	y_MF = []
	y_MF_hi = []
	sigma_tot_2_MF =[]
	sigma_tot_2_MF_hi =[]
	y_true = []
	y_true_hi = []
	for i in range(0,file_number):
		filename = 'history/saved_iter_' + str(i+1)
		# filename = 'history/ego/saved_iter_EGO_' + str(i+1)
		with open(filename, 'r') as filehandler:
				# gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
			 	gp1,gp2,_,_,_,history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
		distance.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
		xmin = np.reshape(xmin_MF, (1,2))
		y = (f1(xmin)/ fidelity_variance_1(xmin) + f2(xmin)/ fidelity_variance_2(xmin))/(1/fidelity_variance_1(xmin) + fidelity_variance_2(xmin)) 
		mean, variance = MF_surrogate(xmin,gp1,gp2)
		y_MF.append(mean[0,0])
		sigma_tot_2_MF.append(variance[0,0])
		y_true.append(y[0])

		if history_list[i][0] == 1:
			distance_hi.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
			index_hi.append(i+1)
			y_MF_hi.append(mean[0,0])
			sigma_tot_2_MF_hi.append(variance[0,0])
			y_true_hi.append(y[0])


	fig = plt.figure()
	fig.set_size_inches(20,10)

	ax1 = fig.add_subplot(221)
	rd,   = plt.semilogy(range(1,file_number+1),	distance, marker = u's', ls = '-', color = 'black', markeredgecolor = 'black')
	rd2,  = plt.semilogy(index_hi, distance_hi, marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black', ms =8.0)
	plt.legend([rd2,rd], [r'IS 1 queried',r'IS 2 queried']\
		 , loc = 1,numpoints = 1,prop={'size':14})
	plt.xlabel(r'Iteration $n$', fontsize = 20)
	plt.ylabel(r'$||\mathbf{x}^{*}_{n} - \mathbf{x}^{*}||$', fontsize = 20)
	

	ax2 = fig.add_subplot(222)
	rd2  = plt.semilogy(range(1,len(index_hi)+1),distance_hi, marker = 'o', ls = '-', color = 'red', markeredgecolor = 'black', ms =8.0)
	plt.xlabel(r'IS 1 calls $k$', fontsize = 20)
	plt.ylabel(r'$||\mathbf{x}^{*}_{k} - \mathbf{x}^{*}||$', fontsize = 20)
	plt.xlim((0,1+len(index_hi)))
	
	ax4 = fig.add_subplot(224)
	sc4 = plt.scatter(range(1,1+len(index_hi)),y_true_hi, color = 'blue')
	hdl4 = plt.errorbar(range(1,1+len(index_hi)),y_MF_hi,ls='None', yerr= 3*np.sqrt(sigma_tot_2_MF_hi),\
		fmt='o',mfc='none',mec='red',lw=0.5, color ='red')
	plt.legend([hdl4,sc4], [r'$\overline{\mu}(\mathbf{x}^{*}_{k}) \pm 3 \overline{\sigma}(\mathbf{x}^{*}_{k})$ IS 1 queried'\
		 ,r'$\mu(\mathbf{x}^{*}_{k})$'], loc = 1,numpoints = 1, scatterpoints = 1, prop={'size':15})
	# plt.ylim((-2.0, 4.0))
	plt.xlim((0,1+len(index_hi)))
	# plt.xlim((0,25))
	plt.xlabel(r'IS 1 calls $k$', fontsize = 20)
	plt.ylabel(r'$\overline{\mu}(\mathbf{x}^{*}_{k})$', fontsize = 20)



	distance = []
	y_true = []
	y_MF = []
	sigma_tot_2_MF =[]
	for i in range(0,n_file_EGO):
		filename = 'history/ego/saved_iter_EGO_' + str(i+1)
		with open(filename, 'r') as filehandler:
				# gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
			 	gp1,gp2,_,_,_,history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
		
		distance.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
		xmin = np.reshape(xmin_MF, (1,2))
		y = f1(xmin)
		mean, variance = gp1.evaluate_gp(xmin)
		var_fid = fidelity_variance_1(xmin)
		y_MF.append(mean[0,0])
		sigma_tot_2_MF.append(variance[0,0]+ var_fid[0])
		y_true.append(y[0])

	ax3 = fig.add_subplot(223)
	rd_EGO,   = plt.semilogy(range(1,len(distance)+1),	distance, marker = u'd', ls = '-', color = 'green', markeredgecolor = 'black')

	distance = []
	distance_hi = []
	index_hi    = []
	y_MF = []
	y_MF_hi = []
	sigma_tot_2_MF =[]
	sigma_tot_2_MF_hi =[]
	y_true = []
	y_true_hi = []
	for i in range(0,n_file_MF):
		filename = 'history/saved_iter_' + str(i+1)
		# filename = 'history/ego/saved_iter_EGO_' + str(i+1)
		with open(filename, 'r') as filehandler:
				# gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
			 	gp1,gp2,_,_,_,history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
		distance.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
		xmin = np.reshape(xmin_MF, (1,2))
		y = (f1(xmin)/ fidelity_variance_1(xmin) + f2(xmin)/ fidelity_variance_2(xmin))/(1/fidelity_variance_1(xmin) + fidelity_variance_2(xmin)) 
		mean, variance = MF_surrogate(xmin,gp1,gp2)
		y_MF.append(mean[0,0])
		sigma_tot_2_MF.append(variance[0,0])
		y_true.append(y[0])

		if history_list[i][0] == 1:
			distance_hi.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
			index_hi.append(i+1)
			y_MF_hi.append(mean[0,0])
			sigma_tot_2_MF_hi.append(variance[0,0])
			y_true_hi.append(y[0])


	rd_MF, = plt.semilogy(range(1,len(index_hi)+1),distance_hi, marker = 'o', ls = '-', color = 'red', markeredgecolor = 'black', ms =8.0)

	plt.legend([rd_MF, rd_EGO], [r'MF algorithm',r'EGO algorithm']\
		 , loc = 1,numpoints = 1, scatterpoints = 1,prop={'size':15})

	plt.xlabel(r'IS 1 calls $k$', fontsize = 20)
	plt.ylabel(r'$||\mathbf{x}^{*}_k - \mathbf{x}^{*}||$', fontsize = 20)



	# ax3 = fig.add_subplot(223)
	# sc3 = plt.scatter(range(1,file_number+1),y_true, color = 'blue')
	# hdl3 = plt.errorbar(range(1,file_number+1),y_MF,ls='None', yerr= 3*np.sqrt(sigma_tot_2_MF),\
	# 	fmt='o',mfc='none',mec='black',lw=0.5, color ='black')
	# hdl4 = plt.errorbar(index_hi,y_MF_hi,ls='None', yerr= 3*np.sqrt(sigma_tot_2_MF_hi),\
	# 	fmt='o',mfc='none',mec='red',lw=0.5, color ='red')
	# plt.legend([hdl4,hdl3,sc3], [r'$\overline{\mu}(\mathbf{x}^{*}_{n}) \pm 3 \overline{\sigma}(\mathbf{x}^{*}_{n})$ IS 1 queried',r'$\overline{\mu}(\mathbf{x}^{*}_{n}) \pm 3 \overline{\sigma}(\mathbf{x}^{*}_{n})$ IS 2 queried'\
	# 	 ,r'$\mu(\mathbf{x}^{*}_{n})$'], loc = 1,numpoints = 1, scatterpoints = 1, prop={'size':15})
	# # plt.ylim((-2.0, 4.0))
	# plt.xlim((0,300))
	# plt.xlabel(r'Iteration $n$', fontsize = 20)
	# plt.ylabel(r'$\overline{\mu}(\mathbf{x}^{*}_{n})$', fontsize = 20)
	

	plt.savefig('plots_2d/convergence.pdf', bbox_inches="tight")
	plt.close(fig)

	return 1


def convergence_plot_cce(file_number, n_file_EGO = 1):
	
	n_file_MF = file_number

	x_optim = [0.5828, 0.3385]
	x_optim = [0.5775, 0.3325]


	x_optim = [0.57, 0.33]
	x = np.reshape(x_optim, (1,2))

	f = lambda x:   f_true_fused(np.reshape(x, (1,2)))[0]
	c = lambda x:   c_true_fused(np.reshape(x, (1,2)))[0]


	solution = opt.fmin_slsqp(f,x, ieqcons = [c],\
		bounds = [(-2.0,2.0),(-2.0,2.0)] , iprint = 0, full_output = 1, iter = 1000, acc = 10e-10)

	x_optim = solution[0]
	print x_optim

	y_optim = f_true_fused(np.reshape(x_optim, (1,2)))
	
	distance = []
	distance_hi = []
	index_hi    = []
	y_MF = []
	y_MF_hi = []
	sigma_tot_2_MF =[]
	sigma_tot_2_MF_hi =[]
	y_true = []
	y_true_hi = []
	for i in range(0,file_number):
		filename = 'history/saved_iter_' + str(i+1)
		# filename = 'history/ego/saved_iter_EGO_' + str(i+1)
		with open(filename, 'r') as filehandler:
				# gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
			 	gp1,gp2,_,_,_,history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
		distance.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
		xmin = np.reshape(xmin_MF, (1,2))
		y = (f1(xmin)/ fidelity_variance_1(xmin) + f2(xmin)/ fidelity_variance_2(xmin))/(1/fidelity_variance_1(xmin) + fidelity_variance_2(xmin)) 
		mean, variance = MF_surrogate(xmin,gp1,gp2)
		y_MF.append(mean[0,0])
		sigma_tot_2_MF.append(variance[0,0])
		y_true.append(y[0])

		if history_list[i][0] == 1:
			distance_hi.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
			index_hi.append(i+1)
			y_MF_hi.append(mean[0,0])
			sigma_tot_2_MF_hi.append(variance[0,0])
			y_true_hi.append(y[0])


	fig = plt.figure()
	fig.set_size_inches(30,7.5)

	ax1 = fig.add_subplot(121)
	rd,   = plt.semilogy(range(1,file_number+1),	distance, marker = u's', ls = '-', color = 'black', markeredgecolor = 'black')
	rd2,  = plt.semilogy(index_hi, distance_hi, marker = 'o', ls = '*', color = 'red', markeredgecolor = 'black', ms =8.0)
	plt.legend([rd2,rd], [r'IS 1 queried',r'IS 2 queried']\
		 , loc = 1,numpoints = 1,prop={'size':24})
	plt.xlabel(r'Iteration $n$', fontsize = 20)
	plt.ylabel(r'$||\mathbf{x}^{*}_{n} - \mathbf{x}^{*}||$', fontsize = 20)
	

	# ax2 = fig.add_subplot(222)
	# rd2  = plt.semilogy(range(1,len(index_hi)+1),distance_hi, marker = 'o', ls = '-', color = 'red', markeredgecolor = 'black', ms =8.0)
	# plt.xlabel(r'IS 1 calls $k$', fontsize = 20)
	# plt.ylabel(r'$||\mathbf{x}^{*}_{k} - \mathbf{x}^{*}||$', fontsize = 20)
	# plt.xlim((0,1+len(index_hi)))
	
	# ax4 = fig.add_subplot(224)
	# sc4 = plt.scatter(range(1,1+len(index_hi)),y_true_hi, color = 'blue')
	# hdl4 = plt.errorbar(range(1,1+len(index_hi)),y_MF_hi,ls='None', yerr= 3*np.sqrt(sigma_tot_2_MF_hi),\
	# 	fmt='o',mfc='none',mec='red',lw=0.5, color ='red')
	# plt.legend([hdl4,sc4], [r'$\overline{\mu}(\mathbf{x}^{*}_{k}) \pm 3 \overline{\sigma}(\mathbf{x}^{*}_{k})$ IS 1 queried'\
	# 	 ,r'$\mu(\mathbf{x}^{*}_{k})$'], loc = 1,numpoints = 1, scatterpoints = 1, prop={'size':15})
	# # plt.ylim((-2.0, 4.0))
	# plt.xlim((0,1+len(index_hi)))
	# # plt.xlim((0,25))
	# plt.xlabel(r'IS 1 calls $k$', fontsize = 20)
	# plt.ylabel(r'$\overline{\mu}(\mathbf{x}^{*}_{k})$', fontsize = 20)



	distance = []
	y_true = []
	y_MF = []
	sigma_tot_2_MF =[]
	for i in range(0,n_file_EGO):
		filename = 'history/ego/saved_iter_EGO_' + str(i+1)
		with open(filename, 'r') as filehandler:
				# gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
			 	gp1,gp2,_,_,_,history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
		
		distance.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
		xmin = np.reshape(xmin_MF, (1,2))
		y = f1(xmin)
		mean, variance = gp1.evaluate_gp(xmin)
		var_fid = fidelity_variance_1(xmin)
		y_MF.append(mean[0,0])
		sigma_tot_2_MF.append(variance[0,0]+ var_fid[0])
		y_true.append(y[0])

	ax3 = fig.add_subplot(122)
	rd_EGO,   = plt.semilogy(range(1,len(distance)+1),	distance, marker = u'd', ls = '-', color = 'green', markeredgecolor = 'black')

	distance = []
	distance_hi = []
	index_hi    = []
	y_MF = []
	y_MF_hi = []
	sigma_tot_2_MF =[]
	sigma_tot_2_MF_hi =[]
	y_true = []
	y_true_hi = []
	for i in range(0,n_file_MF):
		filename = 'history/saved_iter_' + str(i+1)
		# filename = 'history/ego/saved_iter_EGO_' + str(i+1)
		with open(filename, 'r') as filehandler:
				# gp1,gp2,gpc1,gpc2, ymin, history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
			 	gp1,gp2,_,_,_,history_list,xmin_MF,ymin_MF = pickle.load(filehandler)
		distance.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
		xmin = np.reshape(xmin_MF, (1,2))
		y = (f1(xmin)/ fidelity_variance_1(xmin) + f2(xmin)/ fidelity_variance_2(xmin))/(1/fidelity_variance_1(xmin) + fidelity_variance_2(xmin)) 
		mean, variance = MF_surrogate(xmin,gp1,gp2)
		y_MF.append(mean[0,0])
		sigma_tot_2_MF.append(variance[0,0])
		y_true.append(y[0])

		if history_list[i][0] == 1:
			distance_hi.append(np.sqrt(np.power(xmin_MF[0]-x_optim[0],2.0) + np.power(xmin_MF[1]-x_optim[1],2.0)))
			index_hi.append(i+1)
			y_MF_hi.append(mean[0,0])
			sigma_tot_2_MF_hi.append(variance[0,0])
			y_true_hi.append(y[0])


	rd_MF, = plt.semilogy(range(1,len(index_hi)+1),distance_hi, marker = 'o', ls = '-', color = 'red', markeredgecolor = 'black', ms =8.0)

	plt.legend([rd_MF, rd_EGO], [r'MF algorithm',r'EGO algorithm']\
		 , loc = 1,numpoints = 1, scatterpoints = 1,prop={'size':24})

	plt.xlabel(r'IS 1 calls $k$', fontsize = 20)
	plt.ylabel(r'$||\mathbf{x}^{*}_k - \mathbf{x}^{*}||$', fontsize = 20)



	# ax3 = fig.add_subplot(223)
	# sc3 = plt.scatter(range(1,file_number+1),y_true, color = 'blue')
	# hdl3 = plt.errorbar(range(1,file_number+1),y_MF,ls='None', yerr= 3*np.sqrt(sigma_tot_2_MF),\
	# 	fmt='o',mfc='none',mec='black',lw=0.5, color ='black')
	# hdl4 = plt.errorbar(index_hi,y_MF_hi,ls='None', yerr= 3*np.sqrt(sigma_tot_2_MF_hi),\
	# 	fmt='o',mfc='none',mec='red',lw=0.5, color ='red')
	# plt.legend([hdl4,hdl3,sc3], [r'$\overline{\mu}(\mathbf{x}^{*}_{n}) \pm 3 \overline{\sigma}(\mathbf{x}^{*}_{n})$ IS 1 queried',r'$\overline{\mu}(\mathbf{x}^{*}_{n}) \pm 3 \overline{\sigma}(\mathbf{x}^{*}_{n})$ IS 2 queried'\
	# 	 ,r'$\mu(\mathbf{x}^{*}_{n})$'], loc = 1,numpoints = 1, scatterpoints = 1, prop={'size':15})
	# # plt.ylim((-2.0, 4.0))
	# plt.xlim((0,300))
	# plt.xlabel(r'Iteration $n$', fontsize = 20)
	# plt.ylabel(r'$\overline{\mu}(\mathbf{x}^{*}_{n})$', fontsize = 20)
	

	plt.savefig('plots_2d/convergence_cce.pdf', bbox_inches="tight")
	plt.close(fig)

	return 1

if __name__ == "__main__":

	iterationmax = 300
	n_file_EGO = 300
	# plot_all(200, begin = 1)

	# convergence_plot(iterationmax,n_file_EGO = n_file_EGO)
	convergence_plot_cce(iterationmax,n_file_EGO = n_file_EGO)

	# parallel_plot_all(iterationmax+1, begin = iterationmax)


