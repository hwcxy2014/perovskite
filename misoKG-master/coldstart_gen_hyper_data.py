import cPickle as pickle
import sys

import numpy as np
from joblib import Parallel, delayed

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from assembleToOrder.assembleToOrder import AssembleToOrderPES
from multifidelity_KG.obj_functions import RosenbrockNoiseFreePES, RosenbrockNewNoiseFreePES
from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4
__author__ = 'jialeiwang'

num_pts = 1000
directory = "/fs/europa/g_pf/pickles/coldstart/data"

# ######### ATO
# # func = AssembleToOrderVanilla(mult=-1.0)
# # func = AssembleToOrderVar3(mult=-1.0)
# func = AssembleToOrderVar4(mult=-1.0)
# points_filename = "{0}/hyper_{1}_points_ato.pickle".format(directory, num_pts)
# data_filename = "{0}/hyper_{1}_points_{2}.pickle".format(directory, num_pts, func.getFuncName())
# ##################
# # generate points
# # search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in func._search_domain]) # this file is used below again and hence should be made available there, too
# # points = search_domain.generate_uniform_random_points_in_domain(num_pts)
# # with open(points_filename, "wb") as file:
# #     pickle.dump(points, file)
# # generate points end
# ##################
#
# ##################
# # generate func evaluations
# print data_filename
# with open(points_filename, 'rb') as file:
#     points = pickle.load(file)
# def parallel_func(pt):
#     return func.evaluate(0, pt)
# with Parallel(n_jobs=10) as parallel:
#     vals = parallel(delayed(parallel_func)(pt) for pt in points)
# noise = func.noise_and_cost_func(0, None)[0] * np.ones(num_pts)
# data = {"points": points, "vals": vals, "noise": noise}
# with open(data_filename, "wb") as file:
#     pickle.dump(data, file)
# # generate func evaluations end
# ##################


########## Rosenbrock
# func = RosenbrockVanilla(mult=1.0)
# func = RosenbrockSinus(mult=1.0)
func = RosenbrockBiased(mult=1.0)
points_filename = "{0}/hyper_{1}_points_rb.pickle".format(directory, num_pts)
data_filename = "{0}/hyper_{1}_points_{2}.pickle".format(directory, num_pts, func.getFuncName())
##################
# generate points
# search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in func._search_domain]) # this file is used below again and hence should be made available there, too
# points = search_domain.generate_uniform_random_points_in_domain(num_pts)
# with open(points_filename, "wb") as file:
#     pickle.dump(points, file)
# generate points end
##################

##################
# generate func evaluations
print data_filename
with open(points_filename, 'rb') as file:
    points = pickle.load(file)
vals = [func.evaluate(0, pt) for pt in points]
noise = func.noise_and_cost_func(0, None)[0] * np.ones(num_pts)
data = {"points": points, "vals": vals, "noise": noise}
with open(data_filename, "wb") as file:
    pickle.dump(data, file)
# generate func evaluations end
##################
