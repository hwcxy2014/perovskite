import numpy
import sys

sys.path.append("../")

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.covariance import SquareExponential

from multifidelity_KG.model.hyperparameter_optimization import NormalPrior, hyper_opt
from multifidelity_KG.obj_functions import Rosenbrock
from multifidelity_KG.voi.knowledge_gradient import *

from load_and_store_data import obtainHistoricalDataForEGO

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_shifted import RosenbrockShifted
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted

from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var1 import AssembleToOrderVar1
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4

import sql_util_cs

'''
Hyper-Estimation via MAP estimates
'''

# obj_func_min = RosenbrockVanilla( )
#obj_func_min = AssembleToOrderVanilla( mult=-1.0 )
obj_func_min = AssembleToOrderVanilla( mult=-1.0 )
func_name = obj_func_min.getFuncName()

# to determine the right pickle file
load_historical_data_from_pickle = True
list_IS_to_query = [0]
num_init_pts_each_IS = 200
pathToPickles = "../pickles/csCentered"

init_data_pickle_filename = obj_func_min.getFuncName() + '_' + 'IS_' \
                                        + '_'.join(str(element) for element in list_IS_to_query) + '_' \
                                        + str(num_init_pts_each_IS) + "_points_each_repl_0"

# specific for each acquisition function
hyper_bounds = [(0.01, 100) for i in range(obj_func_min._dim+1)]
num_hyper_multistart = 5
cov = SquareExponential(numpy.ones(obj_func_min._dim+1))

### Obtain points for hyperparam estimation
historical_data = obtainHistoricalDataForEGO(load_historical_data_from_pickle,
                                             obj_func_min, pathToPickles, list_IS_to_query, num_init_pts_each_IS,
                                             init_data_pickle_filename)

### Setup prior for MAP
prior_mean = numpy.concatenate(([numpy.var(historical_data.points_sampled_value)], [1.]*obj_func_min._dim))
prior_sig = numpy.eye(obj_func_min._dim+1) * 25.
prior_sig[0, 0] = 1e6 # Jialei's original value: 5e5
prior = NormalPrior(prior_mean, prior_sig)
hyper_bounds[0] = (1., prior_mean[0] * 2)

# # hyperparam opt
hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
best_f = numpy.inf
for k in range(num_hyper_multistart):
    hyper , f, output = hyper_opt(cov, data=historical_data, init_hyper=multistart_pts[k,:], hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=prior)
    print f
    if f < best_f:
        best_hyper = hyper
        best_f = f

print 'best_hyper=' + str(best_hyper)
print 'best_f= ' + str(best_f)
print "prior mean is: {0}".format(prior_mean)
sql_util_cs.write_array_to_table('ego_hyper_'+func_name, best_hyper)
