import cPickle as pickle
import sys

import numpy as np
from joblib import Parallel, delayed

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted
from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4
__author__ = 'jialeiwang'

argv = sys.argv[1:]
func_name = argv[0]
repl_no = int(argv[1])
func_dict = {
    "rb_van": RosenbrockVanilla(mult=1.0),
    "rb_sinus": RosenbrockSinus(mult=1.0),
    "rb_biased": RosenbrockBiased(mult=1.0),
    "rb_slsh": RosenbrockSlightShifted(mult=1.0),
    "ato_van": AssembleToOrderVanilla(mult=-1.0),
    "ato_var2": AssembleToOrderVar2(mult=-1.0),
    "ato_var3": AssembleToOrderVar3(mult=-1.0),
    "ato_var4": AssembleToOrderVar4(mult=-1.0),
}
func = func_dict[func_name]
num_pts = 1
directory = "/fs/europa/g_pf/pickles/coldstart/data"
data_filename = "{0}/{1}_{2}_points_each_repl_{3}.pickle".format(directory, func.getFuncName(), num_pts, repl_no)
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in func._search_domain])
points = search_domain.generate_uniform_random_points_in_domain(num_pts)
vals = [func.evaluate(0, pt) for pt in points]
noise = func.noise_and_cost_func(0, None)[0] * np.ones(num_pts)
data = {"points": np.array(points), "vals": np.array(vals), "noise": np.array(noise)}
with open(data_filename, "wb") as file:
    pickle.dump(data, file)
