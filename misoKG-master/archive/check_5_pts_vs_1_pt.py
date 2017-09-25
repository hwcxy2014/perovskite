from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted

func_list = [RosenbrockVanilla(mult=1.0), RosenbrockSinus(mult=1.0), RosenbrockBiased(mult=1.0), RosenbrockSlightShifted(mult=1.0)]
plot_dir = "/fs/europa/g_pf/pickles/coldstart/plot"
num_pts_list = range(1, 6)

def check_ave_min(func_idx):
    num_repl = 500
    func = func_list[func_idx]
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in func._search_domain])
    min_vals = np.zeros((num_repl, len(num_pts_list)))
    for i, num_pts in enumerate(num_pts_list):
        for repl in range(num_repl):
            points = search_domain.generate_uniform_random_points_in_domain(num_pts)
            min_vals[repl, i] = np.amin([func.evaluate(0, pt) for pt in points])
    return np.mean(min_vals, axis=0).tolist()

if __name__ == "__main__":
    with PdfPages("{0}/rb_check_init_min_val.pdf".format(plot_dir)) as pdf:
        for idx in range(4):
            y = check_ave_min(idx)
            fig = plt.figure()
            plt.plot(num_pts_list, y)
            plt.title(func_list[idx].getFuncName())
            plt.xlabel("number of init points")
            plt.ylabel("best value so far")
            pdf.savefig()
            plt.close()
