import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

from moe.optimal_learning.python.cpp_wrappers.covariance import MixedSquareExponential as cppMixedSquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcessNew
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

from data_io import construct_hist_data_from_pickle, match_data_filename, check_file_legitimate

hyper_dir = "/fs/europa/g_pf/pickles/coldstart/hyper"
data_dir = "/fs/europa/g_pf/pickles/coldstart/data"
plot_dir = "/fs/europa/g_pf/pickles/coldstart/plot"

def compute_correlation_info_source(the_point, info_1, info_2, which_dim, x_coords, mkg_model):
    cor = np.zeros(len(x_coords))
    local_the_point = np.copy(the_point)
    for i, x in enumerate(x_coords):
        local_the_point[which_dim] = x
        cov_mat = mkg_model.compute_variance_of_points(np.array([np.concatenate(([info_1], local_the_point)), np.concatenate(([info_2], local_the_point))]))
        cor[i] = cov_mat[0, 1] / np.sqrt(cov_mat[0,0] * cov_mat[1,1])
    if np.amax(cor) > 1.0:
        print cor
        raise RuntimeError("cor > 1")
    return cor

def compute_correlation_delta_gp(the_point, info_1, info_2, which_dim, x_coords, mkg_model):
    cor = np.zeros(len(x_coords))
    local_the_point = np.copy(the_point)
    for i, x in enumerate(x_coords):
        local_the_point[which_dim] = x
        pt_0 = np.concatenate(([0], local_the_point))
        pt_1 = np.concatenate(([info_1], local_the_point))
        pt_2 = np.concatenate(([info_2], local_the_point))
        cov_mat = mkg_model.compute_variance_of_points(np.array([pt_0, pt_1, pt_2]))
        cor[i] = (cov_mat[1,2] - cov_mat[0,1] - cov_mat[0,2] + cov_mat[0,0]) / np.sqrt((cov_mat[1,1] - 2.*cov_mat[0,1] + cov_mat[0,0]) * (cov_mat[2,2] - 2.*cov_mat[0,2] + cov_mat[0,0]))
    return cor

def plot_cor(x_coords, cor_IS, cor_delta_gp, dim, plot_dir, func_name):
    with PdfPages("{0}/{1}_cor_is.pdf".format(plot_dir, func_name)) as pdf:
        for which_dim in range(dim):
            plt.figure()
            plt.plot(x_coords, np.mean(cor_IS[:, :, which_dim], axis=0))
            plt.title("dim {0}".format(which_dim))
            plt.ylim(0, 1)
            pdf.savefig()
            plt.close()
    with PdfPages("{0}/{1}_cor_delta_gp.pdf".format(plot_dir, func_name)) as pdf:
        for which_dim in range(dim):
            plt.figure()
            plt.plot(x_coords, np.mean(cor_delta_gp[:, :, which_dim], axis=0))
            plt.ylim(0, 1)
            plt.title("dim {0}".format(which_dim))
            pdf.savefig()
            plt.close()

def plot_ato_cor(num_points, num_discretization):
    dim = 8
    num_func = 4
    num_repl = 100
    search_domain = TensorProductDomain([ClosedInterval(0.0, 20.0) for i in range(dim)])
    average_points = search_domain.generate_uniform_random_points_in_domain(num_points)
    func_name_0_list = ["vanilla", "var2", "var3", "var4"]
    func_name_1_list = ["var3", "var4", "vanilla", "var2"]
    func_name_2_list = ["var4", "vanilla", "var2", "var3"]
    with open("{0}/mkg_ato.pickle".format(hyper_dir), 'rb') as f:
        data = pickle.load(f)
    hyper_param = data['hyperparam']
    kg_cov_cpp = cppMixedSquareExponential(hyperparameters=hyper_param)
    info_1 = 1
    info_2 = 2
    x_coords = np.linspace(0.0, 20.0, num=num_discretization)
    cor_IS = np.zeros((num_func*num_repl*num_points, num_discretization, dim))
    cor_delta_gp = np.zeros((num_func*num_repl*num_points, num_discretization, dim))
    count = 0
    for func_idx in range(num_func):
        for repl_no in range(num_repl):
            hist_data = construct_hist_data_from_pickle(dim=dim, directory=data_dir, IS_filename_dict={0:"kg_atoC_{0}_repl_{1}".format(func_name_0_list[func_idx], repl_no), 1:"kg_atoC_{0}_repl_{1}".format(func_name_1_list[func_idx], repl_no), 2:"kg_atoC_{0}_repl_{1}".format(func_name_2_list[func_idx], repl_no)}, combine_IS=True, sign=-1.0)
            kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, hist_data, num_IS_in=2)
            for the_point in average_points:
                for which_dim in range(dim):
                    cor_IS[count, :, which_dim] = compute_correlation_info_source(the_point, info_1, info_2, which_dim, x_coords, kg_gp_cpp)
                    cor_delta_gp[count, :, which_dim] = compute_correlation_delta_gp(the_point, info_1, info_2, which_dim, x_coords, kg_gp_cpp)
                count += 1
                print "ato, ct {0}".format(count)
    with open("{0}/ato_plot_data.pickle".format(plot_dir), "wb") as f:
        pickle.dump({"cor_is": cor_IS, "cor_delta": cor_delta_gp, "x": x_coords}, f)
    plot_cor(x_coords, cor_IS, cor_delta_gp, dim, plot_dir, "ato")

def plot_rb_cor(num_points, num_discretization):
    dim = 2
    num_func = 3
    num_repl = 100
    search_domain = TensorProductDomain([ClosedInterval(-2., 2.) for _ in range(dim)])
    average_points = search_domain.generate_uniform_random_points_in_domain(num_points)
    func_name_0_list = ["van", "sin", "bias"]
    func_name_1_list = ["sin", "bias", "van"]
    func_name_2_list = ["bias", "van", "sin"]
    with open("{0}/mkg_rb.pickle".format(hyper_dir), 'rb') as f:
        data = pickle.load(f)
    hyper_param = data['hyperparam']
    hyper_param = np.concatenate((hyper_param, hyper_param[-3:]))
    print "hyper {0}".format(hyper_param)
    kg_cov_cpp = cppMixedSquareExponential(hyperparameters=hyper_param)
    info_1 = 1
    info_2 = 2
    x_coords = np.linspace(-2., 2., num=num_discretization)
    cor_IS = np.zeros((num_func*num_repl*num_points, num_discretization, dim))
    cor_delta_gp = np.zeros((num_func*num_repl*num_points, num_discretization, dim))
    count = 0
    for func_idx in range(num_func):
        for repl_no in range(num_repl):
            hist_data = construct_hist_data_from_pickle(dim=dim, directory=data_dir, IS_filename_dict={0:"kg_rbC{0}N_repl_{1}".format(func_name_0_list[func_idx], repl_no), 1:"kg_rbC{0}N_repl_{1}".format(func_name_1_list[func_idx], repl_no), 2:"kg_rbC{0}N_repl_{1}".format(func_name_2_list[func_idx], repl_no)}, combine_IS=True, sign=-1.0)
            kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, hist_data, num_IS_in=2)
            for the_point in average_points:
                for which_dim in range(dim):
                    cor_IS[count, :, which_dim] = compute_correlation_info_source(the_point, info_1, info_2, which_dim, x_coords, kg_gp_cpp)
                    cor_delta_gp[count, :, which_dim] = compute_correlation_delta_gp(the_point, info_1, info_2, which_dim, x_coords, kg_gp_cpp)
                count += 1
                print "rb, ct {0}".format(count)
    with open("{0}/rb_plot_data.pickle".format(plot_dir), "wb") as f:
        pickle.dump({"cor_is": cor_IS, "cor_delta": cor_delta_gp, "x": x_coords}, f)
    plot_cor(x_coords, cor_IS, cor_delta_gp, dim, plot_dir, "rb")

if __name__ == "__main__":
    # plot_rb_cor(100, 100)
    plot_ato_cor(100, 100)

