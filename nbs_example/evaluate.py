#! /usr/local/python/2.7.3/bin/python

"""
This script searches for a local minimum of the following function:
    d(n_0, n_1, C) := |C - C'| + dist(n_0, n_1),
where C' is the output of the map Tau(n_0, n_1, C) defined in the main paper,
and dist(n_0, n_1) is the Euclidean distance between (n_0, n_1) and the set
T(n_0, n_1, C), which is the set of optimal thresholds for the left out agent
gven all other agents follow the same threshold strategy (n_0, n_1). This
script searches local mimimum of this function using the Nelder-Mead
algorithm with a user input initial point.

When d(n_0, n_1, C) = 0, (n_0, n_1, C) is a meanfield equilibrium.

Parameters:
    N: Type: scalar. The truncation place of the state space of the Markov chain MC(n_0, n_1,
kappa)
    beta: Type: scalar. The expected number of agents at a location kept fixed as the system
grows to infinity
    lam: Type: scalar. The rate of the Poisson clock for each agent
    mu_0, mu_1: Type: scalar. The rate of the resource chain.
    gamma: Type: scalar. The survival rate of an agent at the time when a decision has to be
made.
    f: Type: array of size N. The resource sharing function. 

Usage: python evaluate.py n0 n1 C  [-N N] [-beta BETA] [-gamma GAMMA] [-lam
LAMBDA] [-mu0 MU0] [-mu1 MU1] [-f F] 
F can only be 1,2 or 3. If 1, f(n) = 1/n,; if 2, f(n) = 1/sqrt(n); if 3, f(n) =
1/n^2.
"""

import sys,os
from time import time

from math import sqrt
from scipy.optimize import minimize
import numpy as np

from mfe import update_parameters


def minimize_loss(N, beta, lam, gamma, mu_0, mu_1, f, C, n_0, n_1):
    """
    This function finds a local minimum using the Nelder-Mead algorithm
    of the function loss(C, n_0, n_1)
    """
    fun = lambda x: loss(x[0], x[1], x[2], N, beta, lam, mu_0, mu_1, gamma,
            f)
    res = minimize(fun, [C, n_0, n_1], method='Nelder-Mead', options={'maxiter':
        500, 'maxfev': 500, 'xtol': 1e-5, 'ftol': 1e-5})
    [C_opt, n0_opt, n1_opt] = res.x
    opt_loss = res.fun
    return [C_opt, n0_opt, n1_opt, opt_loss]

def distance(n_0, n_1, C, n0_min, n0_max, n1_min, n1_max, C_new):
    """
    Compute the following:
        |C - C_new| + dist((n_0, n_1), [n0_min, n0_max]x[n1_min, n1_max]),
    dist((x, y), [x1, x2]x[y1, y2]) is the Euclidean distance between the 
    point (x, y) and the convex set [x1, x2]x[y1, y2].
    """
    d = abs(C - C_new)
    if n_0 < n0_min and n_1 < n1_min:
        d += sqrt((n0_min - n_0)**2 + (n1_min - n_1)**2)
    elif n_0 >= n0_min and n_0 <= n0_max and n_1 < n1_min:
        d += n1_min - n_1
    elif n_0 > n0_max and n_1 < n1_min:
        d += sqrt((n_0 - n0_max)**2 + (n1_min - n_1)**2)
    elif n_0 < n0_min and n_1 >= n1_min and n_1 <= n1_max:
        d += n0_min - n_0
    elif n_0 > n0_max and n_1 >= n1_min and n_1 <= n1_max:
        d += n_0 - n0_max
    elif n_0 < n0_min and n_1 > n1_max:
        d += sqrt((n0_min - n_0)**2 + (n_1 - n1_max)**2)
    elif n_0 >= n0_min and n_0 <= n0_max and n_1 > n1_max:
        d += n_1 - n1_max
    elif n_0 > n0_max and n_1 > n1_max:
        d += sqrt((n_0 - n0_max)**2 + (n_1 - n1_max)**2)

    return d


def loss(C, n_0, n_1, N, beta, lam, mu_0, mu_1, gamma, f, eps_kappa=1e-4,
    eps_VI=1e-4):
    """
    Compute the distance between (n_0, n_1, C) and Tau(n_0, n_1, C) for a
    particular input of n_0, n_1, C.
    """
    # Assign +inf to inputs outside the feasible region
    if C <= 0 or n_0 <= 0 or n_1 <= 0 or n_0 >= N-1 or n_1 >= N-1:
        return 1e10
    [C_new, n0_min, n0_max, n1_min, n1_max] = update_parameters(N, beta, lam,
        mu_0, mu_1, gamma, f, C, n_0, n_1, eps_kappa=1e-4, eps_VI=1e-4)
    return distance(n_0, n_1, C, n0_min, n0_max, n1_min, n1_max, C_new)


def do_run(N, beta, lam, gamma, mu_0, mu_1, f, f_type, C, n_0, n_1, directory):
    start_time = time()
    [C_opt, n0_opt, n1_opt, opt_loss] = minimize_loss(N, beta, lam, gamma,
        mu_0, mu_1, f, C, n_0, n_1)
    run_time = time() - start_time

    prefix = '/beta_{0}_lambda_{1}_mu0_{2}_mu1_{3}_f_type_{4}'.format(beta,
        lam, mu_0, mu_1, f_type)
    suffix = '_C_{0}_n0_{1}_n1_{2}'.format(C, n_0, n_1)
    filename = directory + prefix + suffix
    f = open(filename, 'w')
    output_str = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}'.format(beta, lam,
        mu_0, mu_1, f_type, C_opt, n0_opt, n1_opt, opt_loss, run_time)
    f.write(output_str)
    f.close()


if __name__=='__main__':
    if len(sys.argv) <= 4:
        print "Usage: python evaluate.py n0 n1 C dir [-N N] [-beta BETA]" + \
                " [-gamma GAMMA] [-lam LAMBDA] [-mu0 MU0] [-mu1 MU1] [-f F]"
        sys.exit()

    # Get user input initial values of n0, n1 and C
    n_0 = float(sys.argv[1])
    n_1 = float(sys.argv[2])
    C = float(sys.argv[3])

    # User must specify which directory should the result be written to 
    directory = sys.argv[4]
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set the default values for optional parameters
    N = 200
    beta = 20.0
    lam = 10.0
    gamma = 0.95
    mu_0 = 1.0
    mu_1 = 1.0
    f_type = 1
    f = 1.0/np.arange(1, N+1, dtype=float)

    if '-N' in sys.argv:
        N = int(sys.argv[sys.argv.index('-N') + 1])
    if '-beta' in sys.argv:
        beta = float(sys.argv[sys.argv.index('-beta') + 1])
    if '-lam' in sys.argv:
        lam = float(sys.argv[sys.argv.index('-lam') + 1])
    if '-gamma' in sys.argv:
        gamma = float(sys.argv[sys.argv.index('-gamma') + 1])
    if '-mu0' in sys.argv:
        mu_0 = float(sys.argv[sys.argv.index('-mu0') + 1])
    if '-mu1' in sys.argv:
        mu_1 = float(sys.argv[sys.argv.index('-mu1') + 1])
    if '-f' in sys.argv:
        f_type = int(sys.argv[sys.argv.index('-f') + 1])
        if f_type == 2:
            f = 1.0/np.arange(1, N+1, dtype=float)**1.5
        if f_type == 3:
            f = 1.0/np.arange(1, N+1, dtype=float)**0.5

    do_run(N, beta, lam, gamma, mu_0, mu_1, f, f_type, C,
        n_0, n_1, directory)
