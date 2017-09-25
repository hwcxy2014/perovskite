import copy
import warnings
from collections import defaultdict
import logging
import numpy as np
import numpy.random as npr
import scipy.stats    as sps
import scipy.linalg   as spla
import numpy.linalg   as npla
import scipy.optimize as spo
import sys
from pes.model import global_optimization_of_GP

sys.path.append("../")


from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess

from unittests.test_util import get_random_gp_data
from covariance import ProductKernel

__author__ = 'jialeiwang'

try:
    import nlopt
except:
    nlopt_imported = False
else:
    nlopt_imported = True
# see http://ab-initio.mit.edu/wiki/index.php/NLopt_Python_Reference


NUM_RANDOM_FEATURES = 1000
GRID_SIZE = 1000 # used in a hacky way when chooser passes in the grid

"""
FOR GP MODELS ONLY
"""

# get samples of the solution to the problem
def sample_solution(grid, num_dims, objective_gp, constraint_gps=[]):
    assert num_dims == grid.shape[1]

    # 1. The procedure is: sample f and all the constraints on the grid "cand" (or use a smaller grid???)
    # 2. Look for the best point on the grid. if none exists, goto 1
    # 3. Do an optimization given this best point as the initializer

    MAX_ATTEMPTS = 10
    num_attempts = 0

    while num_attempts < MAX_ATTEMPTS:

        gp_samples = dict()
        gp_samples['objective'] = sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES)
        gp_samples['constraints'] = [sample_gp_with_random_features(constraint_gp, \
                                                                    NUM_RANDOM_FEATURES) for constraint_gp in constraint_gps]

        x_star_sample = global_optimization_of_GP_approximation(gp_samples, num_dims, grid)

        if x_star_sample is not None: # success
            logging.debug('successfully sampled x* in %d attempt(s)' % (num_attempts+1))

            return x_star_sample

        num_attempts += 1

    logging.info('Failed to sample x*')

    return None

# Compute log of the normal CDF of x in a robust way
# Based on the fact that log(cdf(x)) = log(1-cdf(-x))
# and log(1-z) ~ -z when z is small, so  this is approximately
# -cdf(-x), which is just the same as -sf(x) in scipy
def logcdf_robust(x):

    if isinstance(x, np.ndarray):
        ret = sps.norm.logcdf(x)
        ret[x > 5] = -sps.norm.sf(x[x > 5])
    elif x > 5:
        ret = -sps.norm.sf(x)
    else:
        ret = sps.norm.logcdf(x)

    return ret

# Compute log(exp(a)+exp(b)) in a robust way.
def logSumExp_scalar(a, b):

    if a > b:
        # compute log(exp(a)+exp(b))
        # this is just the log-sum-exp trick but with only 2 terms in the sum
        # we chooser to factor out the largest one
        # log(exp(a)+exp(b)) = log( exp(a) [1 + exp(b-a) ] )
        # = a + log(1 + exp(b-a))
        return a + log_1_plus_exp_x(b-a)
    else:
        return b + log_1_plus_exp_x(a-b)

def logSumExp(a,b):
    if (not isinstance(a, np.ndarray) or a.size==1) and (not isinstance(b, np.ndarray) or b.size==1):
        return logSumExp_scalar(a,b)

    result = np.zeros(a.shape)
    result[a>b] =  a[a>b]  + log_1_plus_exp_x(b[a>b] -a[a>b])
    result[a<=b] = b[a<=b] + log_1_plus_exp_x(a[a<=b]-b[a<=b])
    return result

# Compute log(1+exp(x)) in a robust way
def log_1_plus_exp_x_scalar(x):
    if x < np.log(1e-6):
        # if exp(x) is very small, i.e. less than 1e-6, then we can apply the taylor expansion:
        # log(1+x) approx equals x when x is small
        return np.exp(x)
    elif x > np.log(100):
        # if exp(x) is very large, i.e. greater than 100, then we say the 1 is negligible comared to it
        # so we just return log(exp(x))=x
        return x
    else:
        return np.log(1.0+np.exp(x))

def log_1_plus_exp_x(x):
    if not isinstance(x, np.ndarray) or x.size==1:
        return log_1_plus_exp_x_scalar(x)

    result = np.log(1.0+np.exp(x)) # case 3
    result[x < np.log(1e-6)] = np.exp(x[x < np.log(1e-6)])
    result[x > np.log(100) ] = x [x > np.log(100) ]
    return result

# Compute log(1-exp(x)) in a robust way, when exp(x) is between 0 and 1
# well, exp(x) is always bigger than 0
# but it cannot be above 1 because then we have log of a negative number
def log_1_minus_exp_x_scalar(x):
    if x < np.log(1e-6):
        # if exp(x) is very small, i.e. less than 1e-6, then we can apply the taylor expansion:
        # log(1-x) approx equals -x when x is small
        return -np.exp(x)
    elif x > -1e-6:
        # if x > -1e-6, i.e. exp(x) > exp(-1e-6), then we do the Taylor expansion of exp(x)=1+x+...
        # then the argument of the log, 1- exp(x), becomes, approximately, 1-(1+x) = -x
        # so we are left with log(-x)
        return np.log(-x)
    else:
        return np.log(1.0-np.exp(x))

def log_1_minus_exp_x(x):
    if not isinstance(x, np.ndarray) or x.size==1:
        return log_1_minus_exp_x_scalar(x)

    assert np.all(x <= 0)

    case1 = x < np.log(1e-6) # -13.8
    case2 = x > -1e-6
    case3 = np.logical_and(x >= np.log(1e-6), x <= -1e-6)
    assert np.all(case1+case2+case3 == 1)

    result = np.zeros(x.shape)
    result[case1] = -np.exp(x[case1])
    with np.errstate(divide='ignore'): # if x is exactly 0, give -inf without complaining
        result[case2] = np.log(-x[case2])
    result[case3] = np.log(1.0-np.exp(x[case3]))

    return result

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

def matrixInverse(M):
    return chol2inv(spla.cholesky(M, lower=False))

def ep(obj_model, con_models, x_star, minimize=True):
    # We construct the Vpred matrices and the mPred vectors
    n = obj_model.observed_values.size
    obj = 'objective'
    con = con_models.keys()
    all_tasks = con_models.copy()
    all_tasks[obj] = obj_model

    """ X contains X_star """
    X = np.append(obj_model.observed_inputs, x_star, axis=0)

    mPred         = dict()
    Vpred         = dict()
    cholVpred     = dict()
    VpredInv      = dict()
    cholKstarstar = dict()

    for t in all_tasks:
        mPred[t], Vpred[t] = all_tasks[t].predict(X, full_cov=True)
        cholVpred[t]       = spla.cholesky(Vpred[t])
        VpredInv[t]        = chol2inv(cholVpred[t])
        # Perform a redundant computation of this thing because predict() doesn't return it...
        cholKstarstar[t]   = spla.cholesky(all_tasks[t].noiseless_kernel.cov(X))

    jitter = dict()
    jitter[obj] = obj_model.jitter_value()
    for c in con:
        jitter[c] = con_models[c].jitter_value()

    # We create the posterior approximation
    a = {
        'obj'      : obj,
        'constraints': con,
        'Ahfhat'   : np.zeros((n, 2, 2)), # intiialize approximate factors to 0
        'bhfhat'   : np.zeros((n, 2)),
        'ahchat'   : defaultdict(lambda: np.zeros(n)),
        'bhchat'   : defaultdict(lambda: np.zeros(n)),
        'agchat'   : defaultdict(lambda: np.zeros(1)),
        'bgchat'   : defaultdict(lambda: np.zeros(1)),
        'm'        : defaultdict(lambda: np.zeros(n+1)),  # marginals
        'V'        : defaultdict(lambda: np.zeros((n+1, n+1))),
        'cholV'    : dict(),
        'mc'       : dict(),
        'Vc'       : dict(),
        'cholVc'   : dict(),
        'n'        : n,
        'mPred'    : mPred,
        'Vpred'    : Vpred,
        'VpredInv' : VpredInv,
        'cholKstarstar'  : cholKstarstar,
        'cholKstarstarc' : dict(),
        'jitter'   : jitter
    }

    # We update the marginals
    a = updateMarginals(a)

    # We start the main loop of EP
    convergence = False
    damping     = 1.0
    iteration   = 1
    while not convergence and n>0:

        aOld = copy.deepcopy(a)

        # We update the factors

        while True:

            try:
                aNew = copy.deepcopy(a)

                # We update the factors Ahfhat, bhfhat, ahchat, bhchat, agchat, bgchat
                aNew = updateFactors(aNew, damping, minimize=minimize)

                # We update the marginals V and m
                aNew = updateMarginals(aNew)

                # We verify that we can update the factors with an update of size 0
                updateFactors(aNew, 0, minimize=minimize)

                # This is also a testing step
                checkConstraintsPSD(con_models, aNew, X)

            except npla.linalg.LinAlgError as e:

                a = aOld
                damping *= 0.5

                if damping < 1e-5:
                    aNew = aOld
                    break
            else:
                break # things worked, you are done

        # We check for convergence
        a = aNew

        change = 0.0
        for t in all_tasks:
            change = max(change, np.max(np.abs(a['m'][t] - aOld['m'][t])))
            change = max(change, np.max(np.abs(a['V'][t] - aOld['V'][t])))
        # print 'change=%f' % change

        if change < 1e-4 and iteration > 2:
            convergence = True

        damping   *= 0.99
        iteration += 1

    # We update the means and covariance matrices for the constraint functions
    for c in con_models:
        X_all                  = np.append(X, con_models[c].observed_inputs, axis=0)
        noise                  = con_models[c].noise_value()
        Kstarstar              = con_models[c].noiseless_kernel.cov(X_all)
        a['cholKstarstarc'][c] = spla.cholesky(Kstarstar)
        mTilde                 = np.concatenate((a['bhchat'][c], np.array(a['bgchat'][c]), con_models[c].observed_values / noise))
        vTilde                 = np.concatenate((a['ahchat'][c], np.array(a['agchat'][c]), np.tile(1.0 / noise, con_models[c].observed_values.size)))
        Vc_inv                 = chol2inv(a['cholKstarstarc'][c]) + np.diag(vTilde)
        if np.any(npla.eigvalsh(Vc_inv) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")
        # Vc_inv += np.eye(Vc_inv.shape[0])*jitter[c] # added by mike
        chol_Vc_inv            = spla.cholesky(Vc_inv)
        Vc                     = chol2inv(chol_Vc_inv) # a['Vc'][c]
        a['cholVc'][c]         = spla.cholesky(Vc)
        # a['mc'][c]             = np.dot(Vc, mTilde)
        a['mc'][c]             = spla.cho_solve((chol_Vc_inv, False), mTilde)

        # We compute the cholesky factorization of the posterior covariance functions
        a['cholV'][c] = spla.cholesky(a['V'][c])
    a['cholV'][obj] = spla.cholesky(a['V'][obj])

    return a
    # fields: cholKstarstar, cholKstarstarc, V, cholV, Vc, cholVc, m, mc
    # (Vc not actually used, but include it for consistency)


# This checks that things are PSD. We want this to trigger failure in the EP loop
# so that damping can be reduced if needed
def checkConstraintsPSD(con_models, aNew, X):
    for c in con_models:
        X_all = np.append(X, con_models[c].observed_inputs, axis=0)
        noise   = con_models[c].noise_value()
        Kstarstar  = con_models[c].noiseless_kernel.cov(X_all)
        aNew['cholKstarstarc'][c] = spla.cholesky(Kstarstar)
        vTilde = np.concatenate((aNew['ahchat'][c], np.array(aNew['agchat'][c]),
                                 np.tile(1.0 / noise, con_models[c].observed_values.size)))
        Vc_inv = chol2inv(aNew['cholKstarstarc'][c]) + np.diag(vTilde)
        if np.any(npla.eigvalsh(Vc_inv) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")

# Updated a['V'] and a['m']
def updateMarginals(a):

    n = a['n']
    obj = a['obj']
    constraints = a['constraints']
    all_tasks = [obj] + constraints

    # for the objective
    vTilde = np.zeros((n+1,n+1))
    vTilde[np.eye(n+1).astype('bool')] = np.append(a['Ahfhat'][:, 0, 0], np.sum(a['Ahfhat'][: , 1, 1]))
    vTilde[:n, -1] = a['Ahfhat'][:, 0, 1]
    vTilde[-1, :n] = a['Ahfhat'][:, 0, 1]
    # if np.any(npla.eigvalsh(a['VpredInv'][obj] + vTilde) < 1e-6):
    #     raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")

    a['V'][obj] = matrixInverse(a['VpredInv'][obj] + vTilde)
    mTilde = np.append(a['bhfhat'][:, 0], np.sum(a['bhfhat'][:, 1]))
    a['m'][obj] = np.dot(a['V'][obj], np.dot(a['VpredInv'][obj], a['mPred'][obj]) + mTilde)

    # for the constraints
    for c in constraints:
        vTilde = np.diag(np.append(a['ahchat'][c], a['agchat'][c]))
        if np.any(npla.eigvalsh(a['VpredInv'][c] + vTilde) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")

        a['V'][c] = matrixInverse(a['VpredInv'][c] + vTilde)
        mTilde = np.append(a['bhchat'][c], np.sum(a['bgchat'][c]))
        a['m'][c] = np.dot(a['V'][c], np.dot(a['VpredInv'][c], a['mPred'][c]) + mTilde)


    # Before returning, we verify that the variances of the cavities are positive
    for i in xrange(n):
        # We obtain the cavities
        Vfinv = matrixInverse(a['V'][obj][np.ix_([i,n],[i,n])])
        # if np.any(npla.eigvalsh(Vfinv - a['Ahfhat'][i,:,:]) < 1e-6):
        #     raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")
        for c in constraints:
            if ((1.0 / a['V'][c][i, i] - a['ahchat'][c][i]) < 1e-6):
                raise npla.linalg.LinAlgError("Negative variance in cavity!")
    for c in constraints:
        if np.any(1.0 / a['V'][c][-1, -1] - a['agchat'][c] < 1e-6):
            raise npla.linalg.LinAlgError("Negative variance in cavity!")

    return a


def updateFactors(a, damping, minimize=True):

    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    # We update the h factors
    obj = a['obj']
    constraints = a['constraints']
    all_tasks = [obj] + constraints

    k = len(constraints)
    n = a['n']

    for i in xrange(n):

        # We obtain the cavities

        cholVf = spla.cholesky(a['V'][obj][np.ix_([i,n],[i,n])])
        Vfinv = chol2inv(cholVf)
        # Vfinv = matrixInverse(a['V'][obj][np.ix_([i,n],[i,n])])

        VfOldinv = Vfinv - a['Ahfhat'][i,:,:]
        cholVfOldinv = spla.cholesky(VfOldinv)
        VfOld = chol2inv(cholVfOldinv)

        # VfOld = matrixInverse(Vfinv - a['Ahfhat'][i,:,:])
        # mfOld = np.dot(VfOld, np.dot(Vfinv, a['m'][obj][[i, n]]) - a['bhfhat'][i,:])
        mfOld = spla.cho_solve((cholVfOldinv,False), spla.cho_solve((cholVf,False), a['m'][obj][[i, n]]) - a['bhfhat'][i,:])

        vcOld = np.zeros(k)
        mcOld = np.zeros(k)
        for j,c in enumerate(constraints):
            vcOld[j] = 1.0 / (1.0 / a['V'][c][i,i] - a['ahchat'][c][i])
            mcOld[j] = vcOld[j] * (a['m'][c][i] / a['V'][c][i,i] - a['bhchat'][c][i])

        # We compute the updates
        alphac     = mcOld / np.sqrt(vcOld)
        s           = VfOld[0, 0] - 2.0 * VfOld[1, 0] + VfOld[1, 1]
        alpha       = (mfOld[1] - mfOld[0]) / np.sqrt(s) * sgn
        logProdPhis = np.sum(logcdf_robust(alphac))
        logZ        = logSumExp(logProdPhis + logcdf_robust(alpha),  log_1_minus_exp_x(logProdPhis))
        ratio       = np.exp(sps.norm.logpdf(alphac) - logZ - logcdf_robust(alphac)) * (np.exp(logZ) - 1.0)
        # dlogZdmcOld = ratio / np.sqrt(vcOld)
        d2logZdmcOld2 = -ratio * (alphac + ratio) / vcOld
        ahchatNew = -1.0 / (1.0 / d2logZdmcOld2 + vcOld)
        # bhchatNew = (mcOld + np.sqrt(vcOld) / (alphac + ratio)) * ahchatNew
        bhchatNew = -(mcOld / (1.0 / (-ratio * (alphac + ratio) / vcOld) + vcOld) + np.sqrt(vcOld) / (-vcOld / ratio + (alphac + ratio) * vcOld))
        # above: the bottom way of computing bhchatNew is more stable when (alphac + ratio) = 0

        ratio = np.exp(logProdPhis + sps.norm.logpdf(alpha) - logZ)
        dlogZdmfOld = ratio / np.sqrt(s) * np.array([-1.0, 1.0]) * sgn
        dlogZdVfOld = -0.5 * ratio * alpha / s * np.array([[1.0,-1.0],[-1.0,1.0]])
        mfNew = mfOld + spla.cho_solve((cholVfOldinv,False), dlogZdmfOld)

        VfNew = VfOld - np.dot(spla.cho_solve((cholVfOldinv,False), np.dot(dlogZdmfOld[:,None], dlogZdmfOld[None]) - 2.0 * dlogZdVfOld), VfOld)

        # EXTRA_JITTER = np.eye(VfNew.shape[0])*a['jitter'][obj]
        # VfNew += EXTRA_JITTER

        # this is where the linalg error gets thrown, causing damping to be reduced
        cholVfNew = spla.cholesky(VfNew)
        vfNewInv = chol2inv(cholVfNew)
        # vfNewInv = matrixInverse(VfNew)

        AhfHatNew = vfNewInv - (Vfinv - a['Ahfhat'][i,:,:])
        # bhfHatNew = np.dot(vfNewInv, mfNew) - (np.dot(Vfinv, a['m'][obj][[i, n]]) - a['bhfhat'][i,:])
        bhfHatNew = spla.cho_solve((cholVfNew,False), mfNew) - (spla.cho_solve((cholVf,False), a['m'][obj][[i, n]]) - a['bhfhat'][i,:])

        # We do damping
        a['Ahfhat'][i,:,:] = damping * AhfHatNew + (1.0 - damping) * a['Ahfhat'][i,:,:]
        a['bhfhat'][i,:]   = damping * bhfHatNew + (1.0 - damping) * a['bhfhat'][i,:]

        for j,c in enumerate(constraints):
            a['ahchat'][c][i] = damping * ahchatNew[j] + (1.0 - damping) * a['ahchat'][c][i]
            a['bhchat'][c][i] = damping * bhchatNew[j] + (1.0 - damping) * a['bhchat'][c][i]

    # We update the g factors
    # We obtain the cavities
    for j,c in enumerate(constraints):
        vcOld[j] = 1.0 / (1.0 / a['V'][c][n, n] - a['agchat'][c])
        mcOld[j] = vcOld[j] * (a['m'][c][n] / a['V'][c][n, n] - a['bgchat'][c])

    # We compute the updates
    alpha = mcOld / np.sqrt(vcOld)
    ratio = np.exp(sps.norm.logpdf(alpha) - logcdf_robust(alpha))
    # dlogZdmcOld = ratio / np.sqrt(vcOld)
    d2logZdmcOld2 = -ratio / vcOld * (alpha + ratio)
    agchatNew = -1 / (1.0 / d2logZdmcOld2 + vcOld)
    # bgchatNew = (mcOld + np.sqrt(vcOld) / (alpha + ratio)) * agchatNew
    bgchatNew = -(mcOld / (1.0 / (-ratio * (alpha + ratio) / vcOld) + vcOld) + np.sqrt(vcOld) / (-vcOld / ratio + (alpha + ratio) * vcOld))
    # above: the bottom way of computing bhchatNew is more stable when (alphac + ratio) = 0

    # We do damping
    for j,c in enumerate(constraints):
        a['agchat'][c] = damping * agchatNew[j] + (1.0 - damping) *  a['agchat'][c]
        a['bgchat'][c] = damping * bgchatNew[j] + (1.0 - damping) *  a['bgchat'][c]

    # We are done
    return a

def gp_prediction_given_chol_K(X, Xtest, chol_star, cholV, m, model, jitter):
    # computes the predictive distributions. but the chol of the kernel matrix and the
    # chol of the test matrix are already provided.

    Kstar = model.noiseless_kernel.cross_cov(X, Xtest)
    mf = np.dot(Kstar.T, spla.cho_solve((chol_star, False), m))
    aux = spla.cho_solve((chol_star, False), Kstar)
    # vf = model.params['amp2'].value * (1.0 + jitter) - \
    #     np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
    #     np.sum(np.dot(cholV, aux)**2, axis=0)
    # vf = model.params['amp2'] - \
    #      np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
    #      np.sum(np.dot(cholV, aux)**2, axis=0) + \
    #      jitter
    vf = model.noiseless_kernel.self_var(Xtest) - \
         np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
         np.sum(np.dot(cholV, aux)**2, axis=0) + \
         jitter
    # print "vf1: {0}\n vf2: {1}\n vf3: {2}".format(model.noiseless_kernel.covariance(Xtest[0,:], Xtest[0,:]),
    #                                               np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0),
    #                                               np.sum(np.dot(cholV, aux)**2, axis=0))

    if np.any(vf < 0.0):
        print np.less(vf, 0.0).sum()
        # vf = np.maximum(vf, 0.0001)
        raise Exception("Encountered negative variance: %f" % np.min(vf))

    return Kstar, mf, vf

# Method that approximates the predictive distribution at a particular location.
def predictEP(obj_model, con_models, a, x_star, Xtest, minimize=True):

    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    obj = a['obj']
    constraints = con_models.keys()
    all_tasks = [obj] + constraints

    X = np.append(obj_model.observed_inputs, x_star, axis=0)

    Kstar, mf, vf = gp_prediction_given_chol_K(X, Xtest,
                                               a['cholKstarstar'][obj], a['cholV'][obj], a['m'][obj], obj_model, a['jitter'][obj])

    # We compute the covariance between the test point and the optimum
    KstarXstar = obj_model.noiseless_kernel.cross_cov(X, x_star)
    aux1       = spla.solve_triangular(a['cholKstarstar'][obj].T, Kstar, lower=True)
    aux2       = spla.solve_triangular(a['cholKstarstar'][obj].T, KstarXstar, lower=True)
    aux11      = np.dot(a['cholV'][obj], spla.solve_triangular(a['cholKstarstar'][obj], aux1, lower=False))
    aux12      = np.dot(a['cholV'][obj], spla.solve_triangular(a['cholKstarstar'][obj], aux2, lower=False))
    cov        = Kstar[-1,:] - np.sum(aux2 * aux1, axis=0) + np.sum(aux12 * aux11, axis=0)
    # Above: in computing "cov" we use broadcasting, so we deviate a bit from the R code

    assert Kstar.shape[0] == X.shape[0]

    # We obtain the posterior mean and variance at the optimum
    mOpt = a['m'][obj][-1]
    vOpt = a['V'][obj][-1, -1]

    # We compute the predictive distribution for the constraints
    mc = np.zeros((Xtest.shape[0], len(constraints)))
    vc = np.zeros((Xtest.shape[0], len(constraints)))
    for i,c in enumerate(constraints):
        Xc = np.append(X, con_models[c].observed_inputs, axis=0)

        Kstar, mc[:,i], vc[:,i] = gp_prediction_given_chol_K(Xc, Xtest,
                                                             a['cholKstarstarc'][c], a['cholVc'][c], a['mc'][c], con_models[c], a['jitter'][c])

    # scale things for stability
    scale = 1.0 - 1e-4
    while np.any(vf - 2.0 * scale * cov + vOpt < 1e-10):
        scale = scale**2
    cov = scale * cov

    # We update the predictive distribution for f to take into account that it has to be smaller than the optimum
    s       = vf - 2.0 * cov + vOpt
    alpha   = (mOpt - mf) / np.sqrt(s) * sgn
    alphac = mc / np.sqrt(vc)
    logProdPhis = np.sum(logcdf_robust(alphac), axis=1) # sum over constraints
    logZ    = logSumExp(logProdPhis + logcdf_robust(alpha), log_1_minus_exp_x(logProdPhis))
    ratio   = np.exp(logProdPhis + sps.norm.logpdf(alpha) - logZ)
    # mfNew   = mf + (cov - vf) *  ratio / np.sqrt(s) * sgn
    # above: not used in the acquisition function
    vfNew   = vf - ratio / s * (ratio + alpha) * (vf - cov)**2

    logZ = logZ[:,None] # normalization constant for the product of gaussian and factor that ensures x star is the solution # supplement line 8

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            ratio = np.exp(sps.norm.logpdf(alphac) - logZ - logcdf_robust(alphac)) * (np.exp(logZ) - 1.0)
        except RuntimeWarning:
            print ("Capped ratio!  "*50)
            ratio = np.exp(50) # cap ratio to avoid overflow issues

    # dlogZdmcOld = ratio / np.sqrt(vc)
    d2logZdmcOld2 = -ratio * (alphac + ratio) / vc # for numerical stability
    ahchatNew = -1.0 / (1.0 / d2logZdmcOld2 + vc)
    # bhchatNew = (mc + np.sqrt(vc) / (alphac + ratio)) * ahchatNew
    bhchatNew = -(mc / (1.0 / (-ratio * (alphac + ratio) / vc) + vc) + np.sqrt(vc) / (-vc / ratio + (alphac + ratio) * vc))
    # above: the bottom way of computing bhchatNew is more stable when (alphac + ratio) = 0
    vcNew = 1.0 / (1.0 / vc + ahchatNew)
    # mcNew = vcNew *  (mc / vc + bhchatNew)
    # mcNew not actually used

    # make sure variances are not negative, by replacing with old values
    vfNew[vfNew < 0] = vf[vfNew < 0]
    vcNew[vcNew < 0] = vc[vcNew < 0]


    if np.any(vfNew <= 0):
        raise Exception("vfnew is negative: %g at index %d" % (np.min(vfNew), np.argmin(vfNew)))
    if np.any(vcNew <= 0):
        raise Exception("vcnew is negative: %g at index %d" % (np.min(vcNew), np.argmin(vcNew)))
    # if np.any(np.isnan(mfNew)):
    #     raise Exception("mfnew contains nan at index %s" % str(np.nonzero(np.isnan(mfNew))))
    # if np.any(np.isnan(mcNew)):
    #     raise Exception("mcNew contains nan at index %s" % str(np.nonzero(np.isnan(mcNew))))
    if np.any(np.isnan(vcNew)):
        raise Exception("vcnew constrains nan")
    if np.any(np.isnan(vfNew)):
        raise Exception("vfnew constrains nan")

    return {'mf':None, 'vf':vfNew, 'mc':None, 'vc':vcNew}
    # don't bother computing mf and mc since they are not used in the acquisition function
    # m = mean, v = var, f = objective, c = constraint


"""
See Miguel's paper (http://arxiv.org/pdf/1406.2541v1.pdf) section 2.1 and Appendix A

Returns a function the samples from the approximation...

if testing=True, it does not return the result but instead the random cosine for testing only

We express the kernel as an expectation. But then we approximate the expectation with a weighted sum
theta are the coefficients for this weighted sum. that is why we take the dot product of theta at the end
we also need to scale at the end so that it's an average of the random features.

if use_woodbury_if_faster is False, it never uses the woodbury version
"""
def sample_gp_with_random_features(gp, nFeatures, testing=False, use_woodbury_if_faster=True):

    d = gp.num_dims
    N_data = gp.observed_truth_values.size

    nu2 = gp.noise_value()

    sigma2 = gp.params['amp2'].value  # the kernel amplitude

    # We draw the random features
    if gp.options['kernel'] == "SquaredExp":
        W = npr.randn(nFeatures, d) / gp.params['ls'].value
    elif gp.options['kernel'] == "Matern52":
        m = 5.0/2.0
        W = npr.randn(nFeatures, d) / gp.params['ls'].value / np.sqrt(npr.gamma(shape=m, scale=1.0/m, size=(nFeatures,1)))
    else:
        raise Exception('This random feature sampling is for the squared exp or Matern5/2 kernels and you are using the %s' % gp.options['kernel'])
    b = npr.uniform(low=0, high=2*np.pi, size=nFeatures)[:,None]

    # Just for testing the  random features in W and b... doesn't test the weights theta

    if testing:
        return lambda x: np.sqrt(2 * sigma2 / nFeatures) * np.cos(np.dot(W, gp.noiseless_kernel.transformer.forward_pass(x).T) + b)

    randomness = npr.randn(nFeatures)

    # W has size nFeatures by d
    # tDesignMatrix has size Nfeatures by Ndata
    # woodbury has size Ndata by Ndata
    # z is a vector of length nFeatures

    if gp.has_data:
        tDesignMatrix = np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, gp.observed_truth_inputs.T) + b)

        if use_woodbury_if_faster and N_data < nFeatures:
            # you can do things in cost N^2d instead of d^3 by doing this woodbury thing

            # We obtain the posterior on the coefficients
            woodbury = np.dot(tDesignMatrix.T, tDesignMatrix) + nu2*np.eye(N_data)
            chol_woodbury = spla.cholesky(woodbury)
            # inverseWoodbury = chol2inv(chol_woodbury)
            z = np.dot(tDesignMatrix, gp.observed_truth_values / nu2)
            # m = z - np.dot(tDesignMatrix, np.dot(inverseWoodbury, np.dot(tDesignMatrix.T, z)))
            m = z - np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), np.dot(tDesignMatrix.T, z)))
            # (above) alternative to original but with cho_solve

            # z = np.dot(tDesignMatrix, gp.observed_truth_values / nu2)
            # m = np.dot(np.eye(nFeatures) - \
            # np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), tDesignMatrix.T)), z)

            # woodbury has size N_data by N_data
            D, U = npla.eigh(woodbury)
            # sort the eigenvalues (not sure if this matters)
            idx = D.argsort()[::-1] # in decreasing order instead of increasing
            D = D[idx]
            U = U[:,idx]
            R = 1.0 / (np.sqrt(D) * (np.sqrt(D) + np.sqrt(nu2)))
            # R = 1.0 / (D + np.sqrt(D*nu2))

            # We sample from the posterior of the coefficients
            theta = randomness - \
                    np.dot(tDesignMatrix, np.dot(U, (R * np.dot(U.T, np.dot(tDesignMatrix.T, randomness))))) + m

        else:
            # all you are doing here is sampling from the posterior of the linear model
            # that approximates the GP
            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) / nu2 + np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_truth_values / nu2))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma, lower=False)).T

            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_truth_values))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T

            chol_Sigma_inverse = spla.cholesky(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            Sigma = chol2inv(chol_Sigma_inverse)
            m = spla.cho_solve((chol_Sigma_inverse, False), np.dot(tDesignMatrix, gp.observed_truth_values))
            theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T


    else:
        # We sample from the prior -- same for Matern
        theta = npr.randn(nFeatures)

    def wrapper(x, gradient):
        # the argument "gradient" is
        # not the usual compute_grad that computes BOTH when true
        # here it only computes the objective when true

        if x.ndim == 1:
            x = x[None,:]

        # x = gp.noiseless_kernel.transformer.forward_pass(x)

        if not gradient:
            result = np.dot(theta.T, np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b))
            if result.size == 1:
                result = float(result) # if the answer is just a number, take it out of the numpy array wrapper
                # (failure to do so messed up NLopt and it only gives a cryptic error message)
            return result
        else:
            grad = np.dot(theta.T, -np.sqrt(2.0 * sigma2 / nFeatures) * np.sin(np.dot(W, x.T) + b) * W)
            # return gp.noiseless_kernel.transformer.backward_pass(grad)
            return grad

    return wrapper

"""
Given some approximations to the GP sample, find its minimum
We do that by first evaluating it on a grid, taking the best, and using that to
initialize an optimization. If nothing on the grid satisfies the constraint, then
we return None

wrapper_functions should be a dict with keys 'objective' and optionally 'constraints'
"""
# find MINIMUM if minimize=True, else find a maximum
def global_optimization_of_GP_approximation(funs, num_dims, grid, minimize=True):

    assert num_dims == grid.shape[1]

    num_con = len(funs['constraints'])

    # print 'evaluating on grid'
    # First, evaluate on a grid and see what you get
    obj_evals = funs['objective'](grid, gradient=False)
    con_evals = np.ones(grid.shape[0]).astype('bool')
    for con_fun in funs['constraints']:
        con_evals = np.logical_and(con_evals, con_fun(grid, gradient=False)>=0)

    if not np.any(con_evals):
        return None

    if minimize:
        best_guess_index = np.argmin(obj_evals[con_evals])
        best_guess_value = np.min(obj_evals[con_evals])
    else:
        best_guess_index = np.argmax(obj_evals[con_evals])
        best_guess_value = np.max(obj_evals[con_evals])
    x_initial = grid[con_evals][best_guess_index]


    fun_counter = defaultdict(int)

    if nlopt_imported:

        opt = nlopt.opt(nlopt.LD_MMA, num_dims)
        opt.set_lower_bounds(0.0)#np.zeros(num_dims))
        opt.set_upper_bounds(1.0)#np.ones(num_dims))

        def f(x, put_gradient_here):
            fun_counter['obj'] += 1
            if put_gradient_here.size > 0:
                # set grad to the gradient, here

                put_gradient_here[:] = funs['objective'](x, gradient=True)
            # return the value of f(x)
            return float(funs['objective'](x, gradient=False))

        if minimize:
            opt.set_min_objective(f)
        else:
            opt.set_max_objective(f)

        # constraints in NLopt are <= 0 constraint. So we want to take the negative of them...
        def g(put_result_here, x, put_gradient_here):
            fun_counter['con'] += 1
            for i,constraint_wrapper in enumerate(funs['constraints']):
                if put_gradient_here.size > 0:
                    put_gradient_here[i,:] = -constraint_wrapper(x, gradient=True)
                put_result_here[i] = -constraint_wrapper(x, gradient=False)

        # tol = [1e-8]*len(funs['constraints'])
        tol = np.zeros(len(funs['constraints']))
        opt.add_inequality_mconstraint(g, tol)
        opt.set_maxeval(10000)

        opt.set_xtol_abs(1e-6)

        # print 'Optimizing in %d dimensions with %s.' % (opt.get_dimension(), opt.get_algorithm_name())
        opt_x = opt.optimize(x_initial.copy())

        returncode = opt.last_optimize_result()
        y_opt = f(opt_x, np.array([]))

        # logging.debug('returncode=%d'%returncode)
        # logging.debug('Evaluated the objective %d times and constraints %d times' % (fun_counter['obj'], fun_counter['con']))
        nlopt_constraints_results = np.zeros(num_con)
        g(nlopt_constraints_results, opt_x, np.zeros(0))
        constraint_tol = 1e-8 # my tolerance, not the one I give to NLOPT
        # all_constraint_satisfied = np.all(nlopt_constraints_results <= constraint_tol)
        if (returncode > 0 or returncode==-4) and y_opt < best_guess_value:# and all_constraint_satisfied:
            return opt_x[None]
            # elif not all_constraint_satisfied:
            # logging.debug('NLOPT failed when optimizing x*: violated constraints: %g' % np.max(nlopt_constraints_results))
            # return x_initial[None]
        elif not (returncode > 0 or returncode==-4):
            logging.debug('NLOPT failed when optimizing x*: bad returncode')
            return x_initial[None]
        else:
            logging.debug('NLOPT failed when optimizing x*: objective got worse from %f to %f' %(best_guess_value, y_opt))
            return x_initial[None]


    else:
        assert minimize # todo - can fix later

        f       = lambda x: float(funs['objective'](x, gradient=False))
        f_prime = lambda x: funs['objective'](x, gradient=True).flatten()


        # with SLSQP in scipy, the constraints are written as c(x) >= 0
        def g(x):
            g_func = np.zeros(num_con)
            for i,constraint_wrapper in enumerate(funs['constraints']):
                g_func[i] = constraint_wrapper(x, gradient=False)
            return g_func

        def g_prime(x):
            g_grad_func = np.zeros((num_con, num_dims))
            for i,constraint_wrapper in enumerate(funs['constraints']):
                g_grad_func[i,:] = constraint_wrapper(x, gradient=True)
            return g_grad_func

        bounds = [(0.0,1.0)]*num_dims

        opt_x = spo.fmin_slsqp(f, x_initial.copy(),
                               bounds=bounds, disp=0, fprime=f_prime, f_ieqcons=g, fprime_ieqcons=g_prime)
        # make sure bounds are respected
        opt_x[opt_x > 1.0] = 1.0
        opt_x[opt_x < 0.0] = 0.0

        if f(opt_x) < best_guess_value and np.all(g(opt_x)>=0):
            return opt_x[None]
        else:
            logging.debug('SLSQP failed when optimizing x*')
            return x_initial[None]

            # return opt_x[None]


class PES(object):

    def __init__(self, num_dims, verbose=True, input_space=None, grid=None, opt = None):
        # we want to cache these. we use a dict indexed by the state integer
        self.cached_EP_solutions = dict()
        self.cached_x_star = dict()

        self.has_gradients = False

        self.num_dims = num_dims

        # if grid is None:
        #     self.xstar_grid = sobol_grid.generate(num_dims, grid_size=GRID_SIZE)
        # else:
        #     # This is a total hack. We just do this to make sure we include
        #     # The observed points and spray points that are added on.
        #     # If you had more than GRID_SIZE observations this would be totally messed up...
        #     logging.debug('Note: grid passed in has size %d, truncating to size %d for PESC.' % (grid.shape[0], GRID_SIZE))
        #     self.xstar_grid = grid[-GRID_SIZE:]

        self.input_space = input_space

    # obj_models is a GP
    # con_models is a dict of named constraints and their GPs
    def acquisition(self, obj_model_dict, con_models_dict, cand, current_best,
                    compute_grad, DEBUG_xstar=None, minimize=True, tasks=None):
        obj_model = obj_model_dict.values()[0]

        models = [obj_model] + list(con_models_dict.values())

        for model in models:
            # if model.pending is not None:
            #     raise NotImplementedError("PES not implemented for pending stuff? Not sure. Should just impute the mean...")

            if not model.options['caching']:
                logging.error("Warning: caching is off while using PES!")

        # make sure all models are at the same state
        assert len({model.state for model in models}) == 1, "Models are not all at the same state"
        assert not compute_grad

        N_cand = cand.shape[0]

        # If the epSolution is already saved, load it.
        # Otherwise, compute it with the ep() function and save it
        # print ''
        if obj_model.state in self.cached_EP_solutions:
            x_star     = self.cached_x_star[obj_model.state]
            epSolution = self.cached_EP_solutions[obj_model.state]
        else:
            #logging.debug('Sampling solution for hyper sample %d' % obj_model.state)
            if DEBUG_xstar is None:
                # x_star = sample_solution(self.xstar_grid, self.num_dims, obj_model, con_models_dict.values())
                x_star = global_optimization_of_GP(obj_model.gp_model, [(0,1)] * self.num_dims, 5)

                if x_star is None:
                    self.cached_x_star[obj_model.state] = None
                    self.cached_EP_solutions[obj_model.state] = None
                    return np.zeros(cand.shape[0])

                if self.input_space:
                    logging.debug('x* = %s' % self.input_space.from_unit(x_star))
                else:
                    logging.debug('x* = %s' % str(x_star))
            else:
                logging.debug('DEBUG MODE: using xstar value of %s' % str(DEBUG_xstar))
                x_star = DEBUG_xstar

            # logging.debug('Doing EP for hyper sample %d' % obj_model.state)
            with np.errstate(divide='ignore',over='ignore'):
                epSolution = ep(obj_model, con_models_dict, x_star, minimize=minimize)

            self.cached_x_star[obj_model.state] = x_star
            self.cached_EP_solutions[obj_model.state] = epSolution

        # if you failed to sample a solution, just return 0
        if x_star is None:
            return np.zeros(cand.shape[0])

        # use the EP solutions to compute the acquisition function
        acq_dict = evaluate_acquisition_function_given_EP_solution(obj_model_dict, con_models_dict, cand, epSolution, x_star, minimize=minimize)

        # by default, sum the PESC contribution for all tasks
        if tasks is None:
            tasks = acq_dict.keys()

        # Compute the total acquisition function for the tasks of interests
        total_acq = 0.0
        for task in tasks:
            total_acq += acq_dict[task]

        return total_acq


# Returns the PES(C) for each task given the EP solution and sampled x_star.
def evaluate_acquisition_function_given_EP_solution(obj_model_dict, con_models, cand, epSolution, x_star, minimize=True):
    if cand.ndim == 1:
        cand = cand[None]
    obj_name  = obj_model_dict.keys()[0]
    obj_model = obj_model_dict.values()[0]

    # unconstrainedVariances = np.zeros((N_cand, len(con_models)+1))
    # unconstrainedVariances[:,0] = obj_model.predict(cand)[1] + obj_model.noise_value()
    # for j, c in enumerate(con_models):
    #     unconstrainedVariances[:,j+1] = con_models[c].predict(cand)[1] + con_models[c].noise_value()
    unconstrainedVariances = dict()
    unconstrainedVariances[obj_name] = obj_model.predict(cand)[1] + obj_model.noise_value(cand[0,0])
    for c, con_model in con_models.iteritems():
        unconstrainedVariances[c] = con_model.predict(cand)[1] + con_model.noise_value()

    # We then evaluate the constrained variances
    with np.errstate(divide='ignore', over='ignore'):
        predictionEP = predictEP(obj_model, con_models, epSolution, x_star, cand, minimize=minimize)

    # constrainedVariances = np.zeros((N_cand, len(con_models)+1))
    # constrainedVariances[:,0] = predictionEP['vf'] + obj_model.noise_value()
    # for j, c in enumerate(con_models):
    #     constrainedVariances[:, j+1] = predictionEP['vc'][:,j] + con_models[c].noise_value()
    constrainedVariances = dict()
    constrainedVariances[obj_name] = predictionEP['vf'] + obj_model.noise_value(cand[0,0])
    for j, c in enumerate(con_models):
        constrainedVariances[c] = predictionEP['vc'][:,j] + con_models[c].noise_value()

    # We only care about the variances because the means do not affect the entropy
    acq = dict()
    for t in unconstrainedVariances:
        acq[t] = 0.5 * np.log(2 * np.pi * np.e * unconstrainedVariances[t]) - \
                 0.5 * np.log(2 * np.pi * np.e * constrainedVariances[t])

        # acq = np.sum(0.5 * np.log(2 * np.pi * np.e * unconstrainedVariances), axis=1) - \
        # np.sum(0.5 * np.log(2 * np.pi * np.e * constrainedVariances)  , axis=1)
    # assert not np.any(np.isnan(acq)), "Acquisition function contains NaN"

    for t in acq:
        if np.any(np.isnan(acq[t])):
            raise Exception("Acquisition function containts NaN for task %s" % t)

    return acq

