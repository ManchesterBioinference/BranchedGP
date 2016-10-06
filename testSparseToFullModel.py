# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from GPclust import OMGP
import GPy
import GPyOpt
# import pickle
import time
# Branching files
import VBHelperFunctions
import BranchingTree as bt
import branch_kernParamGPflow as bk
import assigngp_denseSparse
import assigngp_dense
import BayesianOptimiser


def InitParams(m):
    m.likelihood.variance = mo.variance.values[0]
    # set lengthscale to maximum
    m.kern.branchkernelparam.kern.lengthscales = np.max(
        np.array([mo.kern[0].lengthscale.values, mo.kern[1].lengthscale.values]))
    # set process variance to average
    m.kern.branchkernelparam.kern.variance = np.mean(
        np.array([mo.kern[0].variance.values, mo.kern[1].variance.values]))

########################################
#         Test parameters
########################################
fDebug = True  # Enable debugging output - tensorflow print ops
########################################
np.set_printoptions(precision=4)  # precision to print numpy array
seed = 43
np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)
# Data generation
N = 20
t = np.linspace(0, 1, N)
print(t)
trueB = np.ones((1, 1))*0.5
Y = np.zeros((N, 1))
idx = np.nonzero(t > 0.5)[0]
idxA = idx[::2]
idxB = idx[1::2]
print(idx)
print(idxA)
print(idxB)
Y[idxA, 0] = 2 * t[idxA]
Y[idxB, 0] = -2 * t[idxB]
# Create tree structures
tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
tree.add(None, 1, trueB)
(fm, _) = tree.GetFunctionBranchTensor()
XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)
print('XExpanded', XExpanded.shape)
print('indices', len(indices))
# Use OMGP for initialisation of branching model
komgp1 = GPy.kern.Matern32(1)
komgp2 = GPy.kern.Matern32(1)
mo = OMGP(t[:, None], Y, K=2, variance=0.01, kernels=[komgp1, komgp2],
          prior_Z='DP')  # use a truncated DP with K=2 UNDONE
mo.kern[0].lengthscale = 5.*np.ptp(t)  # initialise length scale to range of data
mo.kern[1].lengthscale = 5.*np.ptp(t)
mo.optimize(step_length=0.01, maxiter=5)
# Create model
Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=trueB.copy()) + GPflow.kernels.White(1)
Kbranch.branchkernelparam.kern.variance = 1
Kbranch.white.variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
Kbranch.white.variance.fixed = True  # jitter for numerics
print('Kbranch matrix', Kbranch.compute_K(XExpanded, XExpanded))
print('Branching K free parameters', Kbranch.branchkernelparam)
print('Branching K branching parameter', Kbranch.branchkernelparam.Bv.value)
# Initialise all model parameters using the OMGP model
# Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
# has one common set.
M = 5  # number of inducing pts
# ir = np.random.choice(XExpanded.shape[0], M)
ZExpanded = XExpanded  # [ir, :] Test on full data
mV = assigngp_denseSparse.AssignGPSparse(t, XExpanded, Y, Kbranch, indices, mo.phi,
                                         Kbranch.branchkernelparam.Bv.value, ZExpanded, fDebug=fDebug)
InitParams(mV)

mVFull = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices, mo.phi,
                                 Kbranch.branchkernelparam.Bv.value, fDebug=fDebug)

InitParams(mVFull)

lsparse = mV.compute_log_likelihood()
lfull = mVFull.compute_log_likelihood()
print('Sparse Log lik', lsparse, 'Full Log luk', lfull)
# assert np.allclose(lsparse, lfull), 'Log likelihoods not close'

# check models identical
assert np.all(mV.GetPhiExpanded() == mVFull.GetPhiExpanded())
assert mV.likelihood.variance.value == mVFull.likelihood.variance.value
assert mV.kern is mVFull.kern


# Test prediction
Xtest = np.array([[0.6, 2], [0.6, 3]])
mu_f, var_f = mVFull.predict_f(Xtest)

mu_s, var_s = mV.predict_f(Xtest)

print('Sparse model mu=', mu_s, ' variance=', var_s)
print('Full model mu=', mu_f, ' variance=', var_f)
