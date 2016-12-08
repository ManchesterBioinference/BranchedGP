# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense
from BranchedGP import BayesianOptimiser


def InitKernParams(ms):
    ms.kern.branchkernelparam.kern.variance = 2
    ms.kern.branchkernelparam.kern.lengthscales = 5
    ms.likelihood.variance = 0.01

# Could try
# varianceGaussian = 0.01
# kerlen = 5
########################################
#         Test parameters
########################################
fPlot = True  # do we do plots?
fDebug = True  # Enable debugging output - tensorflow print ops
########################################
np.set_printoptions(suppress=True,  precision=5)
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
# Create model
Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=trueB.copy()) + GPflow.kernels.White(1)
Kbranch.white.variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
Kbranch.white.variance.fixed = True  # jitter for numerics
# Create model
phiPrior = np.ones((N, 2))*0.5  # dont know anything
phiInitial = np.ones((N, 2))*0.5  # dont know anything
phiInitial[:, 0] = np.random.rand(N)
phiInitial[:, 1] = 1-phiInitial[:, 0]
m = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices,
                            Kbranch.branchkernelparam.Bv.value, phiPrior=phiPrior, phiInitial=phiInitial)
InitKernParams(m)
m.likelihood.variance.fixed = True
print('Model before initialisation\n', m, '\n===========================')
m.optimize(disp=0, maxiter=100)
m.likelihood.variance.fixed = False
m.optimize(disp=0, maxiter=100)
print('Model after initialisation\n', m, '\n===========================')
ttestl, mul, varl = VBHelperFunctions.predictBranchingModel(m)
PhiOptimised = m.GetPhi()
print('phiPrior', phiPrior)
print('PhiOptimised', PhiOptimised)

# reset model
m.UpdateBranchingPoint(Kbranch.branchkernelparam.Bv.value, phiInitial)  # reset initial phi
m.UpdatePhiPrior(phiPrior.copy())  # prior
InitKernParams(m)
ll_flatprior = m.compute_log_likelihood()
phiInfPrior = np.ones((N, 2))*0.5  # dont know anything
phiInfPrior[-1, :] = [0.99, 0.01]
# phiInfPrior[-2, :] = [0.01, 0.99]
m.UpdatePhiPrior(phiInfPrior.copy())  # prior
ll_betterprior = m.compute_log_likelihood()
assert ll_betterprior > ll_flatprior

if(fPlot):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.ion()
    plt.close('all')
    f, axes = plt.subplots(1, 3, figsize=(20, 10), sharex=True)
    ax = axes.flatten()
    _, PhiColor = VBHelperFunctions.plotBranchModel(Kbranch.branchkernelparam.Bv.value[0], m.t, m.Y.value,
                                                    ttestl, mul, varl, phiPrior,
                                                    ax=ax[0], fPlotPhi=True, fPlotVar=True, fColorBar=False)
    ax[0].set_title('b=%f, Objective %.3f Prior Phi' % (Kbranch.branchkernelparam.Bv.value, m.objectiveFun()), fontsize=30)
    _, PhiColor = VBHelperFunctions.plotBranchModel(Kbranch.branchkernelparam.Bv.value[0], m.t, m.Y.value,
                                                    ttestl, mul, varl, PhiOptimised,
                                                    ax=ax[1], fPlotPhi=True, fPlotVar=True, fColorBar=False)
    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat], orientation='horizontal')
    f.colorbar(PhiColor, cax=cax, **kw)
    f.show()

#     plt.figure()
#     plt.scatter(m.t, m.Y.value, c=PhiOptimised[:, 1], vmin=0., vmax=1, s=40)
#     plt.show()

# m.UpdatePhiPrior(pZ0)
# Plot results - this will call predict
# prediction
# Plot fit with 0.5 prior vs informative
# Plot fit vs prior setting
# test diff models same as update single model