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
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense
from BranchedGP import BayesianOptimiser


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
fPlot = True  # do we do plots?
fUsePriors = False  # Test priors on kernel hyperparameters
fModelSelectionGrid = True
fBO = False  # Bayesian optimisation
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
mo.optimize(step_length=0.01, maxiter=30)
# plotting OMGP fit
if(fPlot):
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    mo.plot()
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
mV = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices, mo.phi, Kbranch.branchkernelparam.Bv.value)
InitParams(mV)
# put prior to penalise short length scales
assert mV.compute_log_likelihood() == -mV.objectiveFun()
# Prior to penalize small length scales
if(fUsePriors):
    mV.kern.branchkernelparam.kern.lengthscales.prior = GPflow.priors.Gaussian(np.ptp(t), np.square(np.ptp(t) / 3.))
    assert mV.compute_log_likelihood() != -mV.objectiveFun()
#     mV.kern.branchkernelparam.kern.variance.prior = GPflow.priors.Gaussian(10, np.square(3.))
print('Initialised mv', mV)

# Plot results - this will call predict
if(fPlot):
    VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
    plt.title('Initialisation')
print('Initialisation')
print('OMGP Phi matrix', np.round(mo.phi, 2))
print('b=', mV.kern.branchkernelparam.Bv.value, 'Branch model Phi matrix', np.round(mV.GetPhi(), 2))
mV.optimize()
# Plot results - this will call predict
objT = mV.objectiveFun()
if(fPlot):
    VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
    plt.title('B=%g ll=%.2f' % (mV.kern.branchkernelparam.Bv.value, objT))
print('Fitted model')
print('OMGP Phi matrix', np.round(mo.phi, 2))
print('b=', mV.kern.branchkernelparam.Bv.value, 'Branch model Phi matrix', np.round(mV.GetPhi(), 2))
print('Fitted mv', mV)

# Plot lower bound surface
if(fModelSelectionGrid):
    print('================ Model selection by grid search ================')
    timeStart = time.time()
    cb = np.array([0.25, 0.75])  # np.linspace(0.25, 0.75, n)
    obj = np.zeros(cb.size)
    for ib, b in enumerate(cb):
        mV.UpdateBranchingPoint(np.ones((1, 1))*b)
        InitParams(mV)
        print('Model prior to optimi', mV)
        mV.optimize()
        obj[ib] = mV.objectiveFun()
        print('B=', b, 'kernel branch point', mV.kern.branchkernelparam.Bv.value, 'loglig=', obj[ib])
        print(mV)
        if(fPlot):
            VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
            plt.title('B=%g ll=%.2f' % (b, obj[ib]))
    if(fPlot):
        plt.figure()
        plt.plot(cb, obj)
        plt.plot(cb[np.argmin(obj)], obj[np.argmin(obj)], 'ro')
        v = plt.axis()
        plt.plot([trueB[0], trueB[0]], v[-2:], '--m', linewidth=2)
        plt.legend(['Objective', 'mininum', 'true branching point'], loc=2)
        plt.title('log likelihood surface for different branching points')
    print('Model selection took %g secs.' % (time.time()-timeStart))

if(fBO):
    # Bayesian optimiser
    # Create model
    Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=trueB.copy(), fDebug=fDebug) + GPflow.kernels.White(1)
    Kbranch.branchkernelparam.kern.variance = 1
    Kbranch.white.variance = 1e-4  # controls the discontinuity magnitude, the gap at the branching point
    Kbranch.white.variance.fixed = True  # jitter for numerics
    Kbranch.branchkernelparam.kern.variance.fixed = True
    print('Kbranch matrix', Kbranch.compute_K(XExpanded, XExpanded))
    print('Branching K free parameters', Kbranch.branchkernelparam)
    print('Branching K branching parameter', Kbranch.branchkernelparam.Bv.value)
    # Initialise all model parameters using the OMGP model
    # Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
    # has one common set.
    mb = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices, mo.phi, Kbranch.branchkernelparam.Bv.value, fDebug=fDebug)
    InitParams(mb)
    if(fUsePriors):
        mb.kern.branchkernelparam.kern.lengthscales.prior = GPflow.priors.Gaussian(np.ptp(t), np.square(np.ptp(t) / 3.))
        print('Warning: Must not use prior for lengthscale if it is a fixed parameter.')
        Kbranch.branchkernelparam.kern.lengthscales.fixed = False
    else:
        Kbranch.branchkernelparam.kern.lengthscales.fixed = True

    print('Initialised mb', mb)
    mb.UpdateBranchingPoint(np.ones((1, 1))*0.2)
    myobj = BayesianOptimiser.objectiveBAndK(mb)
    eps = 1e-5
    if(Kbranch.branchkernelparam.kern.lengthscales.fixed):
        bounds = [(t.min(), t.max()), (eps, 5 * Y.var()), (1, 10.*np.ptp(t))]
        # Branching point, kernel var, kernel len
    else:
        bounds = [(t.min(), t.max()), (eps, 5 * Y.var())]
        # Branching point, kernel var
    print('Bounds used in optimisation: =', bounds)
    t0 = time.time()
    BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f, bounds=bounds)
    max_iter = 40
    nrestart = 1
    n_cores = 6
    BOobj.run_optimization(max_iter,                            # Number of iterations
                           acqu_optimize_method='fast_random',  # method to optimize the acq. function
                           acqu_optimize_restarts=nrestart,
                           batch_method='lp',
                           n_inbatch=n_cores,                   # size of the collected batches (= number of cores)
                           eps=1e-6)                            # secondary stop criteria (on top of iters)
    print('Bayesian optimisation took %g secs. ' % (time.time() - t0))
    print('Solution found by BO x_opt =  ' + str(BOobj.x_opt) + 'fx_opt = ' + str(BOobj.fx_opt))
    if(fPlot):
        objAtMin = BOobj.f(BOobj.x_opt[None, :])  # get solution, update mb
        VBHelperFunctions.plotVBCode(mb, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
        plt.title('Bayesian Optimisation B=%g ll=%.2f' % (mb.kern.branchkernelparam.Bv.value, objAtMin))

