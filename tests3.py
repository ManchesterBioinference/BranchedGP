# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from GPclust import OMGP
import GPy
import pickle
import time
# Branching files
import VBHelperFunctions
import BranchingTree as bt
import branch_kernParamGPflow as bk
import assigngp_dense
import BayesianOptimiser

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
plt.ion()
plt.title('original data')
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
mV.likelihood.variance = mo.variance.values[0]
fSetParamOMGP = True
if(fSetParamOMGP):
    # set lengthscale to maximum
    mV.kern.branchkernelparam.kern.lengthscales = np.max(
        np.array([mo.kern[0].lengthscale.values, mo.kern[1].lengthscale.values]))
    # set process variance to average
    mV.kern.branchkernelparam.kern.variance = np.mean(
        np.array([mo.kern[0].variance.values, mo.kern[1].variance.values]))
# put prior to penalise short length scales
mV.kern.branchkernelparam.kern.lengthscales.prior = GPflow.priors.Gaussian(np.ptp(t), np.square(np.ptp(t) / 10.))
print('Initialised mv', mV)

# Plot results - this will call predict
VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
plt.title('Initialisation')
print('Initialisation')
print('OMGP Phi matrix', np.round(mo.phi, 2))
print('b=', mV.kern.branchkernelparam.Bv.value, 'Branch model Phi matrix', np.round(mV.GetPhi(), 2))
mV.optimize()
# Plot results - this will call predict
VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
objT = mV.compute_log_likelihood()
plt.title('Fitted model')
print('Fitted model')
print('OMGP Phi matrix', np.round(mo.phi, 2))
print('b=', mV.kern.branchkernelparam.Bv.value, 'Branch model Phi matrix', np.round(mV.GetPhi(), 2))
print('Fitted mv', mV)

# Plot lower bound surface
fModelSelection = True
if(fModelSelection):
    timeStart = time.time()
    s = pickle.dumps(mV)
    n = 6
    cb = np.linspace(0.25, 0.75, n)
    obj = np.zeros(n)
    for ib, b in enumerate(cb):
        mb = pickle.loads(s)
        mb.UpdateBranchingPoint(np.ones((1, 1))*b)
        mb.optimize()
        obj[ib] = mb.compute_log_likelihood()
        print('B=', b, 'kernel branch point', mb.kern.branchkernelparam.Bv.value, 'loglig=', obj[ib])
        print(mb)
        VBHelperFunctions.plotVBCode(mb, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
        plt.title('B=%g ll=%.2f' % (b, obj[ib]))
    plt.figure()
    plt.plot(cb, obj)
    plt.plot(cb[np.argmin(obj)], obj[np.argmin(obj)], 'ro')
    v = plt.axis()
    plt.plot([trueB[0], trueB[0]], v[-2:], '--m', linewidth=2)
    plt.legend(['Objective', 'mininum', 'true branching point'], loc=2)
    plt.title('log likelihood surface for different branching points')
    print('Model selection took %g secs.' % (time.time()-timeStart))
    # test Bayesian optimiser

fBO = False
if(fBO):
    # Run Baeysian optimiser
    pass
    
