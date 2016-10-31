# Generic libraries
import numpy as np
import GPy
from random import seed
from matplotlib import pyplot as plt
# Branching files#
import GPflow
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk


seedpr = 4
N = 100
noiseInSamples = 0.001
Btry = [0.1, 0.5, 0.8, np.nan]  # Real B cases
f, axarr = plt.subplots(len(Btry), sharex=False, sharey=False, figsize=(10, 10))

# kernel hyperparameters
kerlen = 4
kervar = 2
B = np.ones((1, 1))*0.5
t = np.linspace(0, 1, N)
# Create tree structures
tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
tree.add(None, 1, B)  # B can be anything here
(fm, _) = tree.GetFunctionBranchTensor()
XExpanded, _, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)
Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=B)
Kbranch.kern.variance = kerlen
Kbranch.kern.lengthscales = kervar
KInt = GPflow.kernels.Matern32(1)
KInt.variance = kerlen
KInt.lengthscales = kervar
# Branching GP - true locations
# Common data structures
# OMGP
komgp1 = GPy.kern.Matern32(1)
komgp1.lengthscale = kerlen
komgp1.variance = kervar
komgp1.fix()
komgp2 = GPy.kern.Matern32(1)
komgp2.lengthscale = kerlen
komgp2.variance = kervar
komgp2.fix()
# Branching kernel
kb = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=np.zeros((1, 1))) + GPflow.kernels.White(1)
kb.branchkernelparam.kern.variance = kervar
kb.branchkernelparam.kern.variance.fixed = True
kb.branchkernelparam.kern.lengthscales = kerlen
kb.branchkernelparam.kern.lengthscales.fixed = True
kb.white.variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
kb.white.variance.fixed = True  # jitter for numerics
# Null model - integrated GP model
ki = GPflow.kernels.Matern32(1) + GPflow.kernels.White(1)
ki.matern32.variance = kerlen
ki.matern32.variance.fixed = True
ki.matern32.lengthscales = kervar
ki.matern32.lengthscales.fixed = True
ki.white.variance = 1e-6
ki.white.variance.fixed = True
# Run over samples for different branching values
mlist = list()  # save model, prediction, etc for every true B, and candidate B
for ibTrue, bTrue in enumerate(Btry):
    # Get sample for each true branching location and non-branching model
    if np.isnan(bTrue):
        bs = 'Ind'
        # Take random sample so training data same and we have replication
        # leave out first and last point so we can add them back in
        Xmid = XExpanded[3:-3, :]
        X = Xmid[np.random.choice(Xmid.shape[0], N-2, replace=False), :]
        X = np.vstack([X, XExpanded[0, :], XExpanded[-1, :]])
        X = X[X[:, 0].argsort(), :]  # sort for easy plotting
        Y = bk.SampleKernel(KInt, X, tol=noiseInSamples)
    else:
        Y, X = Kbranch.SampleKernel(XExpanded, bTrue, tol=noiseInSamples)
        bs = str(bTrue)
    # Data structures
    tReplicated = X[:, 0]  # for a single site t we have multiple observations after branch pt
    # plotting
    axarr[ibTrue].set_title('b=%f N=%g noise=%f' % (bTrue, N, noiseInSamples))
    axarr[ibTrue].scatter(tReplicated, Y)
