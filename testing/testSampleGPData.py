# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import GPy
from GPclust import OMGP
import time
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense

########################################
#         Test parameters
########################################
fPlot = True  # do we do plots?
fDebug = True  # Enable debugging output - tensorflow print ops
########################################
np.set_printoptions(precision=4)  # precision to print numpy array
seed = 0
np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)
# Data generation
N = 50
# kernel hyperparameters
kerlen = 3
kervar = 2
B = np.ones((1, 1))*0.5
t = np.linspace(0, 1, N)
# Create tree structures
tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
tree.add(None, 1, B)  # B can be anything here
(fm, _) = tree.GetFunctionBranchTensor()
XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)
Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=B)
Kbranch.kern.variance = kerlen
Kbranch.kern.lengthscales = kervar
KInd = bk.IndKern(GPflow.kernels.Matern32(1))
KInd.kern.variance = kerlen
KInd.kern.lengthscales = kervar
Yi = bk.SampleKernel(KInd, XExpanded)
if(fPlot):
    plt.ion()
    plt.close('all')
    bk.PlotSample(XExpanded, Yi)
    plt.title('Independent kernel')
# Branching GP
Btry = [0.1, 0.5, 0.8]
# Btry = [0.5]  # Real B cases
BgridSearch = [0.0, 0.1, 0.5, 0.8, 1.0]
# BgridSearch = [0.5]
Yb = list()
XTree = list()
YX_sampled = [Kbranch.SampleKernel(XExpanded, b, tol=0.1) for b in Btry]
if(fPlot):
    for i, YX in enumerate(YX_sampled):
        bk.PlotSample(YX[1], YX[0], np.ones((1, 1))*Btry[i])
# YX_sampled.append([Yi, XExpanded])  # Last entry is independent GP
# Fit Model on true hyperparameters
for iyx, YX in enumerate(YX_sampled):
    tstart = time.time()
    X = YX[1]  # XTree
    Y = YX[0]  # Y
    if(iyx == len(Btry)):
        bs = 'Ind'
    else:
        bs = '%s' % Btry[iyx]
    # Data structures
    tReplicated = X[:, 0]  # for a single site t we have multiple observations after branch pt
    XExpandedRepl, indicesRepl, _ = VBHelperFunctions.GetFunctionIndexListGeneral(tReplicated)

    # Use OMGP for initialisation of branching model
    komgp1 = GPy.kern.Matern32(1)
    komgp1.lengthscale = kerlen
    komgp1.variance = kervar
    komgp1.fix()
    komgp2 = GPy.kern.Matern32(1)
    komgp2.lengthscale = kerlen
    komgp2.variance = kervar
    komgp2.fix()
    mo = OMGP(tReplicated[:, None], Y, K=2, variance=0.01, kernels=[komgp1, komgp2],
              prior_Z='DP')  # use a truncated DP with K=2 UNDONE
    mo.optimize(step_length=0.01, maxiter=50, verbose=False)  # This just optimizers allocations
    if(fPlot):
        fig = plt.figure(figsize=(5, 5))
        mo.plot()
        plt.title('OMGPInit B=%s' % bs)
    kb = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=np.zeros((1, 1))) + GPflow.kernels.White(1)
    kb.branchkernelparam.kern.variance = kervar
    kb.branchkernelparam.kern.variance.fixed = True
    kb.branchkernelparam.kern.lengthscales = kerlen
    kb.branchkernelparam.kern.lengthscales.fixed = True
    kb.white.variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
    kb.white.variance.fixed = True  # jitter for numerics
#     # Null model - independent model
#     ki = bk.IndKern(GPflow.kernels.Matern32(1) + GPflow.kernels.White(1))
#     ki.kern.matern32.variance = kerlen
#     ki.kern.matern32.variance.fixed = True
#     ki.kern.matern32.lengthscales = kervar
#     ki.kern.matern32.lengthscales.fixed = True
#     ki.kern.white.variance = 1e-6
#     ki.kern.white.variance.fixed = True
#     mi = assigngp_dense.AssignGP(tReplicated, XExpandedRepl, Y, ki, indicesRepl, mo.phi, -1*np.zeros((1, 1)))
#     mi.optimize(disp=0, maxiter=30)
#     objInd = mi.objectiveFun()
    # Branching model
    mV = assigngp_dense.AssignGP(tReplicated, XExpandedRepl, Y, kb, indicesRepl, mo.phi, kb.branchkernelparam.Bv.value)
    # Do grid search
    obj = np.zeros(len(BgridSearch))
    for ib, b in enumerate(BgridSearch):
        mV.UpdateBranchingPoint(np.ones((1, 1))*b)
        mV.optimize(disp=0, maxiter=30)
        obj[ib] = mV.objectiveFun()
        print('B=', b, 'kernel branch point', mV.kern.branchkernelparam.Bv.value, 'loglig=', obj[ib])
        if(fPlot):
            VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
            plt.title('TrueB=%s, B=%g, ll=%.2f' % (bs, b, obj[ib]))
    print('TrueB %s\n===============\n' % bs, np.asarray([BgridSearch, obj]).T)
    print('Completed in', time.time()-tstart, ' seconds.')
# Try sparse GP Model
# Try learning hyperparameters
# add asserts that minimum objective at true branching point
