# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import GPy
from GPclust import OMGP
import time
import pickle
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense

########################################
#         Test parameters
########################################
fTesting = False  # quick run?
N = 50  # Number of points in the sample path, we sample *3
if(fTesting):
    NSamples = 2
    maxiters = 30
    fPlot = True  # do we do plots?
else:
    NSamples = 100
    maxiters = 50
    fPlot = True

########################################
np.set_printoptions(precision=4)  # precision to print numpy array
seed = 0
np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)
# Data generation
# kernel hyperparameters
kerlen = 3
kervar = 2
noiseInSamples = 0.1  # how much noise to add in GP samples
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
KInt = GPflow.kernels.Matern32(1)
KInt.variance = kerlen
KInt.lengthscales = kervar
if(fPlot):
    plt.ion()
    plt.close('all')
# Branching GP - true locations
if(fTesting):
    Btry = [0.5, np.nan]  # Real B cases
else:
    Btry = [0.1, 0.5, 0.85, np.nan]  # early, med, late, none
# Grid search locations
if(fTesting):
    BgridSearch = [0.5]
else:
    BgridSearch = [0.0+1e-6, 0.1, 0.5, 0.85, 1.0]  # may have problem in Phi matrix calculation (try GetPhi())
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

errorInBranchingPt = np.zeros((NSamples, len(Btry)))  # + integrated GP
logLikelihoodRatio = np.zeros((NSamples, len(Btry)))
timingInfo = np.zeros((NSamples, len(Btry), len(BgridSearch)))
errorInBranchingPt[:] = np.nan
logLikelihoodRatio[:] = np.nan
for ns in range(NSamples):
    print('Samples %g starting now' % ns)
    # Run over samples for different branching values
    for ibTrue, bTrue in enumerate(Btry):
        # Get sample for each true branching location and non-branching model
        if np.isnan(bTrue):
            bs = 'Ind'
            # Take random sample so training data same and we have replication
            Xmid = XExpanded[3:-3, :]  # leave out first and last point
            X = Xmid[np.random.choice(Xmid.shape[0], N-2, replace=False), :]
            X = np.vstack([X, XExpanded[0, :], XExpanded[-1, :]])
            X = X[X[:, 0].argsort(), :]  # sort for easy plotting
            Y = bk.SampleKernel(KInt, X, tol=noiseInSamples)
        else:
            Y, X = Kbranch.SampleKernel(XExpanded, bTrue, tol=noiseInSamples)
            bs = str(bTrue)
        if(fPlot):
            bk.PlotSample(X, Y, np.ones((1, 1))*bTrue)
        # Data structures
        tReplicated = X[:, 0]  # for a single site t we have multiple observations after branch pt
        XExpandedRepl, indicesRepl, _ = VBHelperFunctions.GetFunctionIndexListGeneral(tReplicated)

        # Use OMGP for initialisation of branching model
        mo = OMGP(tReplicated[:, None], Y, K=2, variance=0.01, kernels=[komgp1, komgp2],
                  prior_Z='DP')  # use a truncated DP with K=2 UNDONE
        mo.optimize(step_length=0.01, maxiter=maxiters, verbose=False)  # This just optimizers allocations
        if(fPlot):
            plt.close('all')  # start from scratch for all GP samples
            fig = plt.figure(figsize=(5, 5))
            mo.plot()
            plt.title('OMGPInit B=%s' % bs)
        # Integrated GP
        mi = GPflow.gpr.GPR(tReplicated[:, None], Y, ki)
        mi.optimize(disp=0, maxiter=maxiters)  # only likelihood variance not fixed
        objInt = -mi.compute_log_likelihood()-mi.compute_log_prior()
        # Branching model
        mV = assigngp_dense.AssignGP(tReplicated, XExpandedRepl, Y, kb, indicesRepl, kb.branchkernelparam.Bv.value, phiInitial=mo.phi)
        # Do grid search
        obj = np.zeros(len(BgridSearch))
        for ib, b in enumerate(BgridSearch):
            tstart = time.time()
            try:
                mV.UpdateBranchingPoint(np.ones((1, 1))*b)
                mV.optimize(disp=0, maxiter=maxiters)
            except:
                print('Failed')
                continue
            obj[ib] = mV.objectiveFun()
            timingInfo[ns, ibTrue, ib] = time.time()-tstart
            if(fPlot):
                VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5), fPlotVar=True)
                plt.title('TrueB=%s, B=%g, ll=%.2f' % (bs, b, obj[ib]))
                plt.savefig("%s/testSampleGPData_s%g_tb%s_cb%g.png" % ('figs', ns, bs, ib), bbox_inches='tight')
        S = np.asarray([BgridSearch, obj]).T
        im = np.argmin(S[:, 1])
        print('TrueB %s\n===============\n' % bs, S, '\nMinimum at', S[im, :])
        print('Completed in', timingInfo[ns, ibTrue, :].sum(), ' seconds.')
        # distance from true B
        if np.isnan(bTrue):
            # for integrate GP, store most likely branching point found - should also look at likelihood ratio
            errorInBranchingPt[ns, ibTrue] = S[im, 0]
        else:
            errorInBranchingPt[ns, ibTrue] = bTrue - S[im, 0]
        '''
        Order genes by log likelihood ratio
        -log ( p(B_ML) / p(GPR) ) = - log(p(B)) + (-log(p(G))) = R
        if R == 0: p(B)==p(GPR)
        if R > 0, -log(p(B)) > -log(p(G)) => log(p(B)) < log(p(G)) .. Model is integrative
        So the lower the R, the stronger the evidence for branching
        '''
        logLikelihoodRatio[ns, ibTrue] = S[im, 1] + objInt
        # Save periodically
        saveDict = {'errorInBranchingPt': errorInBranchingPt,
                    'logLikelihoodRatio': logLikelihoodRatio,
                    'Btry': Btry, 'BgridSearch': BgridSearch}
        pickle.dump(saveDict, open("testSampleGPData.p", "wb"))
# Try sparse GP Model
# Try learning hyperparameters
# add asserts that minimum objective at true branching point
