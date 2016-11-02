# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
import GPy
from GPclust import OMGP
import time
from random import seed
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense
from BranchedGP import assigngp_denseSparse


def GetSampleGPFitBranchingModel(seedpr, fTesting=False, N=50, nsparseGP=None, noiseInSamples=0.001):
    '''
    N -> Number of points in the sample path, we sample *3
    noiseInSamples -> how much noise to add in GP samples 
    '''
    np.random.seed(seed=seedpr)  # easy peasy reproducibeasy
    tf.set_random_seed(seedpr)
    seed(seedpr)
    ########################################
    #         Test parameters
    ########################################
    if(fTesting):
        maxiters = 30
    else:
        maxiters = 100
    if(fTesting):
        Btry = [0.5, np.nan]  # Real B cases
    else:
        Btry = [0.1, 0.5, 0.8, np.nan]  # early, med, late, none
    # Grid search locations
    if(fTesting):
        BgridSearch = [0.5]
    else:
        BgridSearch = np.linspace(0, 1, 11)
        # may have problem in Phi matrix calculation (try GetPhi())
        BgridSearch[0] = 1e-6
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
    # setup structures
    errorInBranchingPt = np.zeros((len(Btry)))  # + integrated GP
    logLikelihoodRatio = np.zeros((len(Btry)))
    timingInfo = np.zeros((len(Btry), len(BgridSearch)))
    errorInBranchingPt[:] = np.nan
    logLikelihoodRatio[:] = np.nan
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
        XExpandedRepl, indicesRepl, _ = VBHelperFunctions.GetFunctionIndexListGeneral(tReplicated)
        # Use OMGP for initialisation of branching model
        mo = OMGP(tReplicated[:, None], Y, K=2, variance=0.01, kernels=[komgp1, komgp2],
                  prior_Z='DP')  # use a truncated DP with K=2 UNDONE
        mo.optimize(step_length=0.01, maxiter=maxiters, verbose=False)  # This just optimizers allocations
        # Integrated GP
        mi = GPflow.gpr.GPR(tReplicated[:, None], Y, ki)
        mi.likelihood.variance = noiseInSamples  # initialise to true value
        mi.optimize(disp=0, maxiter=maxiters)  # only likelihood variance not fixed
        objInt = -mi.compute_log_likelihood()-mi.compute_log_prior()
        # Branching model
        if(nsparseGP is None):
            m = assigngp_dense.AssignGP(tReplicated, XExpandedRepl, Y, kb,
                                        indicesRepl, mo.phi, kb.branchkernelparam.Bv.value)
        else:
            ir = np.random.choice(XExpandedRepl.shape[0], nsparseGP)
            ZExpanded = XExpandedRepl[ir, :]
            m = assigngp_denseSparse.AssignGPSparse(tReplicated, XExpandedRepl, Y, kb,
                                                    indicesRepl, mo.phi, kb.branchkernelparam.Bv.value,
                                                    ZExpanded)

        # Do grid search
        obj = np.zeros(len(BgridSearch))
        mlocallist = list()
        for ib, b in enumerate(BgridSearch):
            tstart = time.time()
            print('considering branch', b, 'btrue', bs)
            # if code below fails - just throw away entire run
            m.UpdateBranchingPoint(np.ones((1, 1))*b)
            m.likelihood.variance = noiseInSamples     # reset but not fix
            m.optimize(disp=0, maxiter=maxiters)
            obj[ib] = m.objectiveFun()
            timingInfo[ibTrue, ib] = time.time()-tstart
            # do prediction and save results
            Phi = m.GetPhi()
            ttestl, mul, varl = VBHelperFunctions.predictBranchingModel(m)
            # do not save model as this will break between GPflow versions
            mlocallist.append({'candidateB': b, 'obj': obj[ib], 'Phi': Phi,
                               'ttestl': ttestl, 'mul': mul, 'varl': varl, 'm': m})  # save model for debugging
        S = np.asarray([BgridSearch, obj]).T
        im = np.argmin(S[:, 1])
        print('TrueB %s\n===============\n' % bs, S, '\nMinimum at', S[im, :])
        print('Completed in', timingInfo[ibTrue, :].sum(), ' seconds.')
        mlist.append({'trueBStr': bs, 'bTrue': bTrue,
                      'pt': m.t, 'Y': m.Y.value, 'obj': obj,
                      'mlocallist': mlocallist})
        # distance from true B
        if np.isnan(bTrue):
            # for integrate GP, store most likely branching point found - should also look at likelihood ratio
            errorInBranchingPt[ibTrue] = S[im, 0]
        else:
            errorInBranchingPt[ibTrue] = bTrue - S[im, 0]
        '''
        Order genes by log likelihood ratio
        -log ( p(B_ML) / p(GPR) ) = - log(p(B)) + (-log(p(G))) = R
        if R == 0: p(B)==p(GPR)
        if R > 0, -log(p(B)) > -log(p(G)) => log(p(B)) < log(p(G)) .. Model is integrative
        So the lower the R, the stronger the evidence for branching
        '''
        logLikelihoodRatio[ibTrue] = S[im, 1] - objInt
    # all done!
    returnDict = {'errorInBranchingPt': errorInBranchingPt,
                  'logLikelihoodRatio': logLikelihoodRatio,
                  'Btry': Btry, 'BgridSearch': BgridSearch,
                  'mlist': mlist, 'timingInfo': timingInfo}
    return returnDict
