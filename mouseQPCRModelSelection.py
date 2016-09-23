strFile = 'mouseQPCRModelSelect'
from matplotlib import pyplot as plt
import GPflow
import numpy as np
import time
import pickle as pickle

import AssignGPGibbsSingleLoop
import branch_kernParamGPflow as bk
import assigngp_dense
import BranchingTree as bt
import pods


def LoadRawData(subsetSelection=0):
    data = pods.datasets.singlecell()
    genes = data['Y']
    labels = data['labels']

    print('Loading pods Mouse ESC qpcr data ' + str(genes.shape))
    N = genes.shape[0]
    stageCell = np.zeros(N)
    stageN = np.zeros(N)
    for i, l in enumerate(labels):
        stageCell[i] = int(l[:2])
        stageN[i] = np.log2(stageCell[i]) + 1

    return genes, stageCell, stageN, labels


def LoadMouseQPCRData(subsetSelection=0):
    # UNDONE should also return labels
    # From manifold load pseudotime, Y and labels
    dictData = pickle.load(open("data/guo_ssData.p", "rb"), encoding='latin1')
    YGPLVM = dictData['YGPLVM']
    ptFull = dictData['pt']
    print('Loaded GPLVM data/guo_ssData.p with nrowsXncols = ' + str(YGPLVM.shape) + '.')
    assert ptFull.ndim == 1
    assert ptFull.size == YGPLVM.shape[0]
    if(subsetSelection == 0):
        pt = ptFull[:].copy()
        Y = YGPLVM.copy()
    else:
        # subset selection
        pt = ptFull[::subsetSelection].copy()
        Y = YGPLVM[::subsetSelection, :].copy()
    print('LoadMouseQPCRData output')
    return pt, Y


def InitModels(pt, Y, nsparse=0):
    tree = bt.BinaryBranchingTree(0, 90, fDebug=False)  # set to true to print debug messages
    tree.add(None, 1, 10)  # single branching point
    (fm, _) = tree.GetFunctionBranchTensor()

    # Initialise Kernel
    BvaluesInit = np.ones((1, 1))  # initial values

    Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, BvInitial=BvaluesInit) + GPflow.kernels.White(1)
    # NB: fix the branching point if optimizing. Kbranch.branchkernelparam.Bv.fixed = True
    print('Branching kernel =====================')
    print(Kbranch)

    if(nsparse > 0):
        l = np.min(pt)
        u = np.max(pt)
        Z = np.linspace(l, u, nsparse)
        print('Created %g inducing points in [%.1f,%.1f]' % (nsparse, l, u))
    else:
        Z = None

    print('Initialise models: MAP =====================')
    m = AssignGPGibbsSingleLoop.AssignGPGibbsFast(pt, Y, Kbranch, Z=Z)
    m.kern.white.variance = 1e-6
    m.kern.white.variance.fixed = True  # this causes Param code to fail

    # If hyperparameters are changed, this function should be called to reocmpute KChol
    m.CompileAssignmentProbability(fDebug=False, fMAP=True)
    print(m)

    KbranchVB = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, BvInitial=BvaluesInit) + \
        GPflow.kernels.White(1)  # other copy of kernel
    KbranchVB.branchkernelparam.Bv.fixed = True
    print('Initialise models: VB =====================')
    mV = assigngp_dense.AssignGP(pt, m.XExpanded, Y, KbranchVB, ZExpanded=m.ZExpanded)
    mV.kern.white.variance = 1e-6
    mV.kern.white.variance.fixed = True

    mV._compile()  # creates objective function

    return m, mV


def DoModelSelectionRuns(m, mV, Bpossible=None, strSaveState=None,
                         fSoftVBAssignment=True, fOptimizeHyperparameters=False, fReestimateMAPZ=True,
                         numMAPsteps=10, fPlotFigure=False):
    ''' Trains the model passed in'''
    if(Bpossible is None):
        Bpossible = np.array([10., 25., 45., 60.])
        #Bpossible = np.linspace(2,70,22)

    logLike = []
    logVBBound = []
    pt = mV.t

    # mV.kern.branchkernelparam.Bv.fixed = True # B not part of the state
    stateSaved = mV.get_free_state().copy()

    for ib, b in enumerate(Bpossible):
        Bcrap = np.atleast_2d(b)  # crappy branch point

        # reinitialise hyperparameters
        mV.kern.branchkernelparam.kern.lengthscales = 65  # 20 + (90. - b) / 2. # 65
        mV.kern.branchkernelparam.kern.variance = 2  # 0.0012 #  2.3158
        m.kern.branchkernelparam.kern.lengthscales = 65  # 20 + (90. - b) / 2. # 65
        m.kern.branchkernelparam.kern.variance = 2  # 0.0012 #  2.3158
        m.likelihood.variance = 0.08
        mV.likelihood.variance = m.likelihood.variance._array

        if(fOptimizeHyperparameters):
            # should recompute Kernel everytime we update kernel hyperparameters
            m.CompileAssignmentProbability(fDebug=False, fMAP=True)

        # set branching point
        m.kern.branchkernelparam.Bv = Bcrap

        print('============> B=' + str(m.kern.branchkernelparam.Bv._array.flatten()))

        # Random assignment for given branch point
        randomAssignment = AssignGPGibbsSingleLoop.GetRandomInit(pt, Bcrap, m.indices)
        # print m.XExpanded[randomAssignment,1]
        print('MAP assignment.')
        print('indices')
        print(len(m.indices))
        print(m.indices[:5])
        print('XExpanded')
        print(m.XExpanded.shape)
        print(m.XExpanded[:5, :])
        print('m.t')
        print(m.t.shape)
        print(m.t.shape[:5])
        print('------model-----------------')
        print(m)

        (chainState, bestAssignment, _, condProbs) = \
            m.InferenceGibbsMAP(fReturnAssignmentHistory=True, fDebug=False,
                                maximumNumberOfSteps=numMAPsteps,
                                startingAssignment=list(randomAssignment))

        if(fOptimizeHyperparameters):
            print('MAP Optimizing hyperparameters')
            try:
                m.OptimizeK(list(bestAssignment))
            except:
                print('MAP OptimizeK: Exception most probably due to inversion error - skipping')
                logLike.append(np.nan)
                logVBBound.append(np.nan)
                continue

            if(fReestimateMAPZ):
                print('Re-estimating MAP assignment.')
                (chainState, bestAssignment, _, condProbs) = \
                    m.InferenceGibbsMAP(fReturnAssignmentHistory=True, fDebug=False,
                                        maximumNumberOfSteps=numMAPsteps,
                                        startingAssignment=list(bestAssignment))

        print('MAP done with loglike =%.2f' % chainState[-1])
        logLike.append(chainState[-1])

        # Variational bound computation
        mV.kern.branchkernelparam.Bv = Bcrap
        print('Variational kernel branch value ' + str(mV.kern.branchkernelparam.Bv._array.flatten()))
        # Set state for assignments
        mV.InitialisePhi(m.indices, bestAssignment, b, condProbs, fSoftAssignment=fSoftVBAssignment)
        # Could also optimize!
        # try:
        VBbound = mV.compute_log_likelihood()  # mV._objective(mV.get_free_state())[0] # this is -log of bound
        # except:
        # print 'VB Exception most probably due to inversion error - skipping'
        # logLike.append(np.nan)
        # logVBBound.append(np.nan)
        # continue

        logVBBound.append(VBbound)

        if(fPlotFigure):
            # Plot MAP SolutionStackDescription
            print('Plotting figure')
            D = 1
            plt.ion()
            assigngp_dense.PlotSample(D, m.XExpanded[bestAssignment, :], 3, mV.Y, Bcrap, lw=5., fs=30,
                                      mV=mV, figsizeIn=(D * 10, D * 7), title='Posterior B=%.1f -loglik= %.2f VB= %.2f. Saving model=%s' % (b, chainState[-1], VBbound, strSaveState))

        # save mV so we can plot - also save bestAssignment
        if(strSaveState is not None):
            np.save('modelfiles/' + strSaveState + '_b' + str(ib) + '_MAPModel', bestAssignment)
            np.save('modelfiles/' + strSaveState + '_b' + str(ib) + '_VBmodel', mV.get_free_state())

        print('B=' + str(b) + '. MAP lik=%.2f, VB bound=%.2f' % (chainState[-1], VBbound))

    if(strSaveState is not None):
        saveDict = {'Bpossible': Bpossible, 'logVBBound': logVBBound, 'logLike': logLike}
        pickle.dump(saveDict, open('modelfiles/' + strSaveState + '_Summary.p', "wb"))
    return logVBBound, logLike
    print('Done with model selection.')

if __name__ == '__main__':
    # Unittest
    pt, Y = LoadMouseQPCRData(subsetSelection=3)
    m, mV = InitModels(pt, Y)
    logVBBound, logLike = DoModelSelectionRuns(m, mV, Bpossible=np.array([25.]), strSaveState='test',
                                               fSoftVBAssignment=True, fOptimizeHyperparameters=False, fReestimateMAPZ=True,
                                               numMAPsteps=10)
    print(logVBBound, logLike)
