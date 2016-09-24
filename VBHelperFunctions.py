import GPflow
import numpy as np
import time
import pickle as pickle
import assigngp_dense
import branch_kernParamGPflow as bk
import BranchingTree as bt
import pods
import GPyOpt
from matplotlib import pyplot as plt
from matplotlib import cm


def plotVBCode(mV, figsizeIn=(20, 10), lw=3., fs=10, labels=None, fPlotPhi=True, fPlotVar=False):
    fig = plt.figure(figsize=figsizeIn)
    B = mV.kern.branchkernelparam.Bv.value.flatten()
    assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)
    pt = mV.t
    l = np.min(pt)
    u = np.max(pt)
    d = 0  # constraint code to be 1D for now
    for f in range(1, 4):
        if(f == 1):
            ttest = np.linspace(l, B, 100)[:, None]  # root
        else:
            ttest = np.linspace(B, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest * 0 + f))
        mu, var = mV.predict_f(Xtest)
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
        mean, = plt.plot(ttest, mu[:, d], linewidth=lw)
        col = mean.get_color()
        if(fPlotVar):
            plt.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            plt.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)

    v = plt.axis()
    plt.plot([B, B], v[-2:], '-m', linewidth=lw)

    # Plot Phi or labels
    if(fPlotPhi):
        Phi = FlattenPhi(mV)
        gp_num = 1  # can be 0,1,2 - Plot against this
        plt.scatter(pt, mV.Y.value[:, d], c=Phi[:, gp_num], vmin=0., vmax=1, s=40)
        plt.colorbar(label='GP {} assignment probability'.format(gp_num))
    elif(labels is not None):
        # plot labels
        labelLegend = np.unique(labels)
        with plt.style.context('seaborn-whitegrid'):
            colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
            for lab, c in zip(labelLegend, colors):
                y1 = pt[labels == lab]
                y2 = mV.Y.value[labels == lab]
                plt.scatter(y1, y2, label=lab, c=c, s=80)
                plt.text(np.median(y1), np.median(y2), lab, fontsize=45, color='blue')
            plt.legend(loc='upper left')
    return fig

def InitialisePhiFromOMGP(mV, phiOMGP, b, Y, pt):
    # branching location needed
    # create index
    N = Y.shape[0]
    assert phiOMGP.shape[0] == N
    assert phiOMGP.shape[1] == 2  # run OMGP with K=2 trajectories

    phiInitial = np.zeros((N, 3 * N))
    # large neg number makes exact zeros, make smaller for added jitter
    phiInitial_invSoftmax = -9. * np.ones((N, 3 * N))
    XExpanded = np.zeros((3 * N, 2))
    XExpanded[:] = np.nan
    #phiInitial[:] = np.nan
    eps = 1e-12
    iterC = 0
    for i, p in enumerate(pt):
        if(p < b):  # before branching - it's the root
            phiInitial[i, iterC:iterC + 3] = np.array([1 - 2 * eps, 0 + eps, 0 + eps])
        else:
            phiInitial[i, iterC:iterC + 3] = np.hstack([eps, phiOMGP[i, :] - eps])
        phiInitial_invSoftmax[i, iterC:iterC + 3] = np.log(phiInitial[i, iterC:iterC + 3])
        XExpanded[iterC:iterC + 3, 0] = pt[i]
        XExpanded[iterC:iterC + 3, 1] = np.array(list(range(1, 4)))
        iterC += 3

    assert np.any(np.isnan(phiInitial)) == False, 'no nans please ' + str(np.nonzero(np.isnan(phiInitial)))
    assert np.any(phiInitial < 0) == False, 'no negatives please ' + str(np.nonzero(np.isnan(phiInitial)))
    assert np.any(np.isnan(XExpanded)) == False, 'no nans please in XExpanded '

    if(mV is not None):
        assert np.allclose(Y, mV.Y.value)
        assert np.allclose(pt, mV.t)

        mV.logPhi = phiInitial_invSoftmax
    return phiInitial, phiInitial_invSoftmax, XExpanded


def InitModels(pt, XExpanded, Y):
    # code that's a bit crappy - we dont need this
    tree = bt.BinaryBranchingTree(0, 90, fDebug=False)  # set to true to print debug messages
    tree.add(None, 1, 10)  # single branching point
    (fm, _) = tree.GetFunctionBranchTensor()
    KbranchVB = bk.BranchKernelParam(GPflow.kernels.RBF(1), fm, BvInitial=np.ones((1, 1))) + GPflow.kernels.White(1)
    # KbranchVB = bk.BranchKernelParam(GPflow.kernels.RBF(1), fm,
    # BvInitial=np.ones((1,1))) + GPflow.kernels.White(1) +
    # GPflow.kernels.Linear(1) + GPflow.kernels.Constant(1) # other copy of
    # kernel
    KbranchVB.branchkernelparam.Bv.fixed = True
    mV = assigngp_dense.AssignGP(pt, XExpanded, Y, KbranchVB)
    mV.kern.white.variance = 1e-6
    mV.kern.white.variance.fixed = True
    mV._compile()  # creates objective function
    return mV


def FlattenPhi(mV):
    # return flattened and rounded Phi i.e. N X 3
    phiFlattened = np.zeros((mV.Y.shape[0], 3))  # only single branching point
    Phi = np.round(np.exp(mV.logPhi._array), decimals=4)
    iterC = 0
    for i, _ in enumerate(mV.t):
        phiFlattened[i, :] = Phi[i, iterC:iterC + 3]
        iterC += 3
    return phiFlattened
