import GPflow
import numpy as np
import time
import pickle as pickle
from . import assigngp_dense
from . import branch_kernParamGPflow as bk
from . import BranchingTree as bt
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


def GetFunctionIndexListGeneral(Xin):
    ''' Function to return index list  and input array X repeated as many time as each possible function '''
    # limited to one dimensional X for now!
    assert Xin.shape[0] == np.size(Xin)
    indicesBranch = []

    XSample = np.zeros((Xin.shape[0], 2), dtype=float)
    Xnew = []
    inew = 0
    functionList = list(range(1, 4))  # can be assigned to any of root or subbranches, one based counting

    for ix, x in enumerate(Xin):
        XSample[ix, 0] = Xin[ix]
        XSample[ix, 1] = np.random.choice(functionList)
        # print str(ix) + ' ' + str(x) + ' f=' + str(functionList) + ' ' +  str(XSample[ix,1])
        idx = []
        for f in functionList:
            Xnew.append([x, f])  # 1 based function list - does kernel care?
            idx.append(inew)
            inew = inew + 1
        indicesBranch.append(idx)
    Xnewa = np.array(Xnew)
    return (Xnewa, indicesBranch, XSample)


def SetXExpandedBranchingPoint(XExpanded, B):
    ''' Return XExpanded by removing unavailable branches '''
    # before branching pt, only function 1
    X1 = XExpanded[np.logical_and(XExpanded[:, 0] <= B, XExpanded[:, 1] == 1).flatten(), :]
    # after branching pt, only functions 2 and 2
    X23 = XExpanded[np.logical_and(XExpanded[:, 0] > B, XExpanded[:, 1] != 1).flatten(), :]
    return np.vstack([X1, X23])


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
