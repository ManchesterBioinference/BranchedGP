import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os

def savefig_mine(dirname, fs):
    ensure_dir(dirname)
    ffs = dirname + '/' + fs
    plt.savefig('%s.pdf' % ffs, bbox_inches='tight')

def ensure_dir(d):
    ''' Ensure directory exists '''
    if not os.path.exists(d):
        os.makedirs(d)

def PlotBGPFit(GPy, GPt, Bsearch, d, figsize=(5, 5), height_ratios= [5, 1], colorarray=['darkolivegreen', 'peru', 'mediumvioletred']):
    """
    Plot BGP model
    :param GPt: pseudotime
    :param GPy: gene expression. Should be 0 mean for best performance.
    :param Bsearch: list of candidate branching points
    :param d: output dictionary from FitModel
    :param figsize: figure size
    :param height_ratios: ratio of assignment plot vs posterior branching time plot
    :param colorarray: colors for each branch
    :return: dictionary of log likelihood, GPflow model, Phi matrix, predictive set of points,
    mean and variance, hyperparameter values, posterior on branching time
    """
    fig, axa = plt.subplots(2, 1, figsize=figsize, sharex=True,  gridspec_kw = {'height_ratios': height_ratios})
    ax = axa[0]
    y, pt, mul, ttestl = GPy, GPt, d['prediction']['mu'], d['prediction']['xtest']
    lw = 4
    for f in range(3):
        mu = mul[f]
        ttest = ttestl[f]
        col = colorarray[f]  # mean.get_color()
        mean, = ax.plot(ttest, mu, linewidth=lw, color=col, alpha=0.7)
        gp_num = 1  # can be 0,1,2 - Plot against this
        PhiColor = ax.scatter(pt, y, c=d['Phi'][:, gp_num], vmin=0., vmax=1, s=40, alpha=0.7)
    cb=fig.colorbar(PhiColor, ax=ax, orientation="horizontal")
    ax = axa[1]
    o = d['loglik'][:-1]
    pn = np.exp(o - np.max(o))
    p = pn/pn.sum()
    ax.stem(Bsearch[:-1], p)
    return fig, axa

def plotBranchModel(B, pt, Y, ttestl, mul, varl, Phi, figsizeIn=(5, 5), lw=3., fs=10, labels=None,
                    fPlotPhi=True, fPlotVar=False, ax=None, fColorBar=True, colorarray = ['darkolivegreen', 'goldernrod', 'mediumvioletred']):
    ''' Plotting code that does not require access to the model but takes as input predictions. '''
    if(ax is None):
        fig = plt.figure(figsize=figsizeIn)
        ax = fig.gca()
    else:
        fig = plt.gcf()
    d = 0  # constraint code to be 1D for now
    for f in range(3):
        mu = mul[f]
        var = varl[f]
        ttest = ttestl[f]
        col = colorarray[f]  # mean.get_color()
        mean, = ax.plot(ttest, mu[:, d], linewidth=lw, color=col)
        if(fPlotVar):
            ax.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            ax.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
    v = ax.axis()
    ax.plot([B, B], v[-2:], '-m', linewidth=lw)
    # Plot Phi or labels
    if(fPlotPhi):
        gp_num = 1  # can be 0,1,2 - Plot against this
        PhiColor = ax.scatter(pt, Y[:, d], c=Phi[:, gp_num], vmin=0., vmax=1, s=40)
        if(fColorBar):
            fig.colorbar(PhiColor, label='GP {} assignment probability'.format(gp_num))
        else:
            return fig, PhiColor
    return fig


def predictBranchingModel(m):
    ''' return prediction of branching model '''
    pt = m.t
    B = m.kern.branchkernelparam.Bv.value.flatten()
    l = np.min(pt)
    u = np.max(pt)
    mul = list()
    varl = list()
    ttestl = list()
    for f in range(1, 4):
        if(f == 1):
            ttest = np.linspace(l, B, 100)[:, None]  # root
        else:
            ttest = np.linspace(B, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest * 0 + f))
        mu, var = m.predict_f(Xtest)
        # print('mu', mu)
        idx = np.isnan(mu)
        # print('munan', mu[idx], var[idx], ttest[idx])
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
        mul.append(mu)
        varl.append(var)
        ttestl.append(ttest)
    return ttestl, mul, varl


def plotVBCode(mV, figsizeIn=(20, 10), lw=3., fs=10, labels=None, fPlotPhi=True, fPlotVar=False):
    fig = plt.figure(figsize=figsizeIn)
    B = mV.kern.branchkernelparam.Bv.value.flatten()
    assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)
    pt = mV.t
    ttestl, mul, varl = predictBranchingModel(mV)
    d = 0  # constraint code to be 1D for now
    for f in range(3):
        mu = mul[f]
        var = varl[f]
        ttest = ttestl[f]
        mean, = plt.plot(ttest, mu[:, d], linewidth=lw)
        col = mean.get_color()
        if(fPlotVar):
            plt.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            plt.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
    v = plt.axis()
    plt.plot([B, B], v[-2:], '-m', linewidth=lw)
    # Plot Phi or labels
    if(fPlotPhi):
        Phi = mV.GetPhi()
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
