from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from .data_generation import DEFAULT_COLOURS, Colours


def plot_many_branching_posteriors(
    data,
    Bmode,
    Phi,
    prediction,
    fig=None,
    axa=None,
    num_figs=(2, 5),
    figsize=(10, 10),
    output_index=slice(None, None, None),
    fPlotVar=False,
    fPlotPhi=False,
    show_legend=False,
):
    """
    Function to plot many BGP posteriors at once
    """
    Y_indexed = data.Y[:, output_index]
    assert len(num_figs) == 2
    assert Y_indexed.shape[1] == np.prod(num_figs)
    if axa is None:
        fig, axa = plt.subplots(num_figs[0], num_figs[1], figsize=figsize)
    if not isinstance(axa, np.ndarray):
        axa = np.array([axa])  # for 1 output case
    data.plot(axa=axa.flatten(), show_legend=False)  # , plot_skip=slice(None, None, 4))
    # or can be empty list or?
    for (ib, branching_pointsi), ax in zip(
        enumerate(data.branching_points[output_index]), axa.flatten()
    ):
        ttestl, mul, varl = prediction["xtest"], prediction["mu"], prediction["var"]
        # ensure we do not plot branching points past domain
        estimated_b = np.round(max([0, min([1, Bmode[ib]])]), 2)
        plotBranchModel(
            estimated_b,
            branching_pointsi,
            data.t,
            data.Y,
            ttestl,
            mul,
            varl,
            Phi,
            fPlotVar=fPlotVar,
            d=ib,
            ax=ax,
            fColorBar=False,
            fPlotPhi=fPlotPhi,
            show_legend=show_legend,
        )
    return fig, axa


def CalculateBranchingEvidence(d, Bsearch=None):
    """
    :param d: output dictionary from FitModel
    :param Bsearch: candidate list of branching points
    :return: posterior probability of branching at each point and log Bayes factor
    of branching vs not branching
    """
    if Bsearch is None:
        Bsearch = list(np.linspace(0.05, 0.95, 5)) + [1.1]
    # Calculate probability of branching at each point
    o = d["loglik"][:-1]
    pn = np.exp(o - np.max(o))
    p = pn / pn.sum()  # normalize

    # Calculate log likelihood ratio by averaging out
    o = d["loglik"]
    Nb = o.size - 1
    if Nb != len(Bsearch) - 1:
        raise NameError(
            "Passed in wrong length of Bsearch is %g- should be %g" % (len(Bsearch), Nb)
        )
    obj = o[:-1]
    illmax = np.argmax(obj)
    llmax = obj[illmax]
    lratiostable = (
        llmax
        + np.log(1 + np.exp(obj[np.arange(obj.size) != illmax] - llmax).sum())
        - o[-1]
        - np.log(Nb)
    )

    return {"posteriorBranching": p, "logBayesFactor": lratiostable}


def PlotBGPFit(
    GPy,
    GPt,
    Bsearch,
    d,
    figsize=(5, 5),
    height_ratios=[5, 1],
    colorarray=["darkolivegreen", "peru", "mediumvioletred"],
):
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
    fig, axa = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": height_ratios}
    )
    ax = axa[0]
    y, pt, mul, ttestl = GPy, GPt, d["prediction"]["mu"], d["prediction"]["xtest"]
    lw = 4
    for f in range(3):
        mu = mul[f]
        ttest = ttestl[f]
        col = colorarray[f]  # mean.get_color()
        ax.plot(ttest, mu, linewidth=lw, color=col, alpha=0.7)
        gp_num = 1  # can be 0,1,2 - Plot against this
        PhiColor = ax.scatter(
            pt, y, c=d["Phi"][:, gp_num], vmin=0.0, vmax=1, s=40, alpha=0.7
        )
    _ = fig.colorbar(PhiColor, ax=ax, orientation="horizontal")
    ax = axa[1]
    p = CalculateBranchingEvidence(d, Bsearch)["posteriorBranching"]
    ax.stem(Bsearch[:-1], p)
    return fig, axa


def plotBranchModel(
    estimatedB: np.ndarray,
    globalB: np.ndarray,
    trueB: Optional[np.ndarray],
    pt: np.ndarray,
    Y: np.ndarray,
    ttestl: np.ndarray,
    mul: np.ndarray,
    varl: np.ndarray,
    Phi: np.ndarray,
    figsizeIn: Tuple[int, int] = (5, 5),
    lw: float = 3.0,
    fs: int = 10,
    labels=None,
    fPlotPhi: bool = True,
    fPlotVar: bool = False,
    ax: Optional[plt.Axes] = None,
    fColorBar: bool = True,
    colorarray: Optional[Colours] = None,
    d: int = 0,
    show_legend: bool = True,
    gene_name: str = "A",
):
    """Plotting code that does not require access to the model but takes as input predictions."""
    colorarray = colorarray or DEFAULT_COLOURS

    if ax is None:
        fig = plt.figure(figsize=figsizeIn)
        ax: plt.Axes = fig.gca()  # type: ignore
    else:
        fig = plt.gcf()
    assert isinstance(ax, plt.Axes)  # help out MyPy

    for f in range(3):
        ttest = ttestl[f].flatten()
        mu = mul[f]
        var = varl[f]
        col = colorarray[f]  # mean.get_color()
        if f == 0:
            idx = np.flatnonzero(ttest < estimatedB)
        else:
            idx = np.flatnonzero(ttest >= estimatedB)

        # plot the mean
        ax.plot(ttest[idx], np.asarray(mu)[idx, d], linewidth=lw, color=col)

        if fPlotVar:
            ax.fill_between(
                ttest[idx],
                mu[idx, d] - 2 * np.sqrt(var[idx, d]),
                mu[idx, d] + 2 * np.sqrt(var[idx, d]),
                alpha=0.2,
                color=col,
            )

    y_bounds = ax.axis()[-2:]
    ax.plot(
        [estimatedB, estimatedB], y_bounds, "--m", linewidth=lw, label="Estimated BP"
    )
    if trueB:
        ax.plot([trueB, trueB], y_bounds, "--b", linewidth=lw, label="True BP")

    if show_legend:
        ax.legend()

    # Plot Phi or labels
    if fPlotPhi:
        gp_num = 1  # can be 0,1,2 - Plot against this
        PhiColor = ax.scatter(pt, Y[:, d], c=Phi[:, gp_num], vmin=0.0, vmax=1, s=40)
        if fColorBar:
            fig.colorbar(PhiColor, label="GP {} assignment probability".format(gp_num))
        else:
            return fig, PhiColor

    return fig


def predictBranchingModel(m, Bi=None, full_cov=False):
    """return prediction of branching model"""
    pt = m.t
    l = np.min(pt)
    u = np.max(pt)
    mul = list()
    varl = list()
    ttestl = list()

    for f in range(1, 4):
        ttest = np.linspace(l, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest * 0 + f))
        if full_cov:
            mu, var = m.predict_f_full_cov(Xtest)
        else:
            mu, var = m.predict_f(Xtest)
        # print('mu', mu)
        # idx = np.isnan(mu)
        # print('munan', mu[idx], var[idx], ttest[idx])
        assert np.all(np.isfinite(mu)), "All elements should be finite but are " + str(
            mu
        )
        assert np.all(np.isfinite(var)), "All elements should be finite but are " + str(
            var
        )
        mul.append(mu)
        varl.append(var)
        ttestl.append(ttest)
    return ttestl, mul, varl


# Original code
def GetFunctionIndexListGeneral(Xin):
    """Function to return index list  and input array X repeated as many time as each possible function"""
    # limited to one dimensional X for now!
    # print(Xin.shape)
    assert Xin.shape[0] == np.size(Xin)
    indicesBranch = []

    XSample = np.zeros((Xin.shape[0], 2), dtype=float)
    Xnew = []
    inew = 0
    functionList = list(
        range(1, 4)
    )  # can be assigned to any of root or subbranches, one based counting

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


# Author Sumon
# def GetFunctionIndexListGeneral(Xin):
#     ''' Function to return index list  and input array X repeated as many time as each possible function '''
#     # limited to one dimensional X for now!
#     assert Xin.shape[0] == np.size(Xin)
#     indicesBranch = []
#
#     XSample = np.zeros((Xin.shape[0], 2), dtype=float)
#     Xnew = []
#     inew = 0
#     functionList = list(range(1, 4))  # can be assigned to any of root or subbranches, one based counting
#
#     for f in functionList:
#         idx = []
#         for ix, x in enumerate(Xin):
#             XSample[ix, 0] = Xin[ix]
#             XSample[ix, 1] = np.random.choice(functionList)
#             # print str(ix) + ' ' + str(x) + ' f=' + str(functionList) + ' ' +  str(XSample[ix,1])
#
#             Xnew.append([x, f])  # 1 based function list - does kernel care?
#             idx.append(inew)
#             inew = inew + 1
#         indicesBranch.append(idx)
#     Xnewa = np.array(Xnew)
#     return (Xnewa, indicesBranch, XSample)


def SetXExpandedBranchingPoint(XExpanded, B):
    """Return XExpanded by removing unavailable branches"""
    # before branching pt, only function 1
    X1 = XExpanded[
        np.logical_and(XExpanded[:, 0] <= B, XExpanded[:, 1] == 1).flatten(), :
    ]
    # after branching pt, only functions 2 and 2
    X23 = XExpanded[
        np.logical_and(XExpanded[:, 0] > B, XExpanded[:, 1] != 1).flatten(), :
    ]
    return np.vstack([X1, X23])
