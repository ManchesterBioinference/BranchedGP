# Generic libraries
import argparse
import GPflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from GPclust import OMGP
import GPy
import GPyOpt
import pickle
import time
import pods
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense


def plotVBCode(mV, lw=3., fs=10, fPlotVar=False):
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


def plotGene(t,g):
    with plt.style.context('seaborn-whitegrid'):
        colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
        plt.figure(figsize=(10, 10))
        for lab, c in zip(labelLegend, colors):
            y1 = t[labels == lab]
            y2 = g[labels == lab]
            plt.scatter(y1,y2,label=lab, c=c,s=80)
            plt.text(np.median(y1),np.median(y2),lab, fontsize=45, color='blue')
        plt.legend(loc='upper left')


def InitParams(m):
    m.likelihood.variance = mo.variance.values[0]
    # set lengthscale to maximum
    m.kern.branchkernelparam.kern.lengthscales = np.max(
        np.array([mo.kern[0].lengthscale.values, mo.kern[1].lengthscale.values]))
    # set process variance to average
    m.kern.branchkernelparam.kern.variance = np.mean(
        np.array([mo.kern[0].variance.values, mo.kern[1].variance.values]))


def LoadMouseQPCRData(subsetSelection=0):
    # UNDONE should also return labels
    # From manifold load pseudotime, Y and labels
    dictData = pickle.load(open("data/guo_ssData.p", "rb"), encoding='latin1')
    YGPLVM = dictData['YGPLVM']
    ptFull = dictData['pt']
    print('Loaded GPLVM data/guo_ssData.p with nrowsXncols = ' + str(YGPLVM.shape) + '.')
    assert ptFull.ndim == 1
    assert ptFull.size == YGPLVM.shape[0]
    data = pods.datasets.singlecell()
    genes = data['Y']
    labels = data['labels']
    assert genes.shape[0] == ptFull.size
    if(subsetSelection == 0):
        pt = ptFull[:].copy()
        Y = YGPLVM.copy()
        Ygene = genes
        labels = labels
    else:
        # subset selection
        pt = ptFull[::subsetSelection].copy()
        Y = YGPLVM[::subsetSelection, :].copy()
        Ygene = genes.iloc[::subsetSelection, :]
        labels = labels[::subsetSelection]
    assert labels.size == Ygene.shape[0]
    print('LoadMouseQPCRData output')
    labelLegend = np.unique(labels)
    return pt, Y, Ygene, labels, labelLegend

########################################
#         Test parameters
########################################
fOptimizeLocal = False  # do we do any local optimization?
fPlot = True  # do we do plots?
fDebug = False  # Enable debugging output - tensorflow print ops
subsetSel = 0  # use every nth point - should use 0 for all data
########################################
np.set_printoptions(precision=4)  # precision to print numpy array
seed = 43
np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)
# Data loading
pt, Yall, Ygene, labels, labelLegend = LoadMouseQPCRData(subsetSel)
t = pt/100.
if(fPlot):
    plt.ion()
# Run code
N = t.size
# get gene name
parser = argparse.ArgumentParser(description='Process genes..')
parser.add_argument('g', help='gene name')
args = vars(parser.parse_args())
ginter = args['g']
print('Processing gene %s' % ginter)
Y = Ygene[ginter].values[:, None]
trueB = np.ones((1, 1))*0.4  # not really the true one
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

# Bayesian optimiser
# Create model
Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=trueB.copy(), fDebug=fDebug) + GPflow.kernels.White(1)
Kbranch.branchkernelparam.kern.variance = 1
Kbranch.white.variance = 1e-4  # controls the discontinuity magnitude, the gap at the branching point
Kbranch.white.variance.fixed = True  # jitter for numerics
print('Kbranch matrix', Kbranch.compute_K(XExpanded, XExpanded))
print('Branching K free parameters', Kbranch.branchkernelparam)
print('Branching K branching parameter', Kbranch.branchkernelparam.Bv.value)
# Initialise all model parameters using the OMGP model
# Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
# has one common set.
mb = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices, mo.phi, Kbranch.branchkernelparam.Bv.value, fDebug=fDebug)
InitParams(mb)
eps = 1e-5
if(fOptimizeLocal):
    Kbranch.branchkernelparam.kern.lengthscales.fixed = True
    Kbranch.branchkernelparam.kern.variance.fixed = True
    bounds = [(t.min(), t.max()), (eps, 5 * Y.var()), (1, 10.*np.ptp(t))]
    # Branching point, kernel var, kernel len
else:
    # If not optimising local - keep fixed=False to allow updating without recompile
    bounds = [(t.min(), t.max()), (eps, 5 * Y.var()), (1, 10.*np.ptp(t)), (eps, 1 * Y.var())]
    # Branching point, kernel var, kernel len, lik vari

parametersTry = [0.35, 2.41, 1.04]  # Fgf4

# parametersTry = [0.16, 4.85, 1.06]  # Pdgfra
mb.UpdateBranchingPoint(np.ones((1, 1))*parametersTry[0])
Kbranch.branchkernelparam.kern.variance = np.array(parametersTry[1])
Kbranch.branchkernelparam.kern.lengthscales = np.array(parametersTry[2])
print('Initialised mb', mb)
VBHelperFunctions.plotVBCode(mb, labels=labels, figsizeIn=(5, 5), fPlotVar=True)
plt.title('Init %s B=%g ll=%.2f' % (ginter, mb.kern.branchkernelparam.Bv.value, mb.objectiveFun()))

plotGene(t, Y)
plotVBCode(mb, lw=3., fs=10, fPlotVar=True)
plt.title('%s B=%g' % (ginter, mb.kern.branchkernelparam.Bv.value))

# 
# mb.optimize()
# VBHelperFunctions.plotVBCode(mb, labels=labels, figsizeIn=(5, 5), fPlotVar=True)
# plt.title('Optimised %s B=%g ll=%.2f' % (ginter, mb.kern.branchkernelparam.Bv.value, mb.objectiveFun()))
