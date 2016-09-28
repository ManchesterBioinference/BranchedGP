# Generic libraries
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
import VBHelperFunctions
import BranchingTree as bt
import branch_kernParamGPflow as bk
import assigngp_dense
import BayesianOptimiser


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

########################################
#         Test parameters
########################################
fPlot = True  # do we do plots?
fUsePriors = False  # Test priors on kernel hyperparameters
fModelSelectionGrid = False
fBO = True  # Bayesian optimisation
fDebug = False  # Enable debugging output - tensorflow print ops
########################################
np.set_printoptions(precision=4)  # precision to print numpy array
seed = 43
np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)
# Data loading
pt, Yall, Ygene, labels, labelLegend = LoadMouseQPCRData()
t = pt/100.
if(fPlot):
    plt.ion()
    # plotGene(t, Ygene['Pdgfra'])

# Go over interesting genes
interestingGenes = ['Pdgfra', 'Gata4', 'Sox2', 'Bmp4']
# Run code
N = t.size

for ginter in interestingGenes:
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
    Kbranch.branchkernelparam.kern.variance.fixed = True
    print('Kbranch matrix', Kbranch.compute_K(XExpanded, XExpanded))
    print('Branching K free parameters', Kbranch.branchkernelparam)
    print('Branching K branching parameter', Kbranch.branchkernelparam.Bv.value)
    # Initialise all model parameters using the OMGP model
    # Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
    # has one common set.
    mb = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices, mo.phi, Kbranch.branchkernelparam.Bv.value, fDebug=fDebug)
    InitParams(mb)
    if(fUsePriors):
        mb.kern.branchkernelparam.kern.lengthscales.prior = GPflow.priors.Gaussian(np.ptp(t), np.square(np.ptp(t) / 3.))
        print('Warning: Must not use prior for lengthscale if it is a fixed parameter.')
        Kbranch.branchkernelparam.kern.lengthscales.fixed = False
    else:
        Kbranch.branchkernelparam.kern.lengthscales.fixed = True

    print('Initialised mb', mb)
    mb.UpdateBranchingPoint(np.ones((1, 1))*0.2)
    myobj = BayesianOptimiser.objectiveBAndK(mb)
    eps = 1e-5
    if(Kbranch.branchkernelparam.kern.lengthscales.fixed):
        bounds = [(t.min(), t.max()), (eps, 5 * Y.var()), (1, 10.*np.ptp(t))]
        # Branching point, kernel var, kernel len
    else:
        bounds = [(t.min(), t.max()), (eps, 5 * Y.var())]
        # Branching point, kernel var
    print('Bounds used in optimisation: =', bounds)
    t0 = time.time()
    BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f, bounds=bounds)
    max_iter = 10
    nrestart = 1
    n_cores = 10
    BOobj.run_optimization(max_iter,                            # Number of iterations
                           acqu_optimize_method='fast_random',  # method to optimize the acq. function
                           acqu_optimize_restarts=nrestart,
                           batch_method='lp',
                           n_inbatch=n_cores,                   # size of the collected batches (= number of cores)
                           eps=1e-6)                            # secondary stop criteria (on top of iters)
    print('Bayesian optimisation took %g secs. ' % (time.time() - t0))
    print('Solution found by BO x_opt =  ' + str(BOobj.x_opt) + 'fx_opt = ' + str(BOobj.fx_opt))
    if(fPlot):
        objAtMin = BOobj.f(BOobj.x_opt[None, :])  # get solution, update mb
        VBHelperFunctions.plotVBCode(mb, labels=labels, figsizeIn=(5, 5), fPlotVar=True)
        plt.title('%s B=%g ll=%.2f' % (ginter, mb.kern.branchkernelparam.Bv.value, objAtMin))
        plt.savefig("~/Dropbox/BranchedGP/figs/GuoBestfit_%s.png" % ginter, bbox_inches='tight')
        pickle.dump(mb, open("~/Dropbox/BranchedGP/figs/GuoBestfit_%s.p" % ginter, "wb"))
        #  read with pickle.load(open( "~/Dropbox/BranchedGP/figs/GuoBestfit_%s.p" % ginter, "rb" ))
