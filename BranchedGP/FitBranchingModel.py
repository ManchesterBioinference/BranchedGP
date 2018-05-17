import numpy as np
from . import VBHelperFunctions
from . import BranchingTree as bt
from . import branch_kernParamGPflow as bk
from . import assigngp_denseSparse
from . import assigngp_dense
import gpflow
import traceback
import sys

def FitModel(bConsider, GPt, GPy, globalBranching, priorConfidence=0.80,
             M=10, likvar=1., kerlen=2., kervar=5., fDebug=False, maxiter=100,
             fPredict=True, fixHyperparameters=False):
    """
    Fit BGP model
    :param bConsider: list of candidate branching points
    :param GPt: pseudotime
    :param GPy: gene expression. Should be 0 mean for best performance.
    :param globalBranching: cell labels
    :param priorConfidence: prior confidence on cell labels
    :param M: number of inducing points
    :param likvar: initial value for Gaussian noise variance
    :param kerlen: initial value for kernel length scale
    :param kervar: initial value for kernel variance
    :param fDebug: Print debugging information
    :param maxiter: maximum number of iterations for optimisation
    :param fPredict: compute predictive mean and variance
    :param fixHyperparameters: should kernel hyperparameters be kept fixed or optimised?
    :return: dictionary of log likelihood, GPflow model, Phi matrix, predictive set of points,
    mean and variance, hyperparameter values, posterior on branching time
    """
    assert isinstance(bConsider, list), 'Candidate B must be list'
    assert GPt.ndim == 1
    assert GPy.ndim == 2
    assert GPt.size == GPy.size, 'pseudotime and gene expression data must be the same size'
    assert globalBranching.size == GPy.size, 'state space must be same size as number of cells'
    assert M >= 0, 'at least 0 or more inducing points should be given'
    phiInitial, phiPrior = GetInitialConditionsAndPrior(globalBranching, priorConfidence, infPriorPhi=True)
    # it = np.argsort(GPt)
    # print('GPt and prior', np.hstack([GPt[it][:,None], phiPrior[it,:], phiInitial[it,:]]))
    # print(phiPrior[:5, :])
    XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(GPt)
    ptb = np.min([np.min(GPt[globalBranching == 2]), np.min(GPt[globalBranching == 3])])
    tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
    tree.add(None, 1, np.ones((1, 1)) * ptb)  # B can be anything here
    (fm, _) = tree.GetFunctionBranchTensor()
    with gpflow.defer_build():
        kb = bk.BranchKernelParam(gpflow.kernels.Matern32(1), fm, b=np.zeros((1, 1))) + gpflow.kernels.White(1)
        kb.kernels[1].variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
        kb.kernels[1].variance.set_trainable(False)  # jitter for numerics
        if(M == 0):
            m = assigngp_dense.AssignGP(GPt, XExpanded, GPy, kb, indices,
                                                    np.ones((1, 1)) * ptb, phiInitial=phiInitial,
                                                    phiPrior=phiPrior)
        else:
            ZExpanded = np.ones((M, 2))
            ZExpanded[:, 0] = np.linspace(0, 1, M, endpoint=False)
            ZExpanded[:, 1] = np.array([i for j in range(M) for i in range(1, 4)])[:M]
            m = assigngp_denseSparse.AssignGPSparse(GPt, XExpanded, GPy, kb, indices,
                                                    np.ones((1, 1)) * ptb, ZExpanded, phiInitial=phiInitial, phiPrior=phiPrior)
        # Initialise hyperparameters
        m.likelihood.variance = likvar
        m.kern.kernels[0].kern.lengthscales = kerlen
        m.kern.kernels[0].kern.variance = kervar
        if(fixHyperparameters):
            print('Fixing hyperparameters')
            m.kern.kernels[0].kern.lengthscales.set_trainable(False)
            m.likelihood.variance.set_trainable(False)
            m.kern.kernels[0].kern.variance.set_trainable(False)
        else:
            if fDebug:
                print('Adding prior logistic on length scale to avoid numerical problems')
            m.kern.kernels[0].kern.lengthscales.prior = gpflow.priors.Gaussian(2, .1)
            m.kern.kernels[0].kern.variance.prior = gpflow.priors.Gaussian(3, 1)
            m.likelihood.variance.prior = gpflow.priors.Gaussian(0.1, .1)
        m.compile()

    # optimization
    ll = np.zeros(len(bConsider))
    Phi_l = list()
    ttestl_l, mul_l, varl_l = list(), list(), list()
    hyps = list()
    for ib, b in enumerate(bConsider):
        m.UpdateBranchingPoint(np.ones((1, 1)) * b, phiInitial)
        try:
            gpflow.train.ScipyOptimizer().minimize(m, maxiter=maxiter)
            # remember winning hyperparameter
            hyps.append({'likvar':  m.likelihood.variance.value, 'kerlen':  m.kern.kernels[0].kern.lengthscales.value,
                    'kervar': m.kern.kernels[0].kern.variance.value})
            ll[ib] = m.compute_log_likelihood()
        except:
            print('Failure', "Unexpected error:", sys.exc_info()[0])
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('Exception caused by model')
            print(m)
            print('-' * 60)
            ll[0] = np.nan
            # return model so can inspect model
            return {'loglik': ll, 'model': m, 'Phi': np.nan,
                    'prediction': {'xtest': np.nan, 'mu': np.nan, 'var': np.nan},
                    'hyperparameters': np.nan, 'posteriorB': np.nan}
        # prediction
        Phi = m.GetPhi()
        Phi_l.append(Phi)
        if(fPredict):
            ttestl, mul, varl = VBHelperFunctions.predictBranchingModel(m)
            ttestl_l.append(ttestl), mul_l.append(mul), varl_l.append(varl)
        else:
            ttestl_l.append([]), mul_l.append([]), varl_l.append([])
    iw = np.argmax(ll)
    postB = GetPosteriorB(ll, bConsider)
    if fDebug:
        print('BGP Maximum at b=%.2f' % bConsider[iw], 'CI= [%.2f, %.2f]' %(postB['B_CI'][0], postB['B_CI'][1]))
    assert np.allclose(bConsider[iw], postB['Bmode']), '%s-%s' % str(postB['B_CI'], bConsider[iw])
    return {'loglik': ll, 'Phi': Phi_l[iw], # 'model': m,
            'prediction': {'xtest': ttestl_l[iw], 'mu': mul_l[iw], 'var': varl_l[iw]},
            'hyperparameters': hyps[iw], 'posteriorB': postB}



def GetPosteriorB(objUnsorted, BgridSearch, ciLimits=[0.01, 0.99]):
    '''
    Return posterior on B for each experiment, confidence interval index, map index
    '''
    # for each trueB calculate posterior over grid
    # ... in a numerically stable way
    assert objUnsorted.size == len(BgridSearch), 'size do not match %g-%g' % (objUnsorted.size, len(BgridSearch))
    gr = np.array(BgridSearch)
    isort = np.argsort(gr)
    gr = gr[isort]
    o = objUnsorted[isort].copy()  # sorted objective funtion
    imode = np.argmax(o)
    pn = np.exp(o - np.max(o))
    p = pn/pn.sum()
    assert np.any(~np.isnan(p)), 'Nans in p! %s' % str(p)
    assert np.any(~np.isinf(p)), 'Infinities in p! %s' % str(p)
    pb_cdf = np.cumsum(p)
    confInt = np.zeros(len(ciLimits), dtype=int)
    for pb_i, pb_c in enumerate(ciLimits):
        pb_idx = np.flatnonzero(pb_cdf <= pb_c)
        if(pb_idx.size == 0):
            confInt[pb_i] = 0
        else:
            confInt[pb_i] = np.max(pb_idx)
    # if((imode+5) > 0 and imode < (len(BgridSearch)-5)):  # for modes at end points conf interval checks do not hold
    #     assert confInt[0] <= (imode-1), 'Lower confidence point bigger than mode! (%s)-%g' % (str(confInt), imode)
    #     assert confInt[1] >= (imode+1), 'Upper confidence point bigger than mode! (%s)-%g' % (str(confInt), imode)
    assert np.all(confInt < o.size), confInt
    B_CI = gr[confInt]
    Bmode = gr[imode]
    # return confidence interval as well as mode, and indexes for each
    return {'B_CI': B_CI, 'Bmode': Bmode, 'idx_confInt': confInt, 'idx_mode': imode, 'BgridSearch_sort': gr, 'isort': isort}




def GetInitialConditionsAndPrior(globalBranching, v, infPriorPhi):
    # Setting initial phi
    np.random.seed(42)  # UNDONE remove TODO
    assert isinstance(v, float), 'v should be scalar is %s' % str(type(v))
    N = globalBranching.size
    phiInitial = np.ones((N, 2))*0.5  # don't know anything
    phiInitial[:, 0] = np.random.rand(N)
    phiInitial[:, 1] = 1-phiInitial[:, 0]
    phiPrior = np.ones_like(phiInitial) * 0.5  # don't know anything
    for i in range(N):
        iBranch = globalBranching[i]-2  # is 1,2,3-> -1, 0, 1
        if(iBranch == -1):
            # trunk - set all equal
            phiPrior[i, :] = 0.5
        else:
            if(infPriorPhi):
                phiPrior[i, :] = 1-v
                phiPrior[i, int(iBranch)] = v
            phiInitial[i, int(iBranch)] = 0.5 + (np.random.random() / 2.)  # number between [0.5, 1]
            phiInitial[i, int(iBranch) != np.array([0, 1])] = 1 - phiInitial[i, int(iBranch)]
    assert np.allclose(phiPrior.sum(1), 1), 'Phi Prior should be close to 1 but got %s' % str(phiPrior)
    assert np.allclose(phiInitial.sum(1), 1), 'Phi Initial should be close to 1 but got %s' % str(phiInitial)
    assert np.all(~np.isnan(phiInitial)), 'No nans please!'
    assert np.all(~np.isnan(phiPrior)), 'No nans please!'
    return phiInitial, phiPrior
