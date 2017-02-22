import numpy as np
import pickle
import GPflow
import os
import sys
import traceback
# Branching model
from . import VBHelperFunctions
from . import BranchingTree as bt
from . import branch_kernParamGPflow as bk
from . import assigngp_denseSparse
from . import assigngp_dense

def ensure_dir(f):
    ''' from http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary '''
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


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


def MultipleRestartsChoose(strsave, m, v, globalBranching, infPriorPhi, b, maxiter, fFixKernel=True,
                           kerlen=None, kervar=None, noise=None, fDebug=False):
    ''' Do multiple random restarts as per v list and kernlen list '''
    assert isinstance(b, float), 'branching should be scalar is %s' % str(type(b))
    if (fFixKernel is False):
        assert kerlen is not None and len(kerlen) > 0
        assert kervar is not None and len(kervar) > 0
        assert noise is not None and len(noise) > 0
        assert m.kern.branchkernelparam.kern.lengthscales.fixed is False
        assert m.kern.branchkernelparam.kern.variance.fixed is False
        assert m.likelihood.variance.fixed is False
        h = [{'l': l, 'kv': kv, 'n': n, 'v': vinitial} for l in kerlen for kv in kervar for n in noise for vinitial in v]
    else:
        assert m.kern.branchkernelparam.kern.lengthscales.fixed
        assert m.kern.branchkernelparam.kern.variance.fixed
        assert m.likelihood.variance.fixed
        h = [{'v': vinitial} for vinitial in v]
    ll = np.zeros(len(h), dtype=float)
    ll[:] = -np.inf
    params = np.zeros(len(h), dtype=object)
    for ih, hc in enumerate(h):
        # set the parameters for initial value - could be Phi and theta or just Phi
        assert hc['v'] >= 0 and hc['v'] <= 1, 'bad v {:f}'.format(hc['v'])
        phiInitial, _ = GetInitialConditionsAndPrior(globalBranching, hc['v'], infPriorPhi)
        m.UpdateBranchingPoint(np.ones((1, 1))*b, phiInitial)
        if(fFixKernel is False):
            m.kern.branchkernelparam.kern.lengthscales = hc['l']
            m.kern.branchkernelparam.kern.variance = hc['kv']
            m.likelihood.variance = hc['n']
        if (fDebug):
            print('MultipleRestartsChoose:: prior {}'.format(infPriorPhi), 'len', kerlen, 'var', kervar, 'noise', noise,
                  'B=', b, 'v', v, 'maxiter', maxiter, 'fFixKernel', fFixKernel)
            print('model', m)
            print('Phi', m.GetPhi())
            print('Initial Phi', phiInitial)
        try:
            m.optimize(disp=fDebug, maxiter=maxiter)
            ll[ih] = m.compute_log_likelihood()
            params[ih] = m.get_free_state()
        except:
            print(strsave, 'Params', hc, "Unexpected error:", sys.exc_info()[0])
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            continue
    # pick winner
    iw = np.argmax(ll)
    assert not np.isinf(ll[iw]), 'MultipleRestartsChoose: ' + str(ll)
    assert iw.size == 1
    if (fFixKernel is False):
        # Only set parameters having optimized hyperparameters
        m.set_state(params[iw])
        strHyp = 'prior {}, n:{:.2f} v:{:.2f} l:{:.2f}'.format(infPriorPhi, m.likelihood.variance.value[0],
                                                               m.kern.branchkernelparam.kern.variance.value[0],
                                                               m.kern.branchkernelparam.kern.lengthscales.value[0])
        print('Optimized model:', len(h), 'multiple restarts', 'fFixKernel', fFixKernel, 'winner', strHyp)
    return ll[iw]


def EstimateBranchModel(strsave, gUse, globalBranching, GPt, GPy, BgridSearchIn=[0.4, 0.7],
                        fSavefile=True, M=20, maxiter=10, infPriorPhi=True, v=[0.95],
                        kervarIn=[0.1, 1, 6], kerlenIn=[0.1, 1, 6], noiseInSamplesIn=[0.01, 0.1, 1],
                        fFixhyperpar=False, fDebug=False):
    ''' Function to analyse specific genes using Branching GP
    Will perform a model hyperparameter estimation at Monocle branching point (0.1).
    Then estimate the model without hyperparemeter estimation at BgridSearchIn locations.
    noiseInSamplesIn initial value for likelihood variance
    M number of inducing points
    maxiter maximum number of optimisation iterations
    return dictionary with model information and model.
    prior is v[0]
    '''
    assert GPt.ndim == 1, 'GPt should be 1-D got %s' % str(GPt.shape)
    assert GPy.ndim == 2, 'GPy should be 2-D got %s' % str(GPy.shape)
    if(fFixhyperpar):
        assert len(kervarIn) == 1
        assert len(kerlenIn) == 1
        assert len(noiseInSamplesIn) == 1
    assert isinstance(v, list), 'v must be list got %s' % str(v)
    ptb = np.min([np.min(GPt[globalBranching == 2]), np.min(GPt[globalBranching == 3])])
    print('Estimating branching model at Monocle2 branching point %.2f' % ptb)
    # Create tree structures
    tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
    tree.add(None, 1, np.ones((1, 1))*ptb)  # B can be anything here
    (fm, _) = tree.GetFunctionBranchTensor()
    # Branching kernel
    kb = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, b=np.zeros((1, 1))) + GPflow.kernels.White(1)
    kb.white.variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
    kb.white.variance.fixed = True  # jitter for numerics
    # Getting functions list
    XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(GPt)
    assert GPt.dtype == np.float
    assert XExpanded.dtype == np.float
    print('Training data is', 'GPt', GPt.shape, 'y', GPy.shape, 'Expanded X', XExpanded.shape)

    # set prior and initial conditions
    phiInitial, phiPrior = GetInitialConditionsAndPrior(globalBranching, v[0], infPriorPhi)
    assert phiInitial.shape[0] == GPt.size
    # Construct model
    if(M == 0):
        m = assigngp_dense.AssignGP(GPt, XExpanded, GPy, kb, indices,
                                    np.ones((1, 1))*ptb, phiInitial=phiInitial, phiPrior=phiPrior)
    else:
        # ir = np.random.choice(XExpanded.shape[0], M)
        # ZExpanded = XExpanded[ir, :]
        ZExpanded = np.ones((M, 2))
        ZExpanded[:, 0] = np.linspace(0, 1, M, endpoint=False)
        ZExpanded[:, 1] = np.array([i for j in range(M) for i in range(1,4)])[:M]
        m = assigngp_denseSparse.AssignGPSparse(GPt, XExpanded, GPy, kb, indices,
                                                np.ones((1, 1))*ptb, ZExpanded, phiInitial=phiInitial, phiPrior=phiPrior)

    if(fFixhyperpar):
        assert len(kervarIn) == 1
        assert len(noiseInSamplesIn) == 1
        assert len(kerlenIn) == 1
        m.likelihood.variance = noiseInSamplesIn[0]
        m.kern.branchkernelparam.kern.lengthscales = kerlenIn[0]
        m.kern.branchkernelparam.kern.variance = kervarIn[0]
        m.likelihood.variance.fixed = True
        m.kern.branchkernelparam.kern.lengthscales.fixed = True
        m.kern.branchkernelparam.kern.variance.fixed = True
    else:
        m.likelihood.variance.fixed = False
        m.kern.branchkernelparam.kern.lengthscales.fixed = False
        m.kern.branchkernelparam.kern.variance.fixed = False
    BgridSearch = [ptb] + BgridSearchIn + [1.1]
    timingInfo = np.zeros(len(BgridSearch))
    obj = np.zeros(len(BgridSearch))
    mlocallist = list()
    # ########################### Grid search ############
    for ib, b in enumerate(BgridSearch):
        if(ib == 0):
            fFixKernel = fFixhyperpar  # for ptb point we can estimate hyperparameters
        # if code below fails - just throw away entire run
        obj[ib] = MultipleRestartsChoose(strsave, m, v, globalBranching, infPriorPhi, b, maxiter, fFixKernel=fFixKernel,
                                         kerlen=kerlenIn, kervar=kervarIn, noise=noiseInSamplesIn, fDebug=fDebug)
        if (ib == 0 and not fFixKernel):
            # for ptb point redo optimisation with fixed hyperparameters
            # Only set these once to avoid recompilation
            print('Fixing kernel hyperparameters.')
            fFixKernel = True  # Use previous estimates
            m.kern.branchkernelparam.kern.variance.fixed = True
            m.kern.branchkernelparam.kern.lengthscales.fixed = True
            m.likelihood.variance.fixed = True
            # Run optimisation again only if hyperparameters were not fixed
            obj[ib] = MultipleRestartsChoose(strsave, m, v, globalBranching, infPriorPhi, b, maxiter,
                                             fFixKernel=fFixKernel,
                                             kerlen=kerlenIn, kervar=kervarIn, noise=noiseInSamplesIn, fDebug=fDebug)
        # do prediction and save results
        Phi = m.GetPhi()
        ttestl, mul, varl = VBHelperFunctions.predictBranchingModel(m)
        # do not save model as this will break between GPflow versions
        mlocallist.append({'candidateB': b, 'obj': obj[ib], 'Phi': Phi,
                           'ttestl': ttestl, 'mul': mul, 'varl': varl})
    # do not save model as this will break between GPflow versions
    retObj = {'mlocallist': mlocallist,
              'obj': obj, 'timingInfo': timingInfo, 'BgridSearch': BgridSearch,
              'XExpanded': XExpanded,
              'GPt': GPt,  'GPy': GPy, 'M': M,
              'maxiter': maxiter, 'gUse': gUse,
              'globalBranching': globalBranching,
              'v': v, 'kerlenIn': kerlenIn, 'kervarIn': kervarIn, 'noiseInSamplesIn': noiseInSamplesIn,
              'likvar': m.likelihood.variance.value, 'kerlen': m.kern.branchkernelparam.kern.lengthscales.value,
              'kervar': m.kern.branchkernelparam.kern.variance.value}  # 'mGPR': mi.get_parameter_dict()
    if(fSavefile):
        ensure_dir(strsave+'/')
        print('Saving file', strsave+'/'+gUse+'.p')
        pickle.dump(retObj, open(strsave+'/'+gUse+'.p', "wb"))
    return retObj, m  # return model in case caller wants to do stuff with it
