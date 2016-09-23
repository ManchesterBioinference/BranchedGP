import GPflow
import numpy as np
import time
import pickle as pickle
import assigngp_dense
import branch_kernParamGPflow as bk
import BranchingTree as bt
import pods
import GPyOpt

# Objective function


class objectiveBAndK:

    def __init__(self, Binit):
        mV.kern.branchkernelparam.Bv.fixed = False  # we wont optimize so this is fine
        mV.logPhi.fixed = False  # allocations not fixed for GPyOpt because we update them for each branch point

        mV.likelihood.variance.fixed = False  # all kernel parameters optimised
        mV.kern.branchkernelparam.kern.lengthscales.fixed = False
        mV.kern.branchkernelparam.kern.variance.fixed = False

        # initial branch point
        mV.kern.branchkernelparam.Bv = Binit
        InitialisePhiFromOMGP(mV, phiOMGP=m_phiOMGP, b=Binit)
        # Initialise all model parameters using the OMGP model
        mV.likelihood.variance = m_likvar
        mV.kern.branchkernelparam.kern.lengthscales = m_lenscale
        mV.kern.branchkernelparam.kern.variance = m_var
        mV._compile()

    def f(self, theta):
        # theta is nxp array, return nx1
        n = theta.shape[0]
        VBboundarray = np.ones((n, 1))
        for i in range(n):
            mV.kern.branchkernelparam.Bv = theta[i, 0]
            InitialisePhiFromOMGP(mV, phiOMGP=m_phiOMGP, b=theta[i, 0])
            mV.likelihood.variance = theta[i, 1]
            mV.kern.branchkernelparam.kern.lengthscales = theta[i, 2]
            mV.kern.branchkernelparam.kern.variance = theta[i, 3]

            VBboundarray[i] = -mV.compute_log_likelihood()  # we wish to minimize!
#             print 'objectiveB B=%.0f likvar=%.0f len=%.0f var=%.0f VB=%.0f'%(theta[i,0], theta[i,1], theta[i,2], theta[i,3], VBboundarray[i] )
#             print '================='
        return VBboundarray


def InitialisePhiFromOMGP(mV, phiOMGP, b):
    # branching location needed
    # create index
    N = mV.Y.shape[0]
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

    assert np.any(np.isnan(phiInitial)) == False, 'no nans plaase ' + str(np.nonzero(np.isnan(phiInitial)))
    assert np.any(phiInitial < 0) == False, 'no negatives plaase ' + str(np.nonzero(np.isnan(phiInitial)))
    assert np.any(np.isnan(XExpanded)) == False, 'no nans plaase in XExpanded '

    if(mV is not None):
        mV.logPhi = phiInitial_invSoftmax
    return phiInitial, phiInitial_invSoftmax, XExpanded


def InitModels(pt, XExpanded, Y):
    # code that's a bit crappy - we dont need this
    tree = bt.BinaryBranchingTree(0, 90, fDebug=False)  # set to true to print debug messages
    tree.add(None, 1, 10)  # single branching point
    (fm, _) = tree.GetFunctionBranchTensor()
    KbranchVB = bk.BranchKernelParam(
        GPflow.kernels.RBF(1), fm, BvInitial=np.ones(
            (1, 1))) + GPflow.kernels.White(1)  # other copy of kernel
    KbranchVB.branchkernelparam.Bv.fixed = True
    mV = assigngp_dense.AssignGP(pt, XExpanded, Y, KbranchVB)
    mV.kern.white.variance = 1e-6
    mV.kern.white.variance.fixed = True
    mV._compile()  # creates objective function
    return mV


if __name__ == '__main__':
    print('Load gene expression')
    data = pods.datasets.singlecell()
    genes = data['Y']

    print('Load initial allocation done through OMGP')
    strMAPAllocation = 'InitialAllocationOMGP'
    dictDataMAP = pickle.load(open('modelfiles/' + strMAPAllocation + '.p', "rb"))
    XExpanded = dictDataMAP['XExpanded']
    pt = dictDataMAP['pt']
    m_phiOMGP = dictDataMAP['phiOMGP']
    m_likvar = dictDataMAP['likvar']
    m_lenscale = dictDataMAP['lenscale']
    m_var = dictDataMAP['var']
    l = pt.min() + 1
    u = pt.max() - 1

    interestingGeneList = ['Id2', 'Runx1', 'Sox2', 'Snail', 'Klf2', 'Gata4']

    for g in interestingGeneList:  # genes.columns:
        Y = genes[g].values
        t0 = time.time()
        mV = InitModels(pt, XExpanded, Y[:, None])  # also do gene by gene
        # --- Optimize both B and K
        myobj = objectiveBAndK(np.ones((1, 1)) * (l + u) / 2)  # pass in initial point - start at mid-point
        eps = 1e-6
        bounds = [(l, u), (eps, 3 * Y.var()), (eps, pt.max()), (eps, 3 * Y.var())]  # B, lik var, len, var
        BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f,  # function to optimize
                                                    bounds=bounds)              # normalized y

        max_iter = 200
        n_cores = 4  # multiprocessing.cpu_count()

        BOobj.run_optimization(max_iter,                             # Number of iterations
                               acqu_optimize_method='fast_random',        # method to optimize the acq. function
                               acqu_optimize_restarts=30,
                               batch_method='lp',
                               n_inbatch=n_cores,
                               # size of the collected batches (= number of cores)
                               eps=1e-6)                                # secondary stop criteria (apart from the number of iterations)

        tTime = time.time() - t0
        print('Gene ' + g + ' ' + str(tTime) + 'secs. Solution found by BO xopt=' +
              str(BOobj.x_opt) + ' fxopt=' + str(BOobj.fx_opt))
        saveDict = {'x_opt': BOobj.x_opt, 'fx_opt': BOobj.fx_opt}
        pickle.dump(saveDict, open('modelfiles/gene' + g + '.p', "wb"))
