
# coding: utf-8

# In[1]:

import mouseQPCRModelSelection
import numpy as np
import time
import AssignGPGibbsSingleLoop
import GPflow
import numpy as np
import time
import pickle as pickle
import assigngp_dense
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import branch_kernParamGPflow as bk
import BranchingTree as bt


# In[2]:

subsetSelection = 0


# In[3]:

pt, Yall = mouseQPCRModelSelection.LoadMouseQPCRData(subsetSelection=subsetSelection)
Y = Yall[:, 0][None].T
print('Y')
print(Y.shape)
strExp = 'MouseQPCR_Exp1'
m, _ = mouseQPCRModelSelection.InitModels(pt, Y)


# In[4]:

numMAPsteps = 10
D = Y.shape[1]
print(D)


# In[5]:

from GPclust import OMGP
m = OMGP(pt[:, None], Y, K=2, variance=0.01, prior_Z='DP')  # use a truncated DP with K=2
m.rbf.lengthscale = 65
m.rbf_1.lengthscale = 65
m.optimize(step_length=0.01, maxiter=20)
fig = plt.figure(figsize=(10, 10))
m.plot()


# In[6]:

m.optimize(step_length=0.01, maxiter=200)
m.plot()


# In[7]:

m.plot_probs()


# In[8]:

m.phi.shape


# In[9]:

m


# In[10]:

m.phi.sum(axis=1)  # these are probabilities


# In[11]:

def InitialisePhiFromOMGP(mV, phiOMGP, b=40.):
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

    assert np.any(np.isnan(phiInitial)) == False, 'no nans plaase ' + str(np.nonzero(np.isnan(phiInitial)))
    assert np.any(phiInitial < 0) == False, 'no negatives plaase ' + str(np.nonzero(np.isnan(phiInitial)))
    assert np.any(np.isnan(XExpanded)) == False, 'no nans plaase in XExpanded '

    if(mV is not None):
        mV.logPhi = phiInitial_invSoftmax
    return phiInitial, phiInitial_invSoftmax, XExpanded


# In[12]:

phiInitial, phiInitial_invSoftmax, XExpanded = InitialisePhiFromOMGP(None, phiOMGP=m.phi, b=40.)
fig = plt.figure(figsize=(20, 20))
_ = plt.imshow(phiInitial, cmap='Greys')


# In[13]:

fig = plt.figure(figsize=(20, 20))
_ = plt.imshow(phiInitial_invSoftmax, cmap='Greys')


# # VB Branching code

# In[14]:

def InitModels(pt, XExpanded, Y):
    # code that's a bit crappy - we dont need this
    tree = bt.BinaryBranchingTree(0, 90, fDebug=False)  # set to true to print debug messages
    tree.add(None, 1, 10)  # single branching point
    (fm, _) = tree.GetFunctionBranchTensor()
    KbranchVB = bk.BranchKernelParam(
        GPflow.kernels.RBF(1), fm, BvInitial=np.ones(
            (1, 1))) + GPflow.kernels.White(1)  # other copy of kernel
    KbranchVB.branchkernelparam.Bv.fixed = True
    print('Initialise models: VB =====================')
    mV = assigngp_dense.AssignGP(pt, XExpanded, Y, KbranchVB)
    mV.kern.white.variance = 1e-6
    mV.kern.white.variance.fixed = True

    mV._compile()  # creates objective function

    return mV


# In[15]:

mV = InitModels(pt, XExpanded, Y)  # also do gene by gene


# In[16]:

mV


# In[17]:

m


# In[18]:

# Initialise all model parameters using the OMGP model
# Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
# has one common set.
mV.logPhi = phiInitial_invSoftmax  # initialise allocations from OMGP
mV.likelihood.variance = m.variance.values[0]
# set lengthscale to maximum
mV.kern.branchkernelparam.kern.lengthscales = np.max(np.array([m.rbf.lengthscale.values, m.rbf_1.lengthscale.values]))
# set process variance to average
mV.kern.branchkernelparam.kern.variance = np.mean(np.array([m.rbf.variance.values, m.rbf_1.variance.values]))
mV


# In[19]:

def FlattenPhi(mV):
    # return flattened and rounded Phi i.e. N X 3
    phiFlattened = np.zeros((mV.Y.shape[0], 3))  # only single branching point
    Phi = np.round(np.exp(mV.logPhi._array), decimals=4)
    f = 2  # which function to plot phi against - can be 1,2 or 3
    iterC = 0
    for i, p in enumerate(mV.t):
        phiFlattened[i, :] = Phi[i, iterC:iterC + 3]
        iterC += 3
    return phiFlattened
Phi = FlattenPhi(mV)


# In[20]:

def plotVBCode(mV, figsizeIn=(20, 10), lw=3., fs=10):
    from matplotlib import cm
    D = mV.Y.shape[0]
    fig = plt.figure(figsize=figsizeIn)
    B = mV.kern.branchkernelparam.Bv._array.flatten()
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
        plt.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
        plt.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)

    v = plt.axis()
    plt.plot([B, B], v[-2:], '-m', linewidth=lw)

    # could also plot phi
    Phi = FlattenPhi(mV)
    gp_num = 1  # can be 0,1,2
    plt.scatter(pt, mV.Y[:, d], c=Phi[:, gp_num], vmin=0., vmax=1, s=40)
    plt.colorbar(label='GP {} assignment probability'.format(gp_num))


# In[21]:

l = pt.min() + 1
u = pt.max() - 1
nb = 20
''' Create candidate list for branching values.
Add two extreme values on either end.
    If first selected then we have no branching but two independent trajectory (OMG case).
    If last selected, there is no branching and only a single trajectory (Single).
'''
bs = np.hstack([np.array(l - 10), np.linspace(l, u, nb - 2), np.array(u + 10)])
print(bs)
logVBBound = []

mV.kern.branchkernelparam.Bv.fixed = False  # we wont optimize so this is fine
mV._compile()

for ib, b in enumerate(bs):
    t0 = time.time()
    # Variational bound computation
    mV.kern.branchkernelparam.Bv = np.atleast_2d(b)
    print('Variational kernel branch value ' + str(mV.kern.branchkernelparam.Bv._array.flatten()))

    # need to redo Phi computation so branching point is taken into account
    InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=b)

    # could also optimize
    VBbound = mV.compute_log_likelihood()  # we wish to maximise!
    logVBBound.append(VBbound)
    print('------------> Took %g secs. Bound = %.2f' % (time.time() - t0, VBbound))

    plotVBCode(mV)
    plt.title('B=%.2f VBBound=%.2f' % (b, VBbound))


# In[22]:

mV


# In[23]:

fig = plt.figure(figsize=(10, 10))
plt.plot(bs, logVBBound, '-ob')
maxVB = np.argmax(logVBBound)
plt.plot(bs[maxVB], logVBBound[maxVB], 'rs')
print(str(bs[maxVB]) + ' -> ' + str(logVBBound[maxVB]))


# In[ ]:

# try optimisation
l = pt.min() + 1
u = pt.max() - 1
nb = 20
''' Create candidate list for branching values.
Add two extreme values on either end.
    If first selected then we have no branching but two independent trajectory (OMG case).
    If last selected, there is no branching and only a single trajectory (Single).
'''
bs = np.hstack([np.array(l - 10), np.linspace(l, u, nb - 2), np.array(u + 10)])
print(bs)
logVBBound = []

mV.kern.branchkernelparam.Bv.fixed = True  # we will optimize so this needs to be fixed

for ib, b in enumerate(bs):
    t0 = time.time()
    # Variational bound computation
    mV.kern.branchkernelparam.Bv = np.atleast_2d(b)
    print('Variational kernel branch value ' + str(mV.kern.branchkernelparam.Bv._array.flatten()))

    # Initialise all model parameters using the OMGP model
    phiInitial, phiInitial_invSoftmax, XExpanded = InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=b)
    mV.logPhi = phiInitial_invSoftmax  # initialise allocations from OMGP
    mV.likelihood.variance = m.variance.values[0]
    mV.kern.branchkernelparam.kern.lengthscales = np.max(
        np.array([m.rbf.lengthscale.values, m.rbf_1.lengthscale.values]))
    mV.kern.branchkernelparam.kern.variance = np.mean(np.array([m.rbf.variance.values, m.rbf_1.variance.values]))

    # optimize
    mV.optimize()  # should recompile due to updated branch point

    VBbound = mV.compute_log_likelihood()  # we wish to maximise!
    logVBBound.append(VBbound)
    print('------------> Took %g secs. Bound = %.2f' % (time.time() - t0, VBbound))

    plotVBCode(mV)
    plt.title('B=%.2f VBBound=%.2f' % (b, VBbound))


fig = plt.figure(figsize=(10, 10))
plt.plot(bs, logVBBound, '-ob')


# # Bayesian optimisation
# 1. Use GPyOpt to learn branching point and kernel hyperparameters.
# 1. set fixed=False for all parameters except for Phi.fixed=True
# 1. It's still beneficial to use *VB code* rather than *Jings model* since we integrate out (approximately using VB bound) uncertainty in allocation (Phi).
# 1. Store all intermedite values visited by GPyOpt?
# 1. Use Matern 3/2 or 5/2 for both OMGP and our model. Actually different kernels for OMGP and our can make sense as outputs different (potentially)?
#

# In[24]:

import GPyOpt


# In[39]:

# Objective function
class objectiveB:

    def __init__(self, Binit):
        mV.kern.branchkernelparam.Bv.fixed = False  # we wont optimize so this is fine
        mV.logPhi.fixed = False  # allocations not fixed for GPyOpt because we update them for each branch point

        mV.likelihood.variance.fixed = True  # no kernel parameters optimised
        mV.kern.branchkernelparam.kern.lengthscales.fixed = True
        mV.kern.branchkernelparam.kern.variance.fixed = True

        # initial branch point
        mV.kern.branchkernelparam.Bv = Binit
        InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=Binit)
        # Initialise all model parameters using the OMGP model
        mV.likelihood.variance = m.variance.values[0]
        mV.kern.branchkernelparam.kern.lengthscales = np.max(
            np.array([m.rbf.lengthscale.values, m.rbf_1.lengthscale.values]))
        mV.kern.branchkernelparam.kern.variance = np.mean(np.array([m.rbf.variance.values, m.rbf_1.variance.values]))
        mV._compile()

    def f(self, theta):
        # theta is nxp array, return nx1
        n = theta.shape[0]
        VBboundarray = np.ones((n, 1))
        for i in range(n):
            mV.kern.branchkernelparam.Bv = theta[i, 0]
            InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=theta[i, 0])
            VBboundarray[i] = -mV.compute_log_likelihood()  # we wish to minimize!
            print('objectiveB B=' + str(theta[i, 0]) + ' -> ' + str(VBboundarray[i]))
        return VBboundarray


# In[51]:

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
        InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=Binit)
        # Initialise all model parameters using the OMGP model
        mV.logPhi = phiInitial_invSoftmax  # initialise allocations from OMGP
        mV.likelihood.variance = m.variance.values[0]
        mV.kern.branchkernelparam.kern.lengthscales = np.max(
            np.array([m.rbf.lengthscale.values, m.rbf_1.lengthscale.values]))
        mV.kern.branchkernelparam.kern.variance = np.mean(np.array([m.rbf.variance.values, m.rbf_1.variance.values]))
        mV._compile()

    def f(self, theta):
        # theta is nxp array, return nx1
        n = theta.shape[0]
        VBboundarray = np.ones((n, 1))
        for i in range(n):
            mV.kern.branchkernelparam.Bv = theta[i, 0]
            InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=theta[i, 0])
            mV.likelihood.variance = theta[i, 1]
            mV.kern.branchkernelparam.kern.lengthscales = theta[i, 2]
            mV.kern.branchkernelparam.kern.variance = theta[i, 3]

            VBboundarray[i] = -mV.compute_log_likelihood()  # we wish to minimize!
            print(
                'objectiveB B=%.0f likvar=%.0f len=%.0f var=%.0f VB=%.0f' %
                (theta[
                    i, 0], theta[
                    i, 1], theta[
                    i, 2], theta[
                    i, 3], VBboundarray[i]))
        return VBboundarray


# In[26]:

# try optimisation
l = pt.min() + 1
u = pt.max() - 1
# We need constraints on there parameters
# B = [l,u]
# lik.variance > 0
# kern.lengthscale, variance > 0


# In[44]:

# --- Optimize B
myobj = objectiveB(np.ones((1, 1)) * (l + u) / 2)  # pass in initial point - start at mid-point
bounds = [(l, u)]
# BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f,  # function to optimize
#                                             bounds=bounds)              # normalized y
BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f,  # function to optimize
                                            bounds=bounds,                     # box-constrains of the problem
                                            acquisition='EI',                 # Selects the Expected improvement
                                            acquisition_par=0,                 # parameter of the acquisition function
                                            normalize=False)                    # Normalize the acquisition function

t0 = time.time()
max_iter = 20
n_cores = 4

# BOobj.run_optimization(max_iter,                             # Number of iterations
#                         n_inbatch = n_cores,                        # size of the collected batches (= number of cores)
# eps = 1e-6)                                # secondary stop criteria
# (apart from the number of iterations)
BOobj.run_optimization(max_iter,                             # Number of iterations
                       acqu_optimize_method='fast_random',       # method to optimize the acq. function
                       n_inbatch=n_cores,                        # size of the collected batches (= number of cores)
                       # method to collected the batches (maximization-penalization)
                       batch_method='lp',
                       acqu_optimize_restarts=30,                # number of local optimizers
                       eps=1e-6)                                # secondary stop criteria (apart from the number of iterations)

print('GPyOpt took %g secs ' % (time.time() - t0))


# In[53]:

# --- Optimize both B and K
myobj = objectiveBAndK(np.ones((1, 1)) * (l + u) / 2)  # pass in initial point - start at mid-point
eps = 1e-6
bounds = [(l, u), (eps, 3 * Y.var()), (eps, pt.max()), (eps, 3 * Y.var())]  # B, lik var, len, var

BOobj = GPyOpt.methods.BayesianOptimization(f=myobj.f,  # function to optimize
                                            bounds=bounds)              # normalized y
t0 = time.time()
max_iter = 20
import multiprocessing
n_cores = multiprocessing.cpu_count()

BOobj.run_optimization(max_iter,                             # Number of iterations
                       acqu_optimize_method='fast_random',        # method to optimize the acq. function
                       acqu_optimize_restarts=30,
                       batch_method='lp',
                       n_inbatch=n_cores,                        # size of the collected batches (= number of cores)
                       eps=1e-6)                                # secondary stop criteria (apart from the number of iterations)

print('GPyOpt took %g secs ' % (time.time() - t0))


# In[54]:

BOobj.plot_acquisition()
BOobj.plot_convergence()


# In[42]:

print('Solution found by BO')
print(BOobj.x_opt)
print(BOobj.fx_opt)
print('Solution found by grid search on B only')
print(str(bs[maxVB]) + ' -> ' + str(-logVBBound[maxVB]))

# Can explicitly check computation using myobj.f(np.ones((1,1))*16.834)
# should add assert that Phi before branching point is 1 everywhere!


# In[59]:

# plot best solution
mV.kern.branchkernelparam.Bv = BOobj.x_opt[0]
InitialisePhiFromOMGP(mV, phiOMGP=m.phi, b=BOobj.x_opt[0])
mV.likelihood.variance = BOobj.x_opt[1]
mV.kern.branchkernelparam.kern.lengthscales = BOobj.x_opt[2]
mV.kern.branchkernelparam.kern.variance = BOobj.x_opt[3]
mV


# In[60]:

print('Bound got %.2f should be %.2f' % (-mV.compute_log_likelihood(), BOobj.fx_opt))
plotVBCode(mV)


# In[62]:


# In[ ]:
