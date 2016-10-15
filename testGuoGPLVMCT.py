from sklearn import preprocessing
import GPflow
import numpy as np
import time


def guoGPLVM_CT(Y, cptPrior, fSphere=True, fFixedZ=True, Q=1,
                priorStd=0.5, xvar=.1, M=40, fInitialiseRandom=False, fMatern=True):
    N, G = Y.shape
    assert Q == 1 or Q == 2, 'Must use 1 or 2 latent dimension'
    assert G == 48, 'In Guo we should have 48 genes'
    np.random.seed(10)
    prior_mean = np.zeros((N, Q))
    prior_mean[:, 0] = cptPrior
    prior_std = np.ones((N, Q))
    prior_std[:, 0] = priorStd
    Xinit = np.zeros((N, Q))
    if(Q == 2):
        Xinit[:, 1] = np.random.rand(N)

    if(fSphere):
        Yt = preprocessing.scale(Y.values)
    else:
        Yt = Y.values
    # Idea 1: 0 mean, 1 var all outputs
    # Idea 2: multiple random restarts, evaluate likelihood for random samples from prior
    # Idea 3: GPyOpt for kernel parameters
    # Idea 4: Use multiple nuggets
    # Idea 5: Use multiple proc variances (requires BGPLVM extension)
    ''' Bayesian GPLVM with prior '''
    for i in range(N):
        Xinit[i, 0] = prior_mean[i, 0] + prior_std[i, 0]*np.random.randn(1)
    X_variance = xvar * np.ones((N, Q))  # np.random.uniform(0, .1, Xinit.shape)
    Z = np.random.permutation(Xinit.copy())[:M]
    if(not fInitialiseRandom):
        Z[:, 0] = np.linspace(prior_mean[:, 0].min() - 7*priorStd,  prior_mean[:, 0].max() + 7*priorStd, M)

    if(fMatern):
        if(Q == 1):
            k = GPflow.kernels.Matern32(Q)
        else:
            k = GPflow.kernels.Matern32(1, active_dims=[0]) + GPflow.kernels.RBF(1, active_dims=[1])
    else:
        if(Q == 1):
            k = GPflow.kernels.RBF(Q)
        else:
            # additive kernel
            k = GPflow.kernels.RBF(1, active_dims=[0]) + GPflow.kernels.RBF(1, active_dims=[1])

    m = GPflow.gplvm.BayesianGPLVM(M=M, X_mean=Xinit.copy(), X_var=X_variance.copy(),
                                   Y=Yt, kern=k, Z=Z.copy(),
                                   X_prior_mean=prior_mean, X_prior_var=np.square(prior_std))
    t0 = time.time()
    if(fFixedZ):
        m.Z.fixed = True
    if(Q == 1):
        m.kern.lengthscales = 1.5  # Delorean
        m.kern.lengthscales.fixed = True
    else:
        m.kern.kern_list[0].lengthscales = 1.5
        m.kern.kern_list[0].lengthscales.fixed = True
    m.likelihood.variance = 0.01
    m.likelihood.variance.fixed = True
    _ = m.optimize(maxiter=100, disp=0)
    m.likelihood.variance.fixed = False
    if(Q == 1):
        m.kern.lengthscales.fixed = False
    else:
        pass
#         m.kern.kern_list[0].lengthscales.fixed = False
    # Now do full optimisation
    _ = m.optimize(maxiter=400, disp=0)
    elapsedTime = time.time() - t0
    return m.X_mean.value[:, 0], elapsedTime, m
