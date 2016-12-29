# coding: utf-8

import GPflow
import numpy as np
import tensorflow as tf
from . import pZ_construction_singleBP
from matplotlib import pyplot as plt
from GPflow.param import AutoFlow
from GPflow.param import DataHolder

# TODO S:
# 2) tidy up make_pZ_matrix and generalize to multiple latent functions


def PlotSample(D, X, M, samples, B=None, lw=3.,
               fs=10, figsizeIn=(12, 16), title=None, mV=None):
    f, ax = plt.subplots(D, 1, figsize=figsizeIn, sharex=True, sharey=True)
    nb = len(B)  # number of branch points
    for d in range(D):
        for i in range(1, M + 1):
            t = X[X[:, 1] == i, 0]
            y = samples[X[:, 1] == i, d]
            if(t.size == 0):
                continue
            if(D != 1):
                p = ax.flatten()[d]
            else:
                p = ax

            p.plot(t, y, '.', label=i, markersize=2 * lw)
            p.text(t[t.size / 2], y[t.size / 2], str(i), fontsize=fs)
        # Add vertical lines for branch points
        if(title is not None):
            p.set_title(title + ' Dim=' + str(d), fontsize=fs)

        if(B is not None):
            v = p.axis()
            for i in range(nb):
                p.plot([B[i], B[i]], v[-2:], '--r')
        if(mV is not None):
            assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)
            print('plotting mv')
            pt = mV.t
            l = np.min(pt)
            u = np.max(pt)
            for f in range(1, 4):
                if(f == 1):
                    ttest = np.linspace(l, B.flatten(), 100)[:, None]  # root
                else:
                    ttest = np.linspace(B.flatten(), u, 100)[:, None]
                Xtest = np.hstack((ttest, ttest * 0 + f))
                mu, var = mV.predict_f(Xtest)
                assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
                assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
                mean, = p.plot(ttest, mu[:, d], linewidth=lw)
                col = mean.get_color()
                # print 'd='+str(d)+ ' f='+str(f) + '================'
                # variance is common for all outputs!
                p.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
                p.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# PlotSample(D, m.XExpanded[bestAssignment, : ], 3, Y, Bcrap, lw=5., fs=30, mV=mV, figsizeIn=(D*10, D*7),
# title='Posterior B=%.1f -loglik= %.2f VB= %.2f'%(b, -chainState[-1], VBbound))


def plotPosterior(pt, Bv, mV, figsizeIn=(12, 16)):
    l = np.min(pt)
    u = np.max(pt)
    D = mV.Y.shape
    f, ax = plt.subplots(D, 1, figsize=figsizeIn, sharex=True, sharey=True)

    for f in range(1, 4):
        # fig = plt.figure(figsize=(12, 8))
        if(f == 1):
            ttest = np.linspace(l, Bv, 100)[:, None]  # root
        else:
            ttest = np.linspace(Bv, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest * 0 + f))
        mu, var = mV.predict_f(Xtest)
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
        mean, = plt.plot(ttest, mu)
        col = mean.get_color()
        plt.plot(ttest, mu + 2 * np.sqrt(var), '--', color=col)
        plt.plot(ttest, mu - 2 * np.sqrt(var), '--', color=col)


class AssignGP(GPflow.model.GPModel):
    """
    Gaussian Process regression, but where the index to which the data are
    assigned is unknown.

    let f be a vector of GP points (usually longer than the number of data)

        f ~ MVN(0, K)

    and let Z be an (unknown) binary matrix with a single 1 in each row. The
    likelihood is

       y ~ MVN( Z f, \sigma^2 I)

    That is, each element of y is a noisy realization of one (unknown) element
    of f. We use variational Bayes to infer the labels using a sparse prior
    over the Z matrix (i.e. we have narrowed down the choice of which function
    values each y is drawn from).

    """

    def __init__(self, t, XExpanded, Y, kern, indices, b, phiPrior=None, phiInitial=None, fDebug=False):
        GPflow.model.GPModel.__init__(self, XExpanded, Y, kern,
                                      likelihood=GPflow.likelihoods.Gaussian(),
                                      mean_function=GPflow.mean_functions.Zero())
        assert len(indices) == t.size, 'indices must be size N'
        assert len(t.shape) == 1, 'pseudotime should be 1D'
        self.t = t  # could be DataHolder? advantages
        self.indices = indices
        self.logPhi = GPflow.param.Param(np.random.randn(t.shape[0], t.shape[0] * 3))  # 1 branch point => 3 functions
        if(phiInitial is None):
            phiInitial = np.ones((t.size, 2))*0.5  # dont know anything
            phiInitial[:, 0] = np.random.rand(t.size)
            phiInitial[:, 1] = 1-phiInitial[:, 0]
        self.UpdateBranchingPoint(b, phiInitial)
        self.fDebug = fDebug
        # Used as p(Z) prior in KL term. This should add to 1 but will do so after UpdatePhPrior
        self.phiPrior = DataHolder(np.ones((t.shape[0], t.shape[0] * 3)))
        if(phiPrior is None):
            phiPrior = np.ones((self.t.shape[0], 2)) * 0.5
        self.UpdatePhiPrior(phiPrior)

    def UpdatePhiPrior(self, pZ0):
        ''' Update prior on allocations p(Z) used in KL term '''
        assert pZ0.shape[0] == self.t.shape[0]
        assert pZ0.shape[1] == 2  # 1 branching point => 2 functions
        eZ0 = pZ_construction_singleBP.expand_pZ0(pZ0)
        self.phiPrior = eZ0
        assert isinstance(self.phiPrior, GPflow.param.DataHolder), 'Must have DataHolder'

    def UpdateBranchingPoint(self, b, phiInitial):
        ''' Function to update branching point and optionally reset initial conditions for variational phi'''
        eps = 1e-9
        assert isinstance(b, np.ndarray)
        assert b.size == 1, 'Must have scalar branching point'
        self.b = b  # remember branching value
        self.kern.branchkernelparam.Bv = b
        assert isinstance(self.kern.branchkernelparam.Bv, GPflow.param.DataHolder)
        # and b <= (self.t.max()+eps)
        assert b >= (self.t.min() - eps), 'Branching suspicious b=%f is not in [%f, %f] ' % (b, self.t.min(), self.t.max())
        assert self.logPhi.fixed is False, 'Phi should not be constant when changing branching location'
        self.InitialiseVariationalPhi(phiInitial)

    def InitialiseVariationalPhi(self, phiInitialIn):
        ''' Set initial state for Phi using branching location to constrain '''
        assert np.allclose(phiInitialIn.sum(1), 1), 'probs must sum to 1 %s' % str(phiInitialIn)
        assert self.b == self.kern.branchkernelparam.Bv.value, 'Need to call UpdateBranchingPoint'
        N = self.Y.value.shape[0]
        assert phiInitialIn.shape[0] == N
        assert phiInitialIn.shape[1] == 2  # run OMGP with K=2 trajectories
        phiInitialEx = np.zeros((N, 3 * N))
        # large neg number makes exact zeros, make smaller for added jitter
        phiInitial_invSoftmax = -9. * np.ones((N, 3 * N))
        eps = 1e-9
        iterC = 0
        for i, p in enumerate(self.t):
            if(p < self.b):  # before branching - it's the root
                phiInitialEx[i, iterC:iterC + 3] = np.array([1 - 2 * eps, 0 + eps, 0 + eps])
            else:
                phiInitialEx[i, iterC:iterC + 3] = np.hstack([eps, phiInitialIn[i, :] - eps])
            phiInitial_invSoftmax[i, iterC:iterC + 3] = np.log(phiInitialEx[i, iterC:iterC + 3])
            iterC += 3
        assert not np.any(np.isnan(phiInitialEx)), 'no nans please ' + str(np.nonzero(np.isnan(phiInitialEx)))
        assert not np.any(phiInitialEx < -eps), 'no negatives please ' + str(np.nonzero(np.isnan(phiInitialEx)))
        self.logPhi = phiInitial_invSoftmax

    def GetPhi(self):
        ''' Get Phi matrix, collapsed for each possible entry '''
        assert self.b == self.kern.branchkernelparam.Bv.value, 'Need to call UpdateBranchingPoint'
        phiExpanded = self.GetPhiExpanded()
        l = [phiExpanded[i, self.indices[i]] for i in range(len(self.indices))]
        phi = np.asarray(l)
        tolError = 1e-6
        assert np.all(phi.sum(1) <= 1+tolError)
        assert np.all(phi >= 0-tolError)
        assert np.all(phi <= 1+tolError)
        return phi

    @AutoFlow()
    def GetPhiExpanded(self):
        ''' Shortcut function to get Phi matrix out.'''
        return tf.nn.softmax(self.logPhi)

    def optimize(self, **kw):
        ''' Catch optimize call to make sure we have correct Phi '''
        if(self.fDebug):
            print('assigngp_dense intercepting optimize call to check model consistency')
        assert self.b == self.kern.branchkernelparam.Bv.value, 'Need to call UpdateBranchingPoint'
        return GPflow.model.GPModel.optimize(self, **kw)

    def objectiveFun(self):
        ''' Objective function to minimize - log likelihood -log prior.
        Unlike _objective, no gradient calculation is performed.'''
        return -self.compute_log_likelihood()-self.compute_log_prior()

    def build_likelihood(self):
        print('assignegp_dense compiling model (build_likelihood)')
        N = tf.cast(tf.shape(self.Y)[0], tf.float64)
        M = tf.shape(self.X)[0]
        D = tf.cast(tf.shape(self.Y)[1], tf.float64)
        K = self.kern.K(self.X)
        Phi = tf.nn.softmax(self.logPhi)
        # try squashing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6
        sigma2 = self.likelihood.variance
        tau = 1. / self.likelihood.variance
        L = tf.cholesky(K) + GPflow.tf_wraps.eye(M) * 1e-6
        W = tf.transpose(L) * tf.sqrt(tf.reduce_sum(Phi, 0)) / tf.sqrt(sigma2)
        P = tf.matmul(W, tf.transpose(W)) + GPflow.tf_wraps.eye(M)
        R = tf.cholesky(P)
        PhiY = tf.matmul(tf.transpose(Phi), self.Y)
        LPhiY = tf.matmul(tf.transpose(L), PhiY)
        if(self.fDebug):
            Phi = tf.Print(Phi, [tf.shape(P), P], message='P=', name='P', summarize=10)
            Phi = tf.Print(Phi, [tf.shape(LPhiY), LPhiY], message='LPhiY=', name='LPhiY', summarize=10)
            Phi = tf.Print(Phi, [tf.shape(K), K], message='K=', name='K', summarize=10)
            Phi = tf.Print(Phi, [tau], message='tau=', name='tau', summarize=10)
        c = tf.matrix_triangular_solve(R, LPhiY, lower=True) / sigma2
        # compute KL
        KL = self.build_KL(Phi)
        return -0.5 * N * D * tf.log(2. * np.pi / tau)\
            - 0.5 * D * tf.reduce_sum(tf.log(tf.square(tf.diag_part(R))))\
            - 0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2\
            + 0.5 * tf.reduce_sum(tf.square(c)) - KL

    def build_predict(self, Xnew, full_cov=False):
        M = tf.shape(self.X)[0]
        K = self.kern.K(self.X)
        Phi = tf.nn.softmax(self.logPhi)
        # try squashing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6
        sigma2 = self.likelihood.variance
        L = tf.cholesky(K) + GPflow.tf_wraps.eye(M) * 1e-6
        W = tf.transpose(L) * tf.sqrt(tf.reduce_sum(Phi, 0)) / tf.sqrt(sigma2)
        P = tf.matmul(W, tf.transpose(W)) + GPflow.tf_wraps.eye(M)
        R = tf.cholesky(P)
        PhiY = tf.matmul(tf.transpose(Phi), self.Y)
        LPhiY = tf.matmul(tf.transpose(L), PhiY)
        c = tf.matrix_triangular_solve(R, LPhiY, lower=True) / sigma2
        Kus = self.kern.K(self.X, Xnew)
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(R, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tf.transpose(tmp2), tmp2)\
                - tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0)\
                - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.pack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var

    def build_KL(self, Phi):
        Bv_s = tf.squeeze(self.kern.branchkernelparam.Bv, squeeze_dims=[1])
        pZ = pZ_construction_singleBP.make_matrix(self.t, Bv_s, self.phiPrior)
        return tf.reduce_sum(Phi * tf.log(Phi)) - tf.reduce_sum(Phi * tf.log(pZ))
