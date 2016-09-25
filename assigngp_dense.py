# coding: utf-8

import GPflow
import numpy as np
import tensorflow as tf
import pZ_construction_singleBP
from matplotlib import pyplot as plt
from GPflow.param import AutoFlow
# TODO S:
# 1) create a parameter for breakpoints (in the kernel perhaps?) - done
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

# PlotSample(D, m.XExpanded[bestAssignment, : ], 3, Y, Bcrap, lw=5., fs=30, mV=mV, figsizeIn=(D*10, D*7), title='Posterior B=%.1f -loglik= %.2f VB= %.2f'%(b, -chainState[-1], VBbound))


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

    def __init__(self, t, XExpanded, Y, kern, indices, phiInitial, b, ZExpanded=None):
        GPflow.model.GPModel.__init__(self, XExpanded, Y, kern,
                                      likelihood=GPflow.likelihoods.Gaussian(),
                                      mean_function=GPflow.mean_functions.Zero())
        assert phiInitial.shape[0] == t.shape[0]
        assert phiInitial.shape[1] == 2  # 1 branching point => 2 OMGP functions
        assert len(indices) == t.size, 'indices must be size N'
        assert len(t.shape) == 1, 'pseudotime should be 1D'
        self.t = t  # could be DataHolder? advantages
        self.ZExpanded = ZExpanded  # inducing points for sparse GP, optional. Same format as XExpanded
        self.indices = indices
        self.phiInitial = phiInitial
        self.logPhi = GPflow.param.Param(np.random.randn(t.shape[0], t.shape[0] * 3))  # 1 branch point => 3 functions
        self.UpdateBranchingPoint(b)

    def UpdateBranchingPoint(self, b):
        ''' Function to update branching point '''
        eps = 1e-12
        assert isinstance(b, np.ndarray)
        self.b = b  # remember branching value
        self.kern.branchkernelparam.Bv = b
        assert isinstance(self.kern.branchkernelparam.Bv, GPflow.param.DataHolder)
        assert b <= (self.t.max()+eps) and b >= (self.t.min() - eps)
        assert self.logPhi.fixed is False, 'Phi should not be constant when changing branching location'
        self.InitialisePhiFromOMGP()

    def InitialisePhiFromOMGP(self):
        ''' Set initial state for Phi using branching location to constrain '''
        assert self.b == self.kern.branchkernelparam.Bv.value, 'Need to call UpdateBranchingPoint'
        N = self.Y.value.shape[0]
        assert self.phiInitial.shape[0] == N
        assert self.phiInitial.shape[1] == 2  # run OMGP with K=2 trajectories
        phiInitial = np.zeros((N, 3 * N))
        # large neg number makes exact zeros, make smaller for added jitter
        phiInitial_invSoftmax = -9. * np.ones((N, 3 * N))
        eps = 1e-12
        iterC = 0
        for i, p in enumerate(self.t):
            if(p < self.b):  # before branching - it's the root
                phiInitial[i, iterC:iterC + 3] = np.array([1 - 2 * eps, 0 + eps, 0 + eps])
            else:
                phiInitial[i, iterC:iterC + 3] = np.hstack([eps, self.phiInitial[i, :] - eps])
            phiInitial_invSoftmax[i, iterC:iterC + 3] = np.log(phiInitial[i, iterC:iterC + 3])
            iterC += 3
        assert np.any(np.isnan(phiInitial)) == False, 'no nans please ' + str(np.nonzero(np.isnan(phiInitial)))
        assert np.any(phiInitial < 0) == False, 'no negatives please ' + str(np.nonzero(np.isnan(phiInitial)))
        self.logPhi = phiInitial_invSoftmax

    def GetPhi(self):
        ''' Get Phi matrix, collapsed for each possible entry '''
        assert self.b == self.kern.branchkernelparam.Bv.value, 'Need to call UpdateBranchingPoint'
        phiExpanded = self.GetPhiExpanded()
        l = [phiExpanded[i, self.indices[i]] for i in range(len(self.indices))]
        phi = np.asarray(l)
        assert np.all(phi.sum(1) < 1)
        assert np.all(phi > 0)
        assert np.all(phi < 1)
        return phi

    @AutoFlow()
    def GetPhiExpanded(self):
        ''' Shortcut function to get Phi matrix out. Could use autoflow?'''
        return tf.nn.softmax(self.logPhi)

    def optimize(self, **kw):
        ''' Catch optimize call to make sure we have correct Phi '''
        print('assigngp_dense intercepting optimize call to check model consistency')
        assert self.b == self.kern.branchkernelparam.Bv.value, 'Need to call UpdateBranchingPoint'
        return GPflow.model.GPModel.optimize(self, **kw)

    def build_likelihood(self):
        print('assignegp_dense compiling model (build_likelihood)')
        N = tf.cast(tf.shape(self.Y)[0], tf.float64)
        M = tf.shape(self.X)[0]
        D = tf.cast(tf.shape(self.Y)[1], tf.float64)
        K = self.kern.K(self.X)
        Phi = tf.nn.softmax(self.logPhi)
        # try sqaushing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6
        tau = 1. / self.likelihood.variance
        L = tf.cholesky(K) + GPflow.tf_hacks.eye(M) * 1e-6
        LTA = tf.transpose(L) * tf.sqrt(tf.reduce_sum(Phi, 0))
        P = tf.matmul(LTA, tf.transpose(LTA)) * tau + GPflow.tf_hacks.eye(M)
        R = tf.cholesky(P)
        PhiY = tf.matmul(tf.transpose(Phi), self.Y)
        LPhiY = tf.matmul(tf.transpose(L), PhiY)
        RiLPhiY = tf.matrix_triangular_solve(R, LPhiY, lower=True)
        # compute KL
        KL = self.build_KL(Phi)
        return -0.5 * N * D * tf.log(2. * np.pi / tau)\
            - 0.5 * D * tf.reduce_sum(tf.log(tf.square(tf.diag_part(R))))\
            - 0.5 * tau * tf.reduce_sum(tf.square(self.Y))\
            + 0.5 * tf.reduce_sum(tf.square(tau * RiLPhiY)) - KL

    def build_KL(self, Phi):
        Bv_s = tf.squeeze(self.kern.branchkernelparam.Bv, squeeze_dims=[1])
        pZ = pZ_construction_singleBP.make_matrix(self.t, Bv_s)
        return tf.reduce_sum(Phi * tf.log(Phi)) - tf.reduce_sum(Phi * tf.log(pZ))

    def build_predict(self, Xnew):
        M = tf.shape(self.X)[0]
        K = self.kern.K(self.X)
        L = tf.cholesky(K)
        tmp = tf.matrix_triangular_solve(L, GPflow.tf_hacks.eye(M), lower=True)
        Ki = tf.matrix_triangular_solve(tf.transpose(L), tmp, lower=False)
        tau = 1. / self.likelihood.variance
        Phi = tf.nn.softmax(self.logPhi)
        # try sqaushing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6
        A = tf.diag(tf.reduce_sum(Phi, 0))
        Lamb = A * tau + Ki  # posterior precision
        R = tf.cholesky(Lamb)
        PhiY = tf.matmul(tf.transpose(Phi), self.Y)
        tmp = tf.matrix_triangular_solve(R, PhiY, lower=True) * tau
        mean_f = tf.matrix_triangular_solve(tf.transpose(R), tmp, lower=False)
        # project onto Xnew
        Kfx = self.kern.K(self.X, Xnew)
        Kxx = self.kern.Kdiag(Xnew)
        A = tf.matrix_triangular_solve(L, Kfx, lower=True)
        B = tf.matrix_triangular_solve(tf.transpose(L), A, lower=False)
        mean = tf.matmul(tf.transpose(B), mean_f)
        var = Kxx - tf.reduce_sum(tf.square(A), 0)
        RiB = tf.matrix_triangular_solve(R, B, lower=True)
        var = var + tf.reduce_sum(RiB, 0)
        return mean, tf.expand_dims(var, 1)
