# coding: utf-8
import gpflow
import numpy as np
import tensorflow as tf
from . import pZ_construction_singleBP

from gpflow.params import DataHolder
from gpflow.params import Parameter
from gpflow.mean_functions import Zero
from gpflow import settings
from gpflow.decors import params_as_tensors, autoflow
from gpflow.models.model import GPModel

class AssignGP(GPModel):
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

    def __init__(self, t, XExpanded, Y, kern, indices, b, phiPrior=None, phiInitial=None, fDebug=False, KConst=None):
        GPModel.__init__(self, XExpanded, Y, kern,
                                      likelihood=gpflow.likelihoods.Gaussian(),
                                      mean_function=Zero())
        assert len(indices) == t.size, 'indices must be size N'
        assert len(t.shape) == 1, 'pseudotime should be 1D'
        self.N = t.shape[0]
        self.t = t.astype(settings.float_type) # could be DataHolder? advantages
        self.indices = indices
        self.logPhi = Parameter(np.random.randn(t.shape[0], t.shape[0] * 3))  # 1 branch point => 3 functions
        if(phiInitial is None):
            phiInitial = np.ones((self.N, 2))*0.5  # dont know anything
            phiInitial[:, 0] = np.random.rand(self.N)
            phiInitial[:, 1] = 1-phiInitial[:, 0]
        self.fDebug = fDebug
        # Used as p(Z) prior in KL term. This should add to 1 but will do so after UpdatePhPrior
        if(phiPrior is None):
            phiPrior = np.ones((self.N, 2)) * 0.5
        # Fix prior term - this is without trunk
        self.pZ = DataHolder(np.ones((t.shape[0], t.shape[0] * 3)))
        self.UpdateBranchingPoint(b, phiInitial, prior=phiPrior)
        self.KConst = KConst
        if(not fDebug):
            assert KConst is None, 'KConst only for debugging'

    def UpdateBranchingPoint(self, b, phiInitial, prior=None):
        ''' Function to update branching point and optionally reset initial conditions for variational phi'''
        assert isinstance(self.pZ, DataHolder), 'Must have DataHolder'
        assert isinstance(b, np.ndarray)
        assert b.size == 1, 'Must have scalar branching point'
        self.b = b.astype(settings.float_type)  # remember branching value
        assert self.kern.kernels[0].name == 'BranchKernelParam'
        self.kern.kernels[0].Bv = b
        assert isinstance(self.kern.kernels[0].Bv, DataHolder)
        assert self.logPhi.trainable is True, 'Phi should not be constant when changing branching location'
        if prior is not None:
            self.eZ0 = pZ_construction_singleBP.expand_pZ0Zeros(prior)
        self.pZ = pZ_construction_singleBP.expand_pZ0PureNumpyZeros(self.eZ0, b, self.t)
        assert isinstance(self.pZ, DataHolder), 'Must have DataHolder'
        self.InitialiseVariationalPhi(phiInitial)

    def InitialiseVariationalPhi(self, phiInitialIn):
        ''' Set initial state for Phi using branching location to constrain.
        This code has to be consistent with pZ_construction.singleBP.make_matrix to where
        the equality is placed i.e. if x<=b trunk and if x>b branch or vice versa. We use the
         former convention.'''
        assert np.allclose(phiInitialIn.sum(1), 1), 'probs must sum to 1 %s' % str(phiInitialIn)
        assert self.b == self.kern.kernels[0].Bv.value, 'Need to call UpdateBranchingPoint'
        N = self.Y.value.shape[0]
        assert phiInitialIn.shape[0] == N
        assert phiInitialIn.shape[1] == 2  # run OMGP with K=2 trajectories
        phiInitialEx = np.zeros((N, 3 * N))
        # large neg number makes exact zeros, make smaller for added jitter
        phiInitial_invSoftmax = -9. * np.ones((N, 3 * N))
        eps = 1e-9
        iterC = 0
        for i, p in enumerate(self.t):
            if(p > self.b):  # before branching - it's the root
                phiInitialEx[i, iterC:iterC + 3] = np.hstack([eps, phiInitialIn[i, :] - eps])
            else:
                phiInitialEx[i, iterC:iterC + 3] = np.array([1 - 2 * eps, 0 + eps, 0 + eps])
            phiInitial_invSoftmax[i, iterC:iterC + 3] = np.log(phiInitialEx[i, iterC:iterC + 3])
            iterC += 3
        assert not np.any(np.isnan(phiInitialEx)), 'no nans please ' + str(np.nonzero(np.isnan(phiInitialEx)))
        assert not np.any(phiInitialEx < -eps), 'no negatives please ' + str(np.nonzero(np.isnan(phiInitialEx)))
        self.logPhi = phiInitial_invSoftmax

    def GetPhi(self):
        ''' Get Phi matrix, collapsed for each possible entry '''
        assert self.b == self.kern.kernels[0].Bv.value, 'Need to call UpdateBranchingPoint'
        phiExpanded = self.GetPhiExpanded()
        l = [phiExpanded[i, self.indices[i]] for i in range(len(self.indices))]
        phi = np.asarray(l)
        tolError = 1e-6
        assert np.all(phi.sum(1) <= 1+tolError)
        assert np.all(phi >= 0-tolError)
        assert np.all(phi <= 1+tolError)
        return phi

    @autoflow()
    @gpflow.params_as_tensors
    def GetPhiExpanded(self):
        ''' Shortcut function to get Phi matrix out.'''
        return tf.nn.softmax(self.logPhi)

    def objectiveFun(self):
        ''' Objective function to minimize - log likelihood -log prior.
        Unlike _objective, no gradient calculation is performed.'''
        return -self.compute_log_likelihood()-self.compute_log_prior()

    @params_as_tensors
    def _build_likelihood(self):
        print('assignegp_dense compiling model (build_likelihood)')
        N = tf.cast(tf.shape(self.Y)[0], dtype=settings.float_type)
        M = tf.shape(self.X)[0]
        D = tf.cast(tf.shape(self.Y)[1], dtype=settings.float_type)
        if(self.KConst is not None):
            K = tf.cast(self.KConst, settings.float_type)
        else:
            K = self.kern.K(self.X)
        Phi = tf.nn.softmax(self.logPhi)
        # try squashing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6
        sigma2 = self.likelihood.variance
        tau = 1. / self.likelihood.variance
        L = tf.cholesky(K) + tf.eye(M, dtype=settings.float_type) * settings.numerics.jitter_level
        W = tf.transpose(L) * tf.sqrt(tf.reduce_sum(Phi, 0)) / tf.sqrt(sigma2)
        P = tf.matmul(W, tf.transpose(W)) + tf.eye(M, dtype=settings.float_type)
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
        a1 = -0.5 * N * D * tf.log(2. * np.pi / tau)
        a2 = - 0.5 * D * tf.reduce_sum(tf.log(tf.square(tf.diag_part(R))))
        a3 = - 0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        a4 = + 0.5 * tf.reduce_sum(tf.square(c))
        a5 = - KL
        if(self.fDebug):
            a1 = tf.Print(a1, [a1], message='a1=')
            a2 = tf.Print(a2, [a2], message='a2=')
            a3 = tf.Print(a3, [a3], message='a3=')
            a4 = tf.Print(a4, [a4], message='a4=')
            a5 = tf.Print(a5, [a5, Phi], message='a5 and Phi=', summarize=10)
        return a1+a2+a3+a4+a5

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        M = tf.shape(self.X)[0]
        K = self.kern.K(self.X)
        Phi = tf.nn.softmax(self.logPhi)
        # try squashing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6
        sigma2 = self.likelihood.variance
        L = tf.cholesky(K) + tf.eye(M, dtype=settings.float_type) * settings.numerics.jitter_level
        W = tf.transpose(L) * tf.sqrt(tf.reduce_sum(Phi, 0)) / tf.sqrt(sigma2)
        P = tf.matmul(W, tf.transpose(W)) + tf.eye(M, dtype=settings.float_type)
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
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0)\
                - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var

    def build_KL(self, Phi):
        Bv_s = tf.squeeze(self.kern.kernels[0].Bv, squeeze_dims=[1])
        # pZ = pZ_construction_singleBP.make_matrix(self.t, Bv_s, self.phiPrior)
        return tf.reduce_sum(Phi * tf.log(Phi)) - tf.reduce_sum(Phi * tf.log(self.pZ))
