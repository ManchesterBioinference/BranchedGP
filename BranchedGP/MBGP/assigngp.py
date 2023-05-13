# coding: utf-8
from enum import IntEnum
from typing import Optional, Sequence

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, default_jitter, set_trainable
from gpflow.mean_functions import Zero
from gpflow.models.training_mixins import InputData

from . import BranchingTree as bt
from . import VBHelperFunctions

_DEFAULT_BRANCH_POINT = 0.0001  # The idea here is to place a branching point earlier than anything in the data


class Branches(IntEnum):
    TRUNK = 0
    BRANCH1 = 1
    BRANCH2 = 2


def expand_pZ0Zeros(pZ0, epsilon=1e-6):
    """
    Helper function to return prior in expanded form.
    :param pZ0: prior 2-D for each branch.
    :param epsilon: Noise added
    :return: N X N*3 expanded prior matrix. Trunk always set to 0.
    """
    assert pZ0.shape[1] == 3, "Should have exactly three cols got %g " % pZ0.shape[1]
    num_columns = 3 * pZ0.shape[0]
    r = np.zeros((pZ0.shape[0], num_columns)) + epsilon
    count = 0
    for iz0, z0 in enumerate(pZ0):
        assert z0.sum() == 1, "should sum to 1 is %s=%.3f" % (str(z0), z0.sum())
        r[iz0, count : count + 3] = z0
        count += 3
    return r


def get_Bv(branching_vector, branching_index):
    # print('Branching Index:',branching_index)
    # print('Branching vector',branching_vector)
    # we need to return (1,1) shape
    # return branching_vector[branching_index]
    return tf.expand_dims(
        tf.expand_dims(branching_vector[tf.squeeze(branching_index)], axis=0), axis=1
    )


def print_op(descr, op):
    """
    print operation for debugging
    """
    return tf.print(descr, tf.shape(op), op)


def print_all(ops):
    p = list()
    for k, v in ops.items():
        p.append(print_op(str(k), v))
    return p


class BranchKernelParam(gpflow.kernels.Kernel):
    """
    This class implements a branching point kernel for mBGP.
    """

    def __init__(
        self,
        base_kern: gpflow.kernels.Kernel,
        branchPtTensor: tf.Tensor,
        branchParam: gpflow.Parameter,
        fDebug: bool = False,
        noise_level: float = 1e-6,
    ):
        """
        :param base_kern: the base kernel for the latent GP functions.
            **NOTE:** we assume all functions share the same kernel.
        :param branchPtTensor: a tensor with shape (F, F, B) where F is the number of
            functions and B is the number of branching points.
            TODO: need to provide more details here.
        :param branchParam: branch points parameter. Shape (B, ) where B is the number of branching points.
        :param fDebug: control the amount of debug messaging. TODO: move to the logging module instead?
        :param noise_level: TODO document. Also, do we need it?
            TODO Could the user provide these instead via e.g. adding a White Noise kernel to the base kernel?
        """
        # gpflow.kernels.Kern.__init__(self, input_dim=base_kern.input_dim + 1)
        super().__init__()
        self.kern = base_kern
        self.fm = branchPtTensor
        self.fDebug = fDebug
        assert self.fm.shape[0] == self.fm.shape[1]
        assert self.fm.shape[2] > 0
        self.Bv = branchParam
        self.noise_level = noise_level

        set_trainable(self.Bv, True)

    def K(self, X, Y=None, dim=0):
        # print('get_Bv: Calling from K(branching kernel)')
        Br = get_Bv(self.Bv, dim)
        # Br = self.Bv[dim]
        # print('get_Bv: OK in K(branching kernel)')
        if Y is None:
            Y = X  # hack to avoid duplicating code below
            square = True
        else:
            square = False
        if self.fDebug:
            with tf.control_dependencies(print_all(dict(Br=Br, Bv=self.Bv))):
                t1s = tf.expand_dims(X[:, 0], 1)  # N X 1
        else:
            t1s = tf.expand_dims(X[:, 0], 1)  # N X 1
        t2s = tf.expand_dims(Y[:, 0], 1)
        i1s = tf.expand_dims(X[:, 1], 1)
        i2s = tf.expand_dims(Y[:, 1], 1)

        i1s_matrix = tf.tile(i1s, tf.reverse(tf.shape(i2s), [0]))
        i2s_matrix = tf.tile(i2s, tf.reverse(tf.shape(i1s), [0]))
        i2s_matrixT = tf.transpose(i2s_matrix)

        Ktts = self.kern.K(t1s, t2s)  # N*M X N*M
        with tf.name_scope("kttscope"):  # scope
            same_functions = tf.equal(
                i1s_matrix, tf.transpose(i2s_matrix), name="FiEQFj"
            )
            K_s = tf.where(
                same_functions, Ktts, Ktts, name="selectFiEQFj"
            )  # just setup matrix with block diagonal

        m = self.fm.shape[0]
        for fi in range(m):
            for fj in range(m):
                if fi != fj:
                    with tf.name_scope("f" + str(fi) + "f" + str(fj)):  # scope
                        # much easier to remove nans before tensorflow
                        bnan = self.fm[fi, fj, ~np.isnan(self.fm[fi, fj, :])]
                        fi_s = tf.constant(fi + 1, tf.int32, name="function" + str(fi))
                        fj_s = tf.constant(fj + 1, tf.int32, name="function" + str(fj))
                        i1s_matrixInt = tf.cast(i1s_matrix, tf.int32, name="casti1s")
                        i2s_matrixTInt = tf.cast(i2s_matrixT, tf.int32, name="casti2s")
                        fiFilter = fi_s * tf.ones_like(
                            i1s_matrixInt, tf.int32, name="fiFilter"
                        )
                        fjFilter = fj_s * tf.ones_like(
                            i2s_matrixTInt, tf.int32, name="fjFilter"
                        )  # must be transpose
                        f1F = tf.equal(i1s_matrixInt, fiFilter, name="indexF" + str(fi))
                        f2F = tf.equal(
                            i2s_matrixTInt, fjFilter, name="indexF" + str(fj)
                        )
                        t12F = tf.math.logical_and(
                            f1F, f2F, name="F" + str(fi) + "andF" + str(fj)
                        )
                        # Get the actual values of the Bs = B[index of relevant branching points]
                        bint = bnan.astype(int)  # convert to int - set of indexes
                        Bs = tf.concat(
                            [tf.slice(Br, [i - 1, 0], [1, 1]) for i in bint], 0
                        )
                        kbb = (
                            self.kern.K(Bs)
                            + tf.linalg.diag(
                                tf.ones(tf.shape(Bs)[:1], dtype=gpflow.default_float())
                            )
                            * gpflow.default_jitter()
                        )
                        Kbbs_inv = tf.linalg.inv(kbb, name="invKbb")  # B X B
                        Kb1s = self.kern.K(t1s, Bs)  # N*m X B
                        Kb2s = self.kern.K(t2s, Bs)  # N*m X B
                        a = tf.linalg.matmul(Kb1s, Kbbs_inv)
                        K_crosss = tf.linalg.matmul(
                            a, tf.transpose(Kb2s), name="Kt1_Bi_invBB_KBt2"
                        )
                        K_s = tf.where(t12F, K_crosss, K_s, name="selectIndex")

        if square:
            return (
                K_s
                + tf.eye(tf.shape(K_s)[0], dtype=gpflow.default_float())
                * self.noise_level
            )
        else:
            return K_s

    def K_diag(self, X, dim=0):
        # diagonal is just single point no branch point relevant
        return tf.linalg.diag_part(self.kern.K(X)) + self.noise_level


class AssignGP(
    gpflow.models.model.GPModel, gpflow.models.InternalDataTrainingLossMixin
):
    # gpflow.models.training_mixins.InternalDataTrainingLossMixin
    """
    Gaussian Process regression, but where the index to which the data are
    assigned is unknown.

    let f be a vector of GP points (usually longer than the number of data)

        f ~ MVN(0, K)

    and let Z be an (unknown) binary matrix with a single 1 in each row. The
    likelihood is

       y ~ MVN( Z f, \\sigma^2 I)

    That is, each element of y is a noisy realization of one (unknown) element
    of f. We use variational Bayes to infer the labels using a sparse prior
    over the Z matrix (i.e. we have narrowed down the choice of which function
    values each y is drawn from).
    """

    def _GetPhiBeta(self, dim=0):
        """Shortcut function to cut PhiBeta out."""
        Phi = tf.nn.softmax(self.logPhi)
        # BranchedPt = get_Bv(self.BranchingPoints, self.BranchingPointIndex)
        # print('get_Bv: Calling from _GetPhiBeta')
        # BranchedPt = get_Bv(self.BranchingPoints, dim)
        BranchedPt = get_Bv(self.kernel.Bv, dim)
        # print('get_Bv: OK in _GetPhiBeta')
        # x<=b trunk and if x>b branch
        # BranchedPt = np.ones((1,1)) * 0.001
        PhiBeta = tf.where(
            tf.squeeze(tf.greater(self.t, BranchedPt))[:, None], Phi, self.phiTrunk
        )

        # try squashing Phi to avoid numerical errors
        PhiBeta = (1 - 2e-6) * PhiBeta + 1e-6
        return PhiBeta

    def _GetPZ(self, dim=0):
        """
        Get prior term pZ expression dependent on branch point
        :return: pZ
        """
        # BranchedPt = get_Bv(self.BranchingPoints, self.BranchingPointIndex)
        # print('get_Bv: Calling from _GetPZ')
        # BranchedPt = get_Bv(self.BranchingPoints, dim)
        BranchedPt = get_Bv(self.kernel.Bv, dim)
        # print('get_Bv: OK in _GetPZ')
        # x<=b trunk and if x>b use prior
        pZ = tf.where(
            tf.squeeze(tf.greater(self.t, BranchedPt))[:, None], self.eZ0, self.phiTrunk
        )
        # try squashing Phi to avoid numerical errors
        pZ = (1 - 2e-6) * pZ + 1e-6
        return pZ

    # SUMON
    def GetPZ1(self):
        return self._GetPZ()

    # SUMON END

    def __init__(
        self,
        t,
        XExpanded,
        Y,
        indices,
        phiPrior=None,
        phiInitial=None,
        logPhi=None,
        fDebug=False,
        multi=False,
        sparse=True,
        kern: Optional[BranchKernelParam] = None,
    ):
        # assert len(indices) == t.size, 'indices must be size N'
        assert len(t.shape) == 1, "pseudotime should be 1D"

        if sparse:
            M = 60
            # ZExpanded = np.ones((M, 2))
            # ZExpanded[:, 0] = np.linspace(0, 1, M, endpoint=False)
            # ZExpanded[:, 1] = np.array([i for j in range(M) for i in range(1, 4)])[:M]

            Z = np.linspace(0.0, 1.0, M, endpoint=False)

            # Z = t.copy()[::2]
            # Z = np.random.permutation(t.copy())[:M]

            # self.Z = Parameter(Z.flatten().astype(gpflow.default_float()))  # one branching point per gene

            ZExpanded, Zindices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(Z)

            # self.ZExpanded = DataHolder(ZExpanded)  # inducing points for sparse GP. Same as XExpanded
            self.ZExpanded = gpflow.Parameter(ZExpanded)
            set_trainable(self.ZExpanded, False)
            assert ZExpanded.shape[1] == XExpanded.shape[1]

        # Setup branching kernel
        num_outputs = Y.shape[1]
        default_branching_points = _DEFAULT_BRANCH_POINT * np.ones(shape=(num_outputs,))
        kern = kern or get_branching_point_kernel(
            base_kernel=gpflow.kernels.SquaredExponential(),
            branching_points=default_branching_points,  # type: ignore  # ndarray can be consumed as a sequence
        )
        # self.kern.kernels[1].variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
        # self.kern.kernels[1].variance.set_trainable(False)  # jitter for numerics

        super().__init__(
            kernel=kern,
            likelihood=gpflow.likelihoods.Gaussian(),
            mean_function=Zero(),
            num_latent_gps=Y.shape[-1],
        )
        # super().__init__(name=None)
        # Set out state
        # self.likelihood = gpflow.likelihoods.Gaussian()
        self.Y = Y
        self.X = XExpanded
        self.multi = multi
        self.N = t.shape[0]
        self.D = Y.shape[1]
        self.t = t.astype(gpflow.default_float())  # could be DataHolder? advantages
        self.indices = indices
        self.logPhi = gpflow.Parameter(
            np.random.randn(t.shape[0], t.shape[0] * 3).astype(gpflow.default_float())
        )  # 1 branch point => 3 functions

        if phiInitial is None:
            phiInitial = np.ones((self.N, 2)) * 0.5  # dont know anything
            phiInitial[:, 0] = np.random.rand(self.N)
            phiInitial[:, 1] = 1 - phiInitial[:, 0]
        self.fDebug = fDebug
        # Used as p(Z) prior in KL term. This should add to 1 but will do so after UpdatePhPrior
        if phiPrior is None:
            phiPrior = np.ones((self.N, 2)) * 0.5
            phiPrior = np.c_[
                np.zeros(phiPrior.shape[0])[:, None], phiPrior
            ]  # prepend 0 for trunk

        if logPhi is not None:
            self.update_phi(prior=phiPrior)
            self.logPhi = logPhi
            N = self.Y.shape[0]
            self.phiTrunk = np.zeros((N, 3 * N))
            eps = 1e-9
            iterC = 0
            for i, p in enumerate(self.t):
                self.phiTrunk[i, iterC : iterC + 3] = np.array(
                    [1 - 2 * eps, 0 + eps, 0 + eps]
                )
                iterC += 3
        else:
            self.update_phi(phiInitial, prior=phiPrior)

    @property
    def BranchingPoints(self) -> np.ndarray:
        return self.kernel.Bv.numpy()

    def UpdateBranchingPoint(self, b, phiInitial=None, prior=None):
        """Function to update branching point and optionally reset initial conditions for variational phi"""
        assert isinstance(b, np.ndarray)
        self.kernel.Bv.assign(b.flatten())
        self.update_phi(phi_initial=phiInitial, prior=prior)

    def update_phi(self, phi_initial=None, prior=None) -> None:
        if prior is not None:
            self.eZ0 = expand_pZ0Zeros(prior)  # Set expanded prior without branch point
        if phi_initial is not None:
            self.InitialiseVariationalPhi(phi_initial)

    def InitialiseVariationalPhi(self, phiInitialIn):
        """Set initial state for Phi using branching location to constrain.
        This code has to be consistent with pZ_construction.singleBP.make_matrix to where
        the equality is placed i.e. if x<=b trunk and if x>b branch or vice versa. We use the
         former convention."""
        assert np.allclose(phiInitialIn.sum(1), 1), "probs must sum to 1 %s" % str(
            phiInitialIn
        )
        N = self.Y.shape[0]
        assert phiInitialIn.shape[0] == N
        assert phiInitialIn.shape[1] == 2  # run OMGP with K=2 trajectories
        phiInitialEx = np.zeros((N, 3 * N))
        # large neg number makes exact zeros, make smaller for added jitter
        eps = 1e-9
        phiInitial_invSoftmax = -9.0 * np.ones((N, 3 * N))
        iterC = 0
        for i, p in enumerate(self.t):
            phiInitialEx[i, iterC : iterC + 3] = np.hstack(
                [eps, phiInitialIn[i, :] - eps]
            )
            phiInitial_invSoftmax[i, iterC : iterC + 3] = np.log(
                phiInitialEx[i, iterC : iterC + 3]
            )
            iterC += 3
        assert not np.any(np.isnan(phiInitial_invSoftmax)), "no nans please " + str(
            np.nonzero(np.isnan(phiInitialEx))
        )
        # self.logPhi = phiInitial_invSoftmax
        self.logPhi = gpflow.Parameter(phiInitial_invSoftmax)
        # Create numpy matrix for trunk to overwrite Phi array for points below branch point
        self.phiTrunk = np.zeros((N, 3 * N))
        iterC = 0
        for i, p in enumerate(self.t):
            self.phiTrunk[i, iterC : iterC + 3] = np.array(
                [1 - 2 * eps, 0 + eps, 0 + eps]
            )
            iterC += 3

    def GetPhi(self):
        """Get Phi matrix, collapsed for each possible entry"""
        phiExpanded = self.GetPhiExpanded().numpy()
        l = [phiExpanded[i, self.indices[i]] for i in range(len(self.indices))]
        phi = np.asarray(l)
        tolError = 1e-6
        assert np.all(phi.sum(1) <= 1 + tolError)
        assert np.all(phi >= 0 - tolError)
        assert np.all(phi <= 1 + tolError)
        return phi

    def GetPhiExpanded(self):
        """Shortcut function to get Phi matrix out."""
        return tf.nn.softmax(self.logPhi)

        # def objectiveFun(self):
        """ Objective function to minimize - log likelihood -log prior.
        Unlike _objective, no gradient calculation is performed."""
        # return -self.compute_log_likelihood()-self.compute_log_prior()

    def maximum_log_likelihood_objective(self):
        if self.multi:
            # return self._build_likelihood_multi_b()
            return self._build_likelihood_sparse()
        else:
            print("i am here ...")
            return self._build_likelihood_single_b()

    def predict_f(self, Xnew, full_cov=False, full_output_cov: bool = False):
        assert not full_output_cov, "Not supported"
        if self.multi:
            return self._build_predict_multi_b(Xnew, full_cov)
        else:
            return self._build_predict_single_b(Xnew, full_cov)

    def _build_likelihood_multi_b(self):
        # print('assignegp_dense experimental multi compiling model (build_likelihood)')
        N = tf.cast(tf.shape(self.Y)[0], dtype=gpflow.default_float())
        M = tf.shape(self.X)[0]
        sigma2 = self.likelihood.variance
        tau = 1.0 / self.likelihood.variance
        a1 = -0.5 * N * self.D * tf.math.log(2.0 * np.pi / tau)
        a3 = -0.5 * tf.math.reduce_sum(tf.math.square(self.Y)) / sigma2
        # pick for each dimension, the kernel - can we just do this?
        a2, a4, a5 = 0, 0, 0
        for dim in range(self.D):
            K_d = self.kernel.K(self.X, dim=dim)
            Y_d = tf.expand_dims(self.Y[:, dim], axis=1)
            Phi_d = self._GetPhiBeta(dim=dim)
            # Cholesky
            L = (
                tf.linalg.cholesky(K_d)
                + tf.eye(M, dtype=gpflow.default_float()) * gpflow.default_jitter()
            )
            W = (
                tf.transpose(L)
                * tf.sqrt(tf.math.reduce_sum(Phi_d, 0))
                / tf.sqrt(sigma2)
            )
            # W = (tf.transpose(L) + tf.sqrt(tf.math.reduce_sum(Phi_d, 0))) / tf.sqrt(sigma2)
            # W = L^T + sqrt(A)/sigma2
            P = tf.linalg.matmul(W, tf.transpose(W)) + tf.eye(
                M, dtype=gpflow.default_float()
            )
            R = tf.linalg.cholesky(P)
            PhiY = tf.linalg.matmul(tf.transpose(Phi_d), Y_d)
            # LPhiY is a N X D matrix for a given branching kernel
            LPhiY = tf.linalg.matmul(tf.transpose(L), PhiY)
            c = tf.linalg.triangular_solve(R, LPhiY, lower=True) / sigma2
            # add everything up
            a2 += -0.5 * tf.math.reduce_sum(
                tf.math.log(tf.math.square(tf.linalg.diag_part(R)))
            )
            a4 += +0.5 * tf.math.reduce_sum(tf.math.square(c))
            a5 += -self.build_KL(Phi_d, dim=dim)

        if self.fDebug:
            with tf.control_dependencies(
                print_all(dict(a1=a1, a2=a2, a3=a3, a4=a4, a5=a5))
            ):
                ll = a1 + a2 + a3 + a4 + a5
        else:
            ll = a1 + a2 + a3 + a4 + a5
        return ll

    def _build_likelihood_sparse(self):
        # if self.fDebug:
        # print('assignegp_Sparse compiling model (build_likelihood)')

        N = tf.cast(tf.shape(self.Y)[0], dtype=gpflow.default_float())
        M = tf.shape(self.ZExpanded)[0]

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(self.likelihood.variance)

        a1 = (
            -0.5 * N * self.D * tf.math.log(2.0 * np.pi * sigma2)
        )  # -(N/2)log(2*pi*sigma2)
        a3 = (
            -0.5 * tf.math.reduce_sum(tf.math.square(self.Y)) / sigma2
        )  # -( 1 /(2*sigma2) ) Y^TY

        a2, a4, a5, a6 = 0.0, 0.0, 0.0, 0.0

        for dim in range(0, self.D):
            Kuu_d = (
                self.kernel.K(self.ZExpanded, dim=dim)
                + tf.eye(M, dtype=gpflow.default_float()) * gpflow.default_jitter()
            )
            Kuf_d = self.kernel.K(self.ZExpanded, self.X, dim=dim)

            Kdiag_d = self.kernel.K_diag(self.X, dim=dim)

            Y_d = tf.expand_dims(self.Y[:, dim], axis=1)

            L = tf.linalg.cholesky(Kuu_d)
            Phi_d = self._GetPhiBeta(dim=dim)

            A_d = tf.math.reduce_sum(Phi_d, 0)

            LiKuf = tf.linalg.triangular_solve(L, Kuf_d)
            W = LiKuf * tf.sqrt(A_d) / sigma
            P = tf.linalg.matmul(W, tf.transpose(W)) + tf.eye(
                M, dtype=gpflow.default_float()
            )
            traceTerm = -0.5 * tf.math.reduce_sum(
                Kdiag_d * A_d
            ) / sigma2 + 0.5 * tf.math.reduce_sum(tf.math.square(W))
            R = tf.linalg.cholesky(P)
            tmp = tf.linalg.matmul(LiKuf, tf.linalg.matmul(tf.transpose(Phi_d), Y_d))
            c = tf.linalg.triangular_solve(R, tmp, lower=True) / sigma2

            a2 += -0.5 * tf.math.reduce_sum(
                tf.math.log(tf.math.square(tf.linalg.diag_part(R)))
            )
            a4 += +0.5 * tf.math.reduce_sum(tf.math.square(c))
            a5 += -self.build_KL(Phi_d, dim=dim)
            a6 += traceTerm

            if self.fDebug:
                # trace term should be 0 for Z=X (full data)
                traceTerm = tf.print(
                    traceTerm, [traceTerm], name="traceTerm", summarize=10
                )
        if self.fDebug:
            with tf.control_dependencies(
                print_all(dict(a1=a1, a2=a2, a3=a3, a4=a4, a5=a5))
            ):
                ll = a1 + a2 + a3 + a4 + a5 + a6
        else:
            ll = a1 + a2 + a3 + a4 + a5 + a6
        return ll

    def _build_predict_multi_b(self, Xnew, full_cov=False):
        M = tf.shape(self.X)[0]
        sigma2 = self.likelihood.variance
        mean_l, var_l = [], []
        for dim in range(self.D):
            K_d = self.kernel.K(self.X, dim=dim)
            Y_d = tf.expand_dims(self.Y[:, dim], axis=1)
            Phi_d = self._GetPhiBeta(dim=dim)
            L = (
                tf.linalg.cholesky(K_d)
                + tf.eye(M, dtype=gpflow.default_float()) * gpflow.default_jitter()
            )
            W = (
                tf.transpose(L)
                * tf.sqrt(tf.math.reduce_sum(Phi_d, 0))
                / tf.sqrt(sigma2)
            )
            P = tf.linalg.matmul(W, tf.transpose(W)) + tf.eye(
                M, dtype=gpflow.default_float()
            )
            R = tf.linalg.cholesky(P)
            PhiY = tf.linalg.matmul(tf.transpose(Phi_d), Y_d)
            LPhiY = tf.linalg.matmul(tf.transpose(L), PhiY)
            c = tf.linalg.triangular_solve(R, LPhiY, lower=True) / sigma2
            Kus = self.kernel.K(self.X, Xnew, dim=dim)
            tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(R, tmp1, lower=True)
            mean = tf.linalg.matmul(tf.transpose(tmp2), c)
            if full_cov:
                var = (
                    self.kernel.K(Xnew, dim=dim)
                    + tf.linalg.matmul(tf.transpose(tmp2), tmp2)
                    - tf.linalg.matmul(tf.transpose(tmp1), tmp1)
                )
                shape = tf.stack([1, 1, tf.shape(Y_d)[1]])
                var = tf.tile(tf.expand_dims(var, 2), shape)
            else:
                var = (
                    self.kernel.K_diag(Xnew, dim=dim)
                    + tf.math.reduce_sum(tf.math.square(tmp2), 0)
                    - tf.math.reduce_sum(tf.math.square(tmp1), 0)
                )
                shape = tf.stack([1, tf.shape(Y_d)[1]])
                var = tf.tile(tf.expand_dims(var, 1), shape)
            mean_l.append(mean)
            var_l.append(var)
        return tf.concat(mean_l, axis=-1), tf.concat(var_l, axis=-1)

    def _build_likelihood_single_b(self):
        # print('assignegp_dense experimental single compiling model (build_likelihood)')
        N = tf.cast(tf.shape(self.Y)[0], dtype=gpflow.default_float())
        M = tf.shape(self.X)[0]
        D = tf.cast(tf.shape(self.Y)[1], dtype=gpflow.default_float())
        K = self.kernel.K(self.X)
        Phi = self._GetPhiBeta()
        sigma2 = self.likelihood.variance
        tau = 1.0 / self.likelihood.variance
        L = (
            tf.linalg.cholesky(K)
            + tf.eye(M, dtype=gpflow.default_float()) * gpflow.default_jitter()
        )
        W = tf.transpose(L) * tf.sqrt(tf.math.reduce_sum(Phi, 0)) / tf.sqrt(sigma2)
        P = tf.linalg.matmul(W, tf.transpose(W)) + tf.eye(
            M, dtype=gpflow.default_float()
        )
        R = tf.linalg.cholesky(P)
        PhiY = tf.linalg.matmul(tf.transpose(Phi), self.Y)
        LPhiY = tf.linalg.matmul(tf.transpose(L), PhiY)
        c = tf.linalg.triangular_solve(R, LPhiY, lower=True) / sigma2
        # compute KL
        KL = self.build_KL(Phi)
        a1 = -0.5 * N * D * tf.math.log(2.0 * np.pi / tau)
        a2 = (
            -0.5
            * D
            * tf.math.reduce_sum(tf.math.log(tf.math.square(tf.linalg.diag_part(R))))
        )
        a3 = -0.5 * tf.math.reduce_sum(tf.math.square(self.Y)) / sigma2
        a4 = +0.5 * tf.math.reduce_sum(tf.math.square(c))
        a5 = -self.D * KL
        if self.fDebug:
            with tf.control_dependencies(
                print_all(dict(a1=a1, a2=a2, a3=a3, a4=a4, a5=a5))
            ):
                ll = a1 + a2 + a3 + a4 + a5
        else:
            ll = a1 + a2 + a3 + a4 + a5
        return ll

    def _build_predict_single_b(self, Xnew, full_cov=False):
        M = tf.shape(self.X)[0]
        K = self.kernel.K(self.X)
        Phi = self._GetPhiBeta()
        sigma2 = self.likelihood.variance
        L = (
            tf.linalg.cholesky(K)
            + tf.eye(M, dtype=gpflow.default_float()) * gpflow.default_jitter()
        )
        W = tf.transpose(L) * tf.sqrt(tf.math.reduce_sum(Phi, 0)) / tf.sqrt(sigma2)
        P = tf.linalg.matmul(W, tf.transpose(W)) + tf.eye(
            M, dtype=gpflow.default_float()
        )
        R = tf.linalg.cholesky(P)
        PhiY = tf.linalg.matmul(tf.transpose(Phi), self.Y)
        LPhiY = tf.linalg.matmul(tf.transpose(L), PhiY)
        c = tf.linalg.triangular_solve(R, LPhiY, lower=True) / sigma2
        Kus = self.kernel.K(self.X, Xnew)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(R, tmp1, lower=True)
        mean = tf.linalg.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = (
                self.kernel.K(Xnew)
                + tf.linalg.matmul(tf.transpose(tmp2), tmp2)
                - tf.linalg.matmul(tf.transpose(tmp1), tmp1)
            )
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                self.kernel.K_diag(Xnew)
                + tf.math.reduce_sum(tf.math.square(tmp2), 0)
                - tf.math.reduce_sum(tf.math.square(tmp1), 0)
            )
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var

    def build_KL(self, Phi, dim=0):
        return tf.math.reduce_sum(Phi * tf.math.log(Phi)) - tf.math.reduce_sum(
            Phi * tf.math.log(self._GetPZ(dim))
        )

    def _build_predict_sparse(self, Xnew, full_cov=False):
        M = tf.shape(self.ZExpanded)[0]

        Phi = tf.nn.softmax(self.logPhi)
        # try squashing Phi to avoid numerical errors
        Phi = (1 - 2e-6) * Phi + 1e-6

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        Kuu = (
            self.kernel.K(self.ZExpanded)
            + tf.eye(M, dtype=gpflow.default_float()) * gpflow.default_jitter()
        )
        Kuf = self.kernel.K(self.ZExpanded, self.X)
        L = tf.linalg.cholesky(Kuu)

        p = tf.math.reduce_sum(Phi, 0)
        LiKuf = tf.linalg.triangular_solve(L, Kuf)
        W = LiKuf * tf.sqrt(p) / sigma
        P = tf.linalg.matmul(W, tf.transpose(W)) + tf.eye(
            M, dtype=gpflow.default_float()
        )
        R = tf.linalg.cholesky(P)
        tmp = tf.linalg.matmul(LiKuf, tf.linalg.matmul(tf.transpose(Phi), self.Y))
        c = tf.linalg.triangular_solve(R, tmp, lower=True) / sigma2

        Kus = self.kernel.K(self.ZExpanded, Xnew)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(R, tmp1, lower=True)
        mean = tf.linalg.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = (
                self.kernel.K(Xnew)
                + tf.linalg.matmul(tf.transpose(tmp2), tmp2)
                - tf.linalg.matmul(tf.transpose(tmp1), tmp1)
            )
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                self.kernel.K_diag(Xnew)
                + tf.math.reduce_sum(tf.math.square(tmp2), 0)
                - tf.math.reduce_sum(tf.math.square(tmp1), 0)
            )
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = 1,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, 2]
            where N is the number of rows in the "XExpanded" format, see above
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [N, D],
            for any positive integer the return shape contains an extra batch
            dimension, [S, N, D], with S = num_samples and D is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [N, N].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        """
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        S = num_samples
        N = Xnew.shape[0]
        D = self.Y.shape[-1]  # D

        mean, cov = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        # [N, D], [N, N, D] (full_cov=True) or [N, D], [N, D] (full_cov=False)

        if not full_cov:
            std_normal = tf.random.normal([S, N, D], dtype=default_float())
            samples = mean + tf.sqrt(cov) * std_normal  # [S, N, D]
        else:
            # mean: [N, D] and cov [N, N, D]
            jittermat = (
                tf.eye(N, batch_shape=[D], dtype=default_float()) * default_jitter()
            )  # [D, N, N]
            std_normal_samples = tf.random.normal([D, N, S], dtype=default_float())

            cov = tf.transpose(cov, perm=[2, 0, 1])  # [D, N, N]
            chol = tf.linalg.cholesky(cov + jittermat)  # [D, N, N]

            mean = tf.transpose(mean)  # [D, N]
            samples = mean[..., None] + chol @ std_normal_samples  # [D, N, S]
            samples = tf.transpose(samples, perm=[2, 1, 0])

        return samples

    def sample_prior(self, x_expanded: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Sample latent f, g, h from the model's prior (that is, ignore the data).

        Return shape: [S, N, D], where
         - S is the number of samples,
         - N is the number of points in x_expanded, and
         - D is the number of outputs

        The branch function assignment is identical to that of x_expanded.
        That is, if x_expanded[i][1] = j (recall that j is in {1, 2, 3}), then
        self.sample_prior(x_expanded)[:, i, :] is a point in branch j.
        """
        N = x_expanded.shape[0]
        D = self.Y.shape[1]
        S = num_samples

        Ks = []
        for output_dim in range(D):
            # Get the covariances for each gene. These reflect the correct branching location.
            K = self.kernel.K(x_expanded, dim=output_dim)  # type: ignore
            # [N, N]; `dim` arg exists on BranchKernelParam
            K = K[None, ...]  # [1, N, N]
            Ks.append(K)

        cov = tf.concat(Ks, axis=0)  # [D, N, N]

        jittermat = tf.eye(N, dtype=default_float()) * default_jitter()  # [N, N]
        chol = tf.linalg.cholesky(cov + jittermat)  # [D, N, N]

        std_normal_samples = tf.random.normal([D, N, S], dtype=default_float())
        samples = chol @ std_normal_samples  # [D, N, S]

        samples = tf.transpose(samples, perm=[2, 1, 0])  # [S, N, D]

        return samples.numpy()


def get_branch_point_tensor(initial_branching_point: float) -> tf.Tensor:
    tree = bt.BinaryBranchingTree(0.0, 1.0, fDebug=False)
    tree.add(None, 1, np.ones((1, 1)) * initial_branching_point)
    # TODO: why is this a single value? Seems inappropriate for a multi-output model
    (fm, _) = tree.GetFunctionBranchTensor()
    return fm


def get_branching_point_kernel(
    branching_points: Sequence[float],
    base_kernel: Optional[gpflow.kernels.Kernel] = None,
    transform: Optional[tfp.bijectors.Bijector] = None,
    prior: Optional[tfp.distributions.Distribution] = None,
) -> BranchKernelParam:
    """
    Construct a branching point kernel around the provided (initial) branching points.

    Default to the SquaredExponential base kernel and the Sigmoid transform for the branching points.
    Note that the sigmoid default means we assume pseudotime is always normalised to [0, 1].
    """
    base_kernel = base_kernel or gpflow.kernels.SquaredExponential()

    transform = transform or tfp.bijectors.Sigmoid()
    branching_point_locations: np.ndarray = (
        np.array(branching_points).flatten().astype(gpflow.default_float())
    )
    bp_param = gpflow.Parameter(
        branching_point_locations, transform=transform, prior=prior
    )

    return BranchKernelParam(
        base_kern=base_kernel,
        branchPtTensor=get_branch_point_tensor(
            initial_branching_point=branching_point_locations.mean()
        ),
        # TODO: it is not clear what synchronisation is expected between the BP in the kernel and
        #  function branch point tensor. We simply go with the mean for now, but in general it feels
        #  like we should support multidimensional outputs in the function branch tensor.
        branchParam=bp_param,
    )
