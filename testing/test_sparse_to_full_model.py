# Generic libraries
import unittest

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import set_trainable

# import pickle
# Branching files
from BranchedGP import BranchingTree as bt
from BranchedGP import VBHelperFunctions, assigngp_dense, assigngp_denseSparse
from BranchedGP import branch_kernParamGPflow as bk


class TestSparseVariational(unittest.TestCase):
    def InitParams(self, m):
        m.likelihood.variance.assign(0.1)
        # set lengthscale to maximum
        m.kernel.kernels[0].kern.lengthscales.assign(1.0)
        # set process variance to average
        m.kernel.kernels[0].kern.variance.assign(1.0)

    @unittest.skip("Incredibly slow, skipping")
    def test_sparse(self):
        ls, lf = self.runSparseModel()
        lss, lf2 = self.runSparseModel(M=5, atolPrediction=1, atolLik=50)
        self.assertTrue(
            np.allclose(lf, lf2, atol=1e-6),
            "Not equal likelihoods for full models %f-%f" % (lf, lf2),
        )

    def runSparseModel(self, M=None, atolPrediction=1e-3, atolLik=1):
        fDebug = True  # Enable debugging output - tensorflow print ops
        np.set_printoptions(precision=4)  # precision to print numpy array
        seed = 43
        np.random.seed(seed=seed)  # easy peasy reproducibeasy
        tf.random.set_seed(seed)
        # Data generation
        N = 20
        t = np.linspace(0, 1, N)
        print(t)
        trueB = np.ones((1, 1)) * 0.5
        Y = np.zeros((N, 1))
        idx = np.nonzero(t > 0.5)[0]
        idxA = idx[::2]
        idxB = idx[1::2]
        print(idx)
        print(idxA)
        print(idxB)
        Y[idxA, 0] = 2 * t[idxA]
        Y[idxB, 0] = -2 * t[idxB]
        # Create tree structures
        tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
        tree.add(None, 1, trueB)
        (fm, _) = tree.GetFunctionBranchTensor()
        XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)
        print("XExpanded", XExpanded.shape)
        print("indices", len(indices))  # Create model
        Kbranch = (
            bk.BranchKernelParam(gpflow.kernels.Matern32(), fm, b=trueB.copy())
            + gpflow.kernels.White()
        )
        Kbranch.kernels[0].kern.variance.assign(1.0)
        Kbranch.kernels[1].variance.assign(
            1e-6
        )  # controls the discontinuity magnitude, the gap at the branching point
        set_trainable(Kbranch.kernels[1].variance, False)  # jitter for numerics
        print("Kbranch matrix", Kbranch.K(XExpanded, XExpanded))
        print("Branching K free parameters", Kbranch.kernels[0])
        print("Branching K branching parameter", Kbranch.kernels[0].Bv)
        if M is not None:
            ir = np.random.choice(XExpanded.shape[0], M)
            ZExpanded = XExpanded[ir, :]
        else:
            ZExpanded = XExpanded  # Test on full data

        phiInitial = np.ones((N, 2)) * 0.5  # dont know anything
        mV = assigngp_denseSparse.AssignGPSparse(
            t,
            XExpanded,
            Y,
            Kbranch,
            indices,
            Kbranch.kernels[0].Bv,
            ZExpanded,
            phiInitial=phiInitial,
            fDebug=fDebug,
        )
        self.InitParams(mV)

        mVFull = assigngp_dense.AssignGP(
            t,
            XExpanded,
            Y,
            Kbranch,
            indices,
            Kbranch.kernels[0].Bv,
            fDebug=fDebug,
            phiInitial=phiInitial,
        )
        self.InitParams(mVFull)

        lsparse = mV.log_posterior_density()
        lfull = mVFull.log_posterior_density()
        print("Log likelihoods, sparse=%f, full=%f" % (lsparse, lfull))
        self.assertTrue(
            np.allclose(lsparse, lfull, atol=atolLik),
            "Log likelihoods not close, sparse=%f, full=%f" % (lsparse, lfull),
        )

        # check models identical
        assert np.all(mV.GetPhiExpanded() == mVFull.GetPhiExpanded())
        assert mV.likelihood.variance.numpy() == mVFull.likelihood.variance.numpy()
        assert mV.kernel is mVFull.kernel

        # Test prediction
        Xtest = np.array([[0.6, 2], [0.6, 3]])
        mu_f, var_f = mVFull.predict_f(Xtest)
        mu_s, var_s = mV.predict_f(Xtest)
        print("Sparse model mu=", mu_s, " variance=", var_s)
        print("Full model mu=", mu_f, " variance=", var_f)
        self.assertTrue(
            np.allclose(mu_s, mu_f, atol=atolPrediction),
            "mu not close sparse=%s - full=%s " % (str(mu_s), str(mu_f)),
        )
        self.assertTrue(
            np.allclose(var_s, var_f, atol=atolPrediction),
            "var not close sparse=%s - full=%s " % (str(var_s), str(var_f)),
        )
        return lsparse, lfull


if __name__ == "__main__":
    unittest.main()
    # To run a specific test use
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestSparseVariational)
#     unittest.TextTestRunner(verbosity=2).run(suite)
# or from command line python -m unittest test_module.TestClass.test_method
