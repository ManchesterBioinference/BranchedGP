# Generic libraries
import gpflow
import numpy as np
import tensorflow as tf
import unittest

# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense


def InitKernParams(ms):
    ms.kern.kernels[0].kern.variance = 2
    ms.kern.kernels[0].kern.lengthscales = 5
    ms.likelihood.variance = 0.01


class TestSparseVariational(unittest.TestCase):

    @unittest.skip("Failing as of Sep-2020, to be fixed after upgrading to TF2")
    def test(self):
        np.set_printoptions(suppress=True,  precision=5)
        seed = 43
        np.random.seed(seed=seed)  # easy peasy reproducibeasy
        tf.set_random_seed(seed)
        # Data generation
        N = 20
        t = np.linspace(0, 1, N)
        print(t)
        trueB = np.ones((1, 1))*0.5
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
        assert tree.getRoot().val == trueB
        assert  tree.getRoot().idB == 1
        (fm, _) = tree.GetFunctionBranchTensor()
        XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)
        # Create model
        Kbranch = bk.BranchKernelParam(gpflow.kernels.Matern32(1), fm, b=trueB.copy()) + gpflow.kernels.White(1)
        Kbranch.kernels[1].variance = 1e-6  # controls the discontinuity magnitude, the gap at the branching point
        Kbranch.kernels[1].variance.set_trainable(False)  # jitter for numerics
        # Create model
        phiPrior = np.ones((N, 2))*0.5  # dont know anything
        phiInitial = np.ones((N, 2))*0.5  # dont know anything
        phiInitial[:, 0] = np.random.rand(N)
        phiInitial[:, 1] = 1-phiInitial[:, 0]
        m = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch, indices,
                                    Kbranch.kernels[0].Bv.value, phiPrior=phiPrior, phiInitial=phiInitial)
        InitKernParams(m)
        m.likelihood.variance.set_trainable(False)
        print('Model before initialisation\n', m, '\n===========================')
        gpflow.train.ScipyOptimizer().minimize(m, maxiter=100)
        m.likelihood.variance.set_trainable(True)
        gpflow.train.ScipyOptimizer().minimize(m, maxiter=100)
        print('Model after initialisation\n', m, '\n===========================')
        ttestl, mul, varl = VBHelperFunctions.predictBranchingModel(m)
        _, _, covl = VBHelperFunctions.predictBranchingModel(m, full_cov=True)
        for i in range(len(varl)):
            assert np.all(covl[i].diagonal().flatten() == varl[i].flatten())
        assert len(varl) == 3, 'Must have 3 predictions for 3 functions'
        assert np.all(varl[0] > 0), 'neg variances for variance function 0'
        assert np.all(varl[1] > 0), 'neg variances for variance function 1'
        assert np.all(varl[2] > 0), 'neg variances for variance function 2'
        PhiOptimised = m.GetPhi()
        print('phiPrior', phiPrior)
        print('PhiOptimised', PhiOptimised)
        assert np.allclose(PhiOptimised[idxA, 2], 1),  'PhiOptimised idxA=%s' % str(PhiOptimised[idxA, :])
        assert np.allclose(PhiOptimised[idxB, 1], 1),  'PhiOptimised idxB=%s' % str(PhiOptimised[idxB, :])
        # reset model and test informative KL prior
        m.UpdateBranchingPoint(Kbranch.kernels[0].Bv.value, phiInitial)  # reset initial phi
        InitKernParams(m)
        ll_flatprior = m.compute_log_likelihood()
        phiInfPrior = np.ones((N, 2))*0.5  # dont know anything
        phiInfPrior[-1, :] = [0.99, 0.01]
        # phiInfPrior[-2, :] = [0.01, 0.99]
        m.UpdateBranchingPoint(Kbranch.kernels[0].Bv.value, phiInitial, prior=phiInfPrior)
        ll_betterprior = m.compute_log_likelihood()
        assert ll_betterprior > ll_flatprior, '%f <> %f' % (ll_betterprior, ll_flatprior)


if __name__ == '__main__':
    unittest.main()
