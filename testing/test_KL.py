# Generic libraries
import unittest

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import set_trainable

# Branching files
from BranchedGP import BranchingTree as bt
from BranchedGP import FitBranchingModel, VBHelperFunctions, assigngp_dense
from BranchedGP import branch_kernParamGPflow as bk


class TestKL(unittest.TestCase):
    def test(self):
        fDebug = True  # Enable debugging output - tensorflow print ops
        np.set_printoptions(suppress=True, precision=5)
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
        globalBranchingLabels = np.ones(N)
        globalBranchingLabels[4::2] = 2
        globalBranchingLabels[5::2] = 3

        XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)
        phiInitial, phiPrior = FitBranchingModel.GetInitialConditionsAndPrior(
            globalBranchingLabels, 0.51, False
        )
        ptb = np.min(
            [
                np.min(t[globalBranchingLabels == 2]),
                np.min(t[globalBranchingLabels == 3]),
            ]
        )
        tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
        tree.add(None, 1, np.ones((1, 1)) * ptb)  # B can be anything here
        (fm1, _) = tree.GetFunctionBranchTensor()

        # Look at kernels
        fDebug = True
        Kbranch1 = bk.BranchKernelParam(
            gpflow.kernels.Matern32(), fm1, b=np.ones((1, 1)) * ptb, fDebug=fDebug
        )
        K1 = Kbranch1.K(XExpanded, XExpanded)

        Kbranch2 = bk.BranchKernelParam(
            gpflow.kernels.Matern32(), fm1, b=np.ones((1, 1)) * 0.20, fDebug=fDebug
        )
        K2 = Kbranch2.K(XExpanded, XExpanded)

        Kbranch3 = bk.BranchKernelParam(
            gpflow.kernels.Matern32(), fm1, b=np.ones((1, 1)) * 0.22, fDebug=fDebug
        )
        K3 = Kbranch3.K(XExpanded, XExpanded)

        # Look at model
        kb = (
            bk.BranchKernelParam(gpflow.kernels.Matern32(), fm1, b=np.zeros((1, 1)))
            + gpflow.kernels.White()
        )
        kb.kernels[1].variance.assign(
            1e-6
        )  # controls the discontinuity magnitude, the gap at the branching point
        set_trainable(kb.kernels[1].variance, False)  # jitter for numerics
        # m = assigngp_dense.AssignGP(t, XExpanded, Y, kb, indices, np.ones((1, 1)), phiInitial=phiInitial, phiPrior=phiPrior)
        m = assigngp_dense.AssignGP(
            t,
            XExpanded,
            Y,
            kb,
            indices,
            np.ones((1, 1)),
            phiInitial=phiInitial,
            phiPrior=phiPrior,
            KConst=K1,
            fDebug=True,
        )

        m.UpdateBranchingPoint(np.ones((1, 1)) * ptb, phiInitial.copy())
        ptbLL = m.log_posterior_density()
        m.UpdateBranchingPoint(np.ones((1, 1)) * 0.20, phiInitial.copy())
        eLL = m.log_posterior_density()
        m.UpdateBranchingPoint(np.ones((1, 1)) * 0.22, phiInitial.copy())
        lll = m.log_posterior_density()
        print(eLL, ptbLL, lll)
        assert eLL < ptbLL
        assert np.allclose(ptbLL, lll)


if __name__ == "__main__":
    unittest.main()
