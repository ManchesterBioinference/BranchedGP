# Generic libraries
import GPflow
import numpy as np
import unittest
# Branching files
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from GPflow import settings
float_type = settings.dtypes.float_type

class TestSparseVariational(unittest.TestCase):
    def test(self):
        # assert float_type == tf.float32, 'must be set in gpflowrc'
        N = 3  # how many points per function
        tree = bt.BinaryBranchingTree(0, 10, fDebug=False)  # set to true to print debug messages
        tree.add(None, 1, 0.5)  # single branching point
        (fm, fmb) = tree.GetFunctionBranchTensor()
        # print fmb

        tree.printTree()
        print('fm', fm)
        # print fmb
        t = np.linspace(0.01, 1, 10)
        (XForKernel, indicesBranch, Xtrue) = tree.GetFunctionIndexList(t, fReturnXtrue=True)
        # GP flow kernel
        Bvalues = np.expand_dims(np.asarray(tree.GetBranchValues()), 1)
        KbranchParam = bk.BranchKernelParam(GPflow.kernels.RBF(1), fm, b=Bvalues)
        KbranchParam.kern.lengthscales = 2
        KbranchParam.kern.variance = 1

        K = KbranchParam.compute_K(Xtrue, Xtrue)
        assert KbranchParam.Bv.value == 0.5


        samples, L, K = bk.SampleKernel(KbranchParam, XForKernel, D=1, tol=1e-6, retChol=True)
        samples, L, K = bk.SampleKernel(KbranchParam, XForKernel, D=1, tol=1e-6, retChol=False)

        # Also try the independent kernel
        indKernel = bk.IndKern(GPflow.kernels.RBF(1))
        samples, L, K = bk.SampleKernel(indKernel, XForKernel, D=1, tol=1e-6, retChol=True)

        # if you want to plot
        # from matplotlib import pyplot as plt
        # plt.ion()
        # plt.scatter(XForKernel[:, 0], samples, s=200)
        #
if __name__ == '__main__':
    unittest.main()