# Generic libraries
import gpflow
import numpy as np
import unittest
# Branching files
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk

class TestKernelSampling(unittest.TestCase):
    def test(self):
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
        KbranchParam = bk.BranchKernelParam(gpflow.kernels.SquaredExponential(), fm, b=Bvalues)
        KbranchParam.kern.lengthscales.assign(2)
        KbranchParam.kern.variance.assign(1)

        K = KbranchParam.K(Xtrue, Xtrue)
        assert KbranchParam.Bv == 0.5


        samples, L, K = bk.SampleKernel(KbranchParam, XForKernel, D=1, tol=1e-6, retChol=True)
        samples2 = bk.SampleKernel(KbranchParam, XForKernel, D=1, tol=1e-6, retChol=False)

        # Also try the independent kernel
        indKernel = bk.IndKern(gpflow.kernels.SquaredExponential())
        samples3, L, K = bk.SampleKernel(indKernel, XForKernel, D=1, tol=1e-6, retChol=True)

        samples4 = KbranchParam.SampleKernel(XForKernel, b=Bvalues)

        XAssignments = bk.GetFunctionIndexSample(t)  # assign to either branch randomly
        XAssignments[XAssignments[:, 0] <= tree.GetBranchValues(), 1] = 1
        samples5 = KbranchParam.SampleKernelFromTree(XAssignments, b=tree.GetBranchValues())

        # if you want to plot
        # from matplotlib import pyplot as plt
        # plt.ion()
        # plt.scatter(XForKernel[:, 0], samples, s=200)
        #
if __name__ == '__main__':
    unittest.main()