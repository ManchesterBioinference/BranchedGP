import unittest

import gpflow
import numpy as np
import tensorflow as tf

from BranchedGP import BranchingTree as bt
from BranchedGP import FitBranchingModel
from BranchedGP import VBHelperFunctions as bplot
from BranchedGP import branch_kernParamGPflow as bk


class TestSamplingAndPlotting(unittest.TestCase):
    @unittest.skip("Failing on CI, investigate!")
    def test(self):
        branchingPoint = 0.5
        tree = bt.BinaryBranchingTree(
            0, 10, fDebug=False
        )  # set to true to print debug messages
        tree.add(None, 1, branchingPoint)  # single branching point
        (fm, fmb) = tree.GetFunctionBranchTensor()
        # Specify where to evaluate the kernel
        t = np.linspace(0.01, 1, 60)
        (XForKernel, indicesBranch, Xtrue) = tree.GetFunctionIndexList(
            t, fReturnXtrue=True
        )
        # Specify the kernel and its hyperparameters
        # These determine how smooth and variable the branching functions are
        Bvalues = np.expand_dims(np.asarray(tree.GetBranchValues()), 1)
        KbranchParam = bk.BranchKernelParam(
            gpflow.kernels.SquaredExponential(), fm, b=Bvalues
        )
        KbranchParam.kern.lengthscales.assign(2.0)
        KbranchParam.kern.variance.assign(1.0)
        # Sample the kernel
        samples = bk.SampleKernel(KbranchParam, XForKernel)
        # Plot the sample
        bk.PlotSample(XForKernel, samples, B=Bvalues)
        # Fit model
        BgridSearch = [0.0001, branchingPoint, 1.1]
        globalBranchingLabels = XForKernel[:, 1]  # use correct labels for tests
        # could add a mistake
        print("Sparse model")
        d = FitBranchingModel.FitModel(
            BgridSearch,
            XForKernel[:, 0],
            samples,
            globalBranchingLabels,
            maxiter=40,
            priorConfidence=0.80,
            M=10,
        )
        bmode = BgridSearch[np.argmax(d["loglik"])]
        print(
            "tensorflow version", tf.__version__, "GPflow version", gpflow.__version__
        )
        print(
            "TestSamplingAndPlotting:: Sparse Log likelihood",
            d["loglik"],
            "BgridSearch",
            BgridSearch,
        )
        assert bmode == branchingPoint, bmode
        # Plot model
        pred = d["prediction"]  # prediction object from GP
        _ = bplot.plotBranchModel(
            bmode,
            XForKernel[:, 0],
            samples,
            pred["xtest"],
            pred["mu"],
            pred["var"],
            d["Phi"],
            fPlotPhi=True,
            fColorBar=True,
            fPlotVar=True,
        )

        _ = bplot.PlotBGPFit(samples, XForKernel[:, 0], BgridSearch, d)
        d = FitBranchingModel.FitModel(
            BgridSearch,
            XForKernel[:, 0],
            samples,
            globalBranchingLabels,
            maxiter=40,
            priorConfidence=0.80,
            M=0,
        )
        bmode = BgridSearch[np.argmax(d["loglik"])]
        print(
            "TestSamplingAndPlotting:: Dense Log likelihood",
            d["loglik"],
            "BgridSearch",
            BgridSearch,
        )
        assert bmode == branchingPoint, bmode
        print("Try sparse model with fixed hyperparameters")
        d = FitBranchingModel.FitModel(
            BgridSearch,
            XForKernel[:, 0],
            samples,
            globalBranchingLabels,
            maxiter=20,
            priorConfidence=0.80,
            M=15,
            likvar=1e-3,
            kerlen=2.0,
            kervar=1.0,
            fixHyperparameters=True,
        )

        # You can rerun the same code as many times as you want and get different sample paths
        # We can also sample independent functions.
        # This is the assumption in the overlapping mixtures of GPs model (OMGP) discussed in the paper.
        indKernel = bk.IndKern(gpflow.kernels.SquaredExponential())
        _ = bk.SampleKernel(indKernel, XForKernel)


if __name__ == "__main__":
    unittest.main()
