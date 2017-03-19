# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
import unittest
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk
from BranchedGP import assigngp_dense
import tensorflow as tf
from GPflow import settings
float_type = settings.dtypes.float_type

assert float_type == tf.float32, 'must be set in gpflowrc'
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
D = 2
Bvalues = np.expand_dims(np.asarray(tree.GetBranchValues()), 1)
KbranchParam = bk.BranchKernelParam(GPflow.kernels.RBF(D - 1), fm, b=Bvalues)
KbranchParam.kern.lengthscales = 2
KbranchParam.kern.variance = 1

K = KbranchParam.compute_K(Xtrue, Xtrue)
assert KbranchParam.Bv.value == 0.5
#
# GPflow.param.DataHolder(np.asarray(1.0))
#
# custom_config = GPflow.settings.get_settings()
# custom_config.dtypes.float_type = tf.float32
# with GPflow.settings.temp_settings(custom_config):
#     GPflow.param.np_float_type
