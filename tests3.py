import GPflow
import numpy as np
import tensorflow as tf
import AssignGPGibbsSingleLoop
import BranchingTree as bt
np.set_printoptions(precision=4)  # precision to print numpy array
import branch_kernParamGPflow as bk
import time
import assigngp_dense

seed = 43

np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)

N = 20
t = np.linspace(0, 1, N)
print(t)
Bv = 0.5
Y = np.zeros((N, 1))
idx = np.nonzero(t > 0.5)[0]
idxA = idx[::2]
idxB = idx[1::2]
print(idx)
print(idxA)
print(idxB)
Y[idxA, 0] = 2 * t[idxA]
Y[idxB, 0] = -2 * t[idxB]

# plt.plot(t,Y,'ob')
Bvalue = np.ones((1, 1)) * 0.5
print(Bvalue)

tree = bt.BinaryBranchingTree(0, 1, fDebug=False)
tree.add(None, 1, Bvalue)
(fm, _) = tree.GetFunctionBranchTensor()

print(fm)

Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, BvInitial=Bvalue) + GPflow.kernels.White(1)
varianceLik = .001
Kbranch.white.variance = varianceLik  # controls the discontinuity magnitude, the gap at the branching point
Kbranch.branchkernelparam.kern.lengthscales = 5
Kbranch.branchkernelparam.kern.variance = 1
Kbranch.branchkernelparam.Bv = Bvalue
Kbranch.branchkernelparam.Bv.fixed = True


XExpanded, indices, _ = AssignGPGibbsSingleLoop.GetFunctionIndexListGeneral(t)


print('XExpanded', XExpanded.shape)
print('indices', len(indices))
# print XSampleGeneral

mV = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch)
mV.likelihood.variance = varianceLik
mV._compile()  # creates objective function

randomAssignment = AssignGPGibbsSingleLoop.GetRandomInit(t, Bvalue, indices)
print(randomAssignment)
print(XExpanded[randomAssignment, :])
