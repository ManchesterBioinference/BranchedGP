# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from GPclust import OMGP
import GPy
# Branching files
import VBHelperFunctions
import AssignGPGibbsSingleLoop
import BranchingTree as bt
import branch_kernParamGPflow as bk
import assigngp_dense

np.set_printoptions(precision=4)  # precision to print numpy array
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
XExpanded, indices, _ = AssignGPGibbsSingleLoop.GetFunctionIndexListGeneral(t)
print('XExpanded', XExpanded.shape)
print('indices', len(indices))

# A random assignment - not really needed I think TODO
randomAssignment = AssignGPGibbsSingleLoop.GetRandomInit(t, Bvalue, indices)
print(randomAssignment)
print(XExpanded[randomAssignment, :])

plt.ion()
plt.figure()
plt.plot(t, Y, 'ob')
plt.title('original data')

# OMGP initialisation
komgp1 = GPy.kern.Matern32(1)
komgp2 = GPy.kern.Matern32(1)
l = t.min() + 1
u = t.max() - 1

mo = OMGP(t[:, None], Y, K=2, variance=0.01, kernels=[komgp1, komgp2],
          prior_Z='DP')  # use a truncated DP with K=2 UNDONE
mo.kern[0].lengthscale = 10 * (u - l)
mo.kern[1].lengthscale = 10 * (u - l)
mo.optimize(step_length=0.01, maxiter=30)
fig = plt.figure(figsize=(10, 10))
mo.plot()
# Get initial phis - dont set branching point - assume 0, beginning of time
phiInitial, phiInitial_invSoftmax, XExpanded = VBHelperFunctions.InitialisePhiFromOMGP(
            None, phiOMGP=mo.phi, b=0, Y=Y, pt=t) 


# Create model
Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, BvInitial=Bvalue) + GPflow.kernels.White(1)
varianceLik = .001
Kbranch.white.variance = varianceLik  # controls the discontinuity magnitude, the gap at the branching point
Kbranch.branchkernelparam.kern.lengthscales = 0.5
Kbranch.branchkernelparam.kern.lengthscales.fixed = True
Kbranch.branchkernelparam.kern.variance = 1
Kbranch.white.variance = 1e-6
Kbranch.white.variance.fixed = True
print('Kbranch matrix', Kbranch.compute_K(XExpanded, XExpanded))
print('Branching K free parameters', Kbranch.branchkernelparam)
print('Branching K branching parameter', Kbranch.branchkernelparam.Bv)
mV = assigngp_dense.AssignGP(t, XExpanded, Y, Kbranch)
mV.likelihood.variance = varianceLik

# Initialise all model parameters using the OMGP model
# Note that the OMGP model has different kernel hyperparameters for each latent function whereas the branching model
# has one common set.
mV.logPhi = phiInitial_invSoftmax  # initialise allocations from OMGP
mV.likelihood.variance = mo.variance.values[0]
# set lengthscale to maximum
mV.kern.branchkernelparam.kern.lengthscales = np.max(
    np.array([mo.kern[0].lengthscale.values, mo.kern[1].lengthscale.values]))
# set process variance to average
mV.kern.branchkernelparam.kern.variance = np.mean(
    np.array([mo.kern[0].variance.values, mo.kern[1].variance.values]))
print(mV)

# Plot results - this will call predict
VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5))

mV._compile()
mV.optimize()

# Plot results - this will call predict
VBHelperFunctions.plotVBCode(mV, fPlotPhi=True, figsizeIn=(5, 5))
