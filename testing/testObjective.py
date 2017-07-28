# Generic libraries
import GPflow
import numpy as np
import tensorflow as tf
import unittest
# Branching files
from BranchedGP import VBHelperFunctions
from BranchedGP import FitBranchingModel


# Test VB bound is valid objective function.

fDebug = True  # Enable debugging output - tensorflow print ops
np.set_printoptions(suppress=True,  precision=5)
seed = 43
np.random.seed(seed=seed)  # easy peasy reproducibeasy
tf.set_random_seed(seed)
# Data generation
N = 20
t = np.linspace(0, 1, N)
print(t)
trueB = np.ones((1, 1))*0.4
Y = np.zeros((N, 1))
idx = np.nonzero(t > 0.5)[0]
idxA = idx[::2]
idxB = idx[1::2]
print(idx)
print(idxA)
print(idxB)
Y[idxA, 0] = 2 * t[idxA]
Y[idxB, 0] = -2 * t[idxB]
# Create global branches
globalBranchingLabels = np.ones(N)
istart = np.min(np.flatnonzero(t>0.5))
globalBranchingLabels[istart::2] = 2
globalBranchingLabels[(istart+1)::2] = 3
ptb = np.min([np.min(t[globalBranchingLabels == 2]), np.min(t[globalBranchingLabels == 3])])
# Do a search very close to globel branching time
BgridSearch = [ptb, 0.1, 0.4, np.round(ptb,1), np.round(ptb+0.01,2), 0.7, 1.1]
d = FitBranchingModel.FitModel(BgridSearch, t, Y, globalBranchingLabels,
                                          maxiter=50, priorConfidence=0.65, kervar=1., kerlen=1., likvar=0.01)
m = d['model']
iw = np.argmax(d['loglik'])
Bmode = BgridSearch[iw]

fPlot = True
if(fPlot):
    from matplotlib import pyplot as plt
    plt.ion()
    plt.close('all')
    f, ax = VBHelperFunctions.PlotBGPFit(Y, t, BgridSearch, d)


    f, ax = plt.subplots(1,1, sharex=False, sharey=False)
    ax.scatter(t, Y)
# Check winner is the truth
assignment = np.ones(N)
assignment[d['Phi'][:, 1] > 0.5] = 2
assignment[d['Phi'][:, 1] < 0.5] = 3
assignment[d['Phi'][:, 0] > 0.99] = 1

altassignment = np.ones(N)
altassignment[d['Phi'][:, 1] > 0.5] = 3
altassignment[d['Phi'][:, 1] < 0.5] = 2
altassignment[d['Phi'][:, 0] > 0.99] = 1

trueassignment = np.ones(N)
trueassignment[idxA] = 2
trueassignment[idxB] = 3

acc = np.max([(trueassignment == assignment).sum(), (trueassignment == altassignment).sum()])

assert Bmode == trueB, 'Picked B=%f' % Bmode
assert acc >= N-2, 'Got accuracy of %g' % acc

# Check objective monotone
iobjSort = np.argsort(d['loglik'])
assert iobjSort.size == 7, 'We must have added two point, the ptb and 1.1'
print(np.array(BgridSearch)[iobjSort], d['loglik'][iobjSort])
assert np.all(iobjSort == np.array([6, 5, 4, 0, 3, 1, 2]))
idx = np.array([6, 5, 4, 0, 3, 1, 2])
print('We want', (np.array(BgridSearch))[idx])