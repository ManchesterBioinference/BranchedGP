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
r, m = FitBranchingModel.EstimateBranchModel('', 1, globalBranchingLabels, t, Y, BgridSearchIn=[0.1, 0.4, np.round(ptb,1), np.round(ptb+0.01,2), 0.7],
                fSavefile=False, M=10, maxiter=50, infPriorPhi=True, v=[0.65],
                kervarIn=[1], kerlenIn=[1], noiseInSamplesIn=[0.01],
                fFixhyperpar=False, fDebug=True)
iw = np.argmax(r['obj'])
Bmode = r['BgridSearch'][iw]

fPlot = True
if(fPlot):
    from matplotlib import pyplot as plt
    plt.ion()
    plt.close('all')
    plt.figure()
    for i in range(1, 4):
        plt.plot(t[globalBranchingLabels == i], Y[globalBranchingLabels == i].flatten(), 'o', label=i)
    plt.legend()
    f, ax = plt.subplots(2,4, sharex=False, sharey=False)
    ax = ax.flatten()
    for ib, b in enumerate(r['BgridSearch']):
        VBHelperFunctions.plotBranchModel(b, r['GPt'], r['GPy'], r['mlocallist'][ib]['ttestl'],
                                          r['mlocallist'][ib]['mul'], r['mlocallist'][ib]['varl'],
                                          r['mlocallist'][ib]['Phi'],
                                          fPlotPhi=True, fPlotVar=True, ax=ax[ib], fColorBar=False)
        s = ''
        if(iw == ib):
            s = 'winner'
        ax[ib].set_title('B=%.3f LL=%.2f %s' % (b, r['obj'][ib], s))

    ax[-1].scatter(r['BgridSearch'], r['obj'])
# Check winner is the truth
assignment = np.ones(N)
assignment[r['mlocallist'][iw]['Phi'][:, 1] > 0.5] = 2
assignment[r['mlocallist'][iw]['Phi'][:, 1] < 0.5] = 3
assignment[r['mlocallist'][iw]['Phi'][:, 0] > 0.99] = 1

altassignment = np.ones(N)
altassignment[r['mlocallist'][iw]['Phi'][:, 1] > 0.5] = 3
altassignment[r['mlocallist'][iw]['Phi'][:, 1] < 0.5] = 2
altassignment[r['mlocallist'][iw]['Phi'][:, 0] > 0.99] = 1

trueassignment = np.ones(N)
trueassignment[idxA] = 2
trueassignment[idxB] = 3

acc = np.max([(trueassignment == assignment).sum(), (trueassignment == altassignment).sum()])

assert Bmode == trueB, 'Picked B=%f' % Bmode
assert acc >= N-2, 'Got accuracy of %g' % acc

# Check objective monotone
iobjSort = np.argsort(r['obj'])
assert iobjSort.size == 7, 'We must have added two point, the ptb and 1.1'
print(np.array(r['BgridSearch'])[iobjSort], r['obj'][iobjSort])
assert np.all(iobjSort == np.array([6, 5, 4, 0, 3, 1, 2]))
