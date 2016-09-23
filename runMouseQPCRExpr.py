import mouseQPCRModelSelection
import numpy as np
import time
import AssignGPGibbsSingleLoop
import GPflow
import numpy as np
import time
import pickle as pickle
import assigngp_dense

strSaveState = 'runMouseQPCRFull'

pt, Yall = mouseQPCRModelSelection.LoadMouseQPCRData(subsetSelection=0)
Y = Yall[:, 0][None].T
print('Y array')
print(Y.shape)
strExp = 'MouseQPCR_Exp1'
m, mV = mouseQPCRModelSelection.InitModels(pt, Y)

print('Expanded array shape')
print(m.XExpanded.shape)

Bpossible = None
fSoftVBAssignment = False
fOptimizeHyperparameters = False
fReestimateMAPZ = False
numMAPsteps = 10
D = Y.shape[1]
print(D)


# Do the MAP solution and plot assignments
Kbranch = m.kern
pt = mV.t
b = 20

Bcrap = np.atleast_2d(b)  # crappy branch point
t0 = time.time()

# reinitialise hyperparameters
Kbranch.white.variance = 1e-6
Kbranch.branchkernelparam.kern.lengthscales = 120  # 20 + (90. - b) / 2. # 65
Kbranch.branchkernelparam.kern.variance = 2  # 0.0012 #  2.3158
m.likelihood.variance = 0.08
mV.likelihood.variance = m.likelihood.variance._array

# should recompute Kernel everytime we update kernel hyperparameters
m.CompileAssignmentProbability(fDebug=False, fMAP=True)

# set branching point
m.kern.branchkernelparam.Bv = Bcrap

print('============> B=' + str(m.kern.branchkernelparam.Bv._array.flatten()))

# Random assignment for given branch point

np.random.seed(47)

randomAssignment = AssignGPGibbsSingleLoop.GetRandomInit(pt, Bcrap, m.indices)
print('MAP assignment.')


(chainState, bestAssignment, _, condProbs) = m.InferenceGibbsMAP(fReturnAssignmentHistory=True,
                                                                 fDebug=False, maximumNumberOfSteps=numMAPsteps, startingAssignment=list(randomAssignment))

totalTime = time.time() - t0
print('MAP tooks ' + str(totalTime) + ' seconds.')

print('MAP assignment model')
print(m)
print('VB model')
mV

print('Commencing VB run')
Bpossible = np.linspace(22, 50, 10)
logVBBound = []

Kbranch = m.kern
pt = mV.t

mV.kern.branchkernelparam.Bv.fixed = True  # B not part of the state
mV.kern.white.variance = 1e-6
mV.kern.white.variance.fixed = True

stateSaved = mV.get_free_state().copy()

for ib, b in enumerate(Bpossible):
    t0 = time.time()

    mV.set_state(stateSaved)

    Bcrap = np.atleast_2d(b)  # crappy branch point - remove?

    # reset branching allocations for before branching point
    bestAssignmentCensored = list(bestAssignment)
    for i, bi in enumerate(bestAssignment):
        if mV.X[bi, 0] < b:
            # before branching point
            bestAssignmentCensored[i] = m.indices[i][0]

    # Variational bound computation
    mV.kern.branchkernelparam.Bv = Bcrap
    mV._needs_recompile = True

    mV.InitialisePhi(m.indices, bestAssignmentCensored, b, condProbs, fSoftAssignment=True, fSoftUni=True)

    mV.optimize(max_iters=100)

    VBbound = -mV.compute_log_likelihood()
    logVBBound.append(VBbound)

    # save mV so we can plot - also save bestAssignment
    if(strSaveState is not None):
        np.save('modelfiles/' + strSaveState + '_b' + str(ib) + '_MAPModel', bestAssignmentCensored)
        np.save('modelfiles/' + strSaveState + '_b' + str(ib) + '_VBmodel', mV.get_free_state())

    totalTime = time.time() - t0
    print('VB B=%.2f. VB bound=%.2f. Time=%g secs.' % (b, VBbound, totalTime))

if(strSaveState is not None):
    saveDict = {'Bpossible': Bpossible, 'logVBBound': logVBBound}
    pickle.dump(saveDict, open('modelfiles/' + strSaveState + '_Summary.p', "wb"))
