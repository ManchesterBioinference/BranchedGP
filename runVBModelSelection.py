import mouseQPCRModelSelection 
import numpy as np
import time
import AssignGPGibbsSingleLoop
import GPflow
import numpy as np
import time
import cPickle as pickle
import assigngp_dense
import os

t0TotalRuntime = time.time()

# In[2]:
print 'GPflow version'
print GPflow.__version__

strDir = 'modelfiles' # where to state data 
strSaveState = 'runVBModelSelection' # filenames to use

if not os.path.exists(strDir):
    os.makedirs(strDir)
    print 'Created directory ' + strDir
    
# In[3]:
Bpossible = np.linspace(2,70,3)
pt,Yall = mouseQPCRModelSelection.LoadMouseQPCRData(subsetSelection=2)
Y = Yall[:,0][None].T
print 'Y'
print Y.shape
strExp = 'MouseQPCR_Exp1'
m,mV = mouseQPCRModelSelection.InitModels(pt,Y)


# In[4]:

Bpossible=None
strSaveState='test'
fSoftVBAssignment=False
fOptimizeHyperparameters = False
fReestimateMAPZ=False
numMAPsteps = 10
D=Y.shape[1]
print D


# In[5]:

# Do the MAP solution and plot assignments
Kbranch = m.kern
pt = mV.t
b = 20

Bcrap = np.atleast_2d(b) # crappy branch point

# reinitialise hyperparameters
Kbranch.white.variance = 1e-6
Kbranch.branchkernelparam.kern.lengthscales = 100 # 20 + (90. - b) / 2. # 65
Kbranch.branchkernelparam.kern.variance = 2 #0.0012 #  2.3158
m.likelihood.variance = 0.08
mV.likelihood.variance = m.likelihood.variance._array

# should recompute Kernel everytime we update kernel hyperparameters
m.CompileAssignmentProbability(fDebug=False,fMAP=True) 

# set branching point
m.kern.branchkernelparam.Bv = Bcrap 

print '============> B=' + str(m.kern.branchkernelparam.Bv._array.flatten())

# Random assignment for given branch point

np.random.seed(47)
t0 = time.time()
randomAssignment = AssignGPGibbsSingleLoop.GetRandomInit(pt,Bcrap,m.indices)
print 'MAP assignment took ' + str(time.time() - t0) + ' secs.'

(chainState, bestAssignment,_,condProbs) =     m.InferenceGibbsMAP(fReturnAssignmentHistory=True,fDebug=False,    maximumNumberOfSteps=numMAPsteps,    startingAssignment=list(randomAssignment))
    
    
# Very important!    
mV.kern.branchkernelparam.Bv.fixed = False
mV._compile()

print 'MAP model'
print m
print 'Variational model'
print mV
Bpossible = np.linspace(22,90,20)
logVBBound = []
timesInSeconds = []

Kbranch = m.kern
pt = mV.t

# save mV so we can plot - also save bestAssignment
if(strSaveState is not None):
    saveDict = {'Bpossible':Bpossible, 'chainState':chainState, 'bestAssignment':bestAssignment }
    pickle.dump( saveDict, open( strDir + '/'+strSaveState + '_Summary.p', "wb" ) )

for ib,b in enumerate(Bpossible):  
    t0 = time.time()
    Bcrap = np.atleast_2d(b) # crappy branch point

    # reset branching allocations for before branching point
    bestAssignmentCensored = list(bestAssignment)
    for i,bi in enumerate(bestAssignment):
        if mV.X[bi,0] < b :
            # before branching point 
            bestAssignmentCensored[i] = m.indices[i][0]
        
    # Variational bound computation
    mV.kern.branchkernelparam.Bv = Bcrap
    print 'Variational kernel branch value ' + str(mV.kern.branchkernelparam.Bv._array.flatten())
    # Set state for assignments
    
    fSoft = True
    if(fSoft):
        mV.InitialisePhi(m.indices, bestAssignmentCensored, b, condProbs, fSoftAssignment = True, fSoftUni = True)
    else:
        N = Y.shape[0]
        phiInitial = np.zeros((N, 3*N))
        phiInitial_invSoftmax = np.zeros((N, 3*N))  # large neg number makes exact zeros, make smaller for added jitter
        for i, n in enumerate(bestAssignmentCensored):
            phiInitial[i, n] = 1
            phiInitial_invSoftmax[i, n] = 10
        mV.logPhi = phiInitial_invSoftmax

    VBbound = mV._objective(mV.get_free_state())[0] # this is -log of bound
    logVBBound.append(VBbound)
    t1 = time.time()
    totalTime = t1-t0
    
    timesInSeconds.append(totalTime)
    print 'B=' + str(b) + '. VB bound=%.2f. Times=%g secs'%(VBbound,totalTime)    
    
    # save mV so we can plot - also save bestAssignment
    if(strSaveState is not None):
        np.save(strDir + '/'+strSaveState + '_b' + str(ib) + '_VBmodel',mV.get_free_state())
        
        # also keep updating state
        saveDict = {'timesInSeconds':timesInSeconds,'logVBBound':logVBBound,'Bpossible':Bpossible, 'chainState':chainState, 'bestAssignment':bestAssignment}
        pickle.dump( saveDict, open( strDir + '/'+strSaveState + '_Summary.p', "wb" ) )
            

print 'Total run time is ' + str(time.time() - t0TotalRuntime)
