from matplotlib import pyplot as plt
import time
import mouseQPCRModelSelection
import pods
import numpy as np

''' File to run model selection on mouse QPCR data
   Need to set 
   subsetSelection = integer - how many point to skip before next point?
   fPseudoTime     = Boolean - use pseudotime or capture time?
   strgene             = string - which gene to look at
'''
subsetSelection = 4
strgene = 'Id2'
fPseudoTime = True # if false use capture time

print 'Doing subsets selection %g, looking at gene %s and pseudotime=%g'%(subsetSelection,strgene,fPseudoTime)
print 'Loading QPCR data'
data = pods.datasets.singlecell()
genes = data['Y']
labels = data['labels']
label_dict = dict(((i,l) for i,l in enumerate(labels)))

YFull = genes[strgene].values

N = genes.shape[0]
G = genes.shape[1]
genes.describe()
print genes.shape
stageCell = np.zeros(N)
stageN = np.zeros(N)
for i,l in enumerate(labels):
    stageCell[i] = int(l[:2])
    stageN[i] = np.log2(stageCell[i]) + 1
    
# Load pseudotime as estimated by Bayesian GPLVM (Max's method)
if(fPseudoTime):
    ptFull,YGPLVM = mouseQPCRModelSelection.LoadMouseQPCRData(subsetSelection=0)
    assert ptFull.size == stageCell.size, 'Pseudotime should be same size.  stageCell=' + str(stageCell.shape) + ' ptFull=' + str(ptFull.shape)
    assert YGPLVM.shape[0] == YFull.shape[0], 'Y shapes dont match YGPLVM=' + str(YGPLVM.shape) + ' YFull=' + str(YFull.shape)
    print 'Using pseudotime'
else:
    print 'Using capture times'
    ptFull = stageCell
    
print 'Doing map inference. Date shapes='
t0 = time.time()
pt = ptFull[::subsetSelection].copy()
Y = YFull[::subsetSelection,None].copy()    
print pt.shape
print Y.shape
#m,mV = mouseQPCRModelSelection.InitModels(pt,Y,nsparse=100)
m,mV = mouseQPCRModelSelection.InitModels(pt,Y) # non-sparse

# 5, 10, 20., 30., 50., 60.
logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m,mV,Bpossible=np.array([20]), strSaveState='rawData'+str(fPseudoTime), \
    fSoftVBAssignment=True, fOptimizeHyperparameters = False, fReestimateMAPZ=True,\
    numMAPsteps = 10, fPlotFigure=True)
print 'Times=%g secs'%(time.time()-t0)   