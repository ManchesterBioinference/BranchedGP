import runVBModelSelectionCT
import os

os.environ["NSLOTS"] = '2' # use 2 cpus
os.environ["runNode"] = 'GPU' # use 2 cpus

fMatern32 = False
fTestRun = True # fast or full run?
fUsePseudoTime = False # use capture times or pseudotime?
runVBModelSelectionCT.RunAnalysis(fMatern32, fTestRun, fUsePseudoTime,fCluster=True)
