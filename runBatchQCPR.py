from matplotlib import pyplot as plt
import time
import mouseQPCRModelSelection
import pods
import numpy as np
import tensorflow as tf

''' File to run model selection on mouse QPCR data
   Need to set
   subsetSelection = integer - how many point to skip before next point?
   fPseudoTime     = Boolean - use pseudotime or capture time?
   strgene             = string - which gene to look at
'''
subsetSelection = 3
strgene = 'Id2'
for fPseudoTime in [True, False]:  # if false use capture time
    print('Doing subsets selection %g, looking at gene %s and pseudotime=%g' % (subsetSelection, strgene, fPseudoTime))
    print('Loading QPCR data')
    data = pods.datasets.singlecell()
    genes = data['Y']
    labels = data['labels']
    label_dict = dict(((i, l) for i, l in enumerate(labels)))

    N = genes.shape[0]
    G = genes.shape[1]
    genes.describe()
    print(genes.shape)
    stageCell = np.zeros(N)
    stageN = np.zeros(N)
    for i, l in enumerate(labels):
        stageCell[i] = int(l[:2])
        stageN[i] = np.log2(stageCell[i]) + 1

    # Load pseudotime as estimated by Bayesian GPLVM (Max's method)
    if(fPseudoTime):
        ptFull, YGPLVM = mouseQPCRModelSelection.LoadMouseQPCRData(subsetSelection=0)
        assert ptFull.size == stageCell.size, 'Pseudotime should be same size.  stageCell=' + \
            str(stageCell.shape) + ' ptFull=' + str(ptFull.shape)
        assert YGPLVM.shape[0] == N, 'Xhapes dont match YGPLVM=' + str(YGPLVM.shape) + ' ' + str(N)
        print('Using pseudotime')
    else:
        print('Using capture times')
        ptFull = stageCell

    for g in genes.columns:
        tf.reset_default_graph()
        YFull = genes[g].values
        strFile = 'rawData' + str(fPseudoTime) + 'g_' + g
        print('Processing gene %s. Doing map inference. Will save to file %s. Date shapes=' % (g, strFile))
        pt = ptFull[::subsetSelection].copy()
        Y = YFull[::subsetSelection, None].copy()
        print(pt.shape)
        print(Y.shape)
        #m,mV = mouseQPCRModelSelection.InitModels(pt,Y,nsparse=100)
        m, mV = mouseQPCRModelSelection.InitModels(pt, Y)  # non-sparse
        l = np.min(pt)
        u = np.max(pt)
        Bpossible = np.linspace(l + 10, u - 10, 30)
        print('Running inference B=' + str(Bpossible))
        t0 = time.time()
        logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m, mV, Bpossible=Bpossible, strSaveState=strFile,
                                                                           fSoftVBAssignment=True, fOptimizeHyperparameters=False, fReestimateMAPZ=True,
                                                                           numMAPsteps=10, fPlotFigure=False)
        print('Times=%g secs' % (time.time() - t0))

    for g in genes.columns:
        tf.reset_default_graph()
        YFull = genes[g].values
        strFile = 'OptimizerawData' + str(fPseudoTime) + 'g_' + g
        print('Processing gene %s. Doing map inference. Will save to file %s. Date shapes=' % (g, strFile))
        pt = ptFull[::subsetSelection].copy()
        Y = YFull[::subsetSelection, None].copy()
        print(pt.shape)
        print(Y.shape)
        #m,mV = mouseQPCRModelSelection.InitModels(pt,Y,nsparse=100)
        m, mV = mouseQPCRModelSelection.InitModels(pt, Y)  # non-sparse
        l = np.min(pt)
        u = np.max(pt)
        Bpossible = np.linspace(l + 10, u - 10, 30)
        print('Running inference B=' + str(Bpossible))
        t0 = time.time()
        logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m, mV, Bpossible=Bpossible, strSaveState=strFile,
                                                                           fSoftVBAssignment=True, fOptimizeHyperparameters=True, fReestimateMAPZ=True,
                                                                           numMAPsteps=10, fPlotFigure=False)
        print('Times=%g secs' % (time.time() - t0))
