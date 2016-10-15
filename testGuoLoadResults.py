import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pods


def LoadMouseQPCRData(subsetSelection=0):
    # UNDONE should also return labels
    # From manifold load pseudotime, Y and labels
    dictData = pickle.load(open("data/guo_ssData.p", "rb"), encoding='latin1')
    YGPLVM = dictData['YGPLVM']
    ptFull = dictData['pt']
    print('Loaded GPLVM data/guo_ssData.p with nrowsXncols = ' + str(YGPLVM.shape) + '.')
    assert ptFull.ndim == 1
    assert ptFull.size == YGPLVM.shape[0]
    data = pods.datasets.singlecell()
    genes = data['Y']
    labels = data['labels']
    assert genes.shape[0] == ptFull.size
    if(subsetSelection == 0):
        pt = ptFull[:].copy()
        Y = YGPLVM.copy()
        Ygene = genes
        labels = labels
    else:
        # subset selection
        pt = ptFull[::subsetSelection].copy()
        Y = YGPLVM[::subsetSelection, :].copy()
        Ygene = genes.iloc[::subsetSelection, :]
        labels = labels[::subsetSelection]
    assert labels.size == Ygene.shape[0]
    print('LoadMouseQPCRData output')
    labelLegend = np.unique(labels)
    return pt, Y, Ygene, labels, labelLegend


def FindFiles(s):
    for ginter in geneList:
        pngFile = "%s/GuoBestfit_%s.png" % (s, ginter)
        modelFile = "%s/GuoBestfit_%s.p" % (s, ginter)
        bayesOpt = "%s/GuoBestfitBO_%s.npy" % (s, ginter)
        if not os.path.isfile(pngFile) and not os.path.isfile(modelFile) and not os.path.isfile(bayesOpt):
            print('%s files not found' % ginter)


geneList = ['Actb', 'Ahcy', 'Aqp3', 'Atp12a', 'Bmp4', 'Cdx2', 'Creb312', 'Cebpa',
            'Dab2', 'DppaI', 'Eomes', 'Esrrb', 'Fgf4', 'Fgfr2', 'Fn1', 'Gapdh',
            'Gata3', 'Gata4', 'Gata6', 'Grhl1', 'Grhl2', 'Hand1', 'Hnf4a', 'Id2',
            'Klf2', 'Klf4', 'Klf5', 'Krt8', 'Lcp1', 'Mbnl3', 'Msc', 'Msx2', 'Nanog',
            'Pdgfa', 'Pdgfra', 'Pecam1', 'Pou5f1', 'Runx1', 'Sox2', 'Sall4',
            'Sox17', 'Snail', 'Sox13', 'Tcfap2a', 'Tcfap2c', 'Tcf23', 'Utf1',
            'Tspan8']
strSavePath = 'figs'
print('Path', strSavePath)
FindFiles(strSavePath)
strSavePath = '/Users/mqbssaby/Dropbox/BranchedGP/figs'
print('Path', strSavePath)
FindFiles(strSavePath)

# plot B in order
Bgenes = np.zeros((len(geneList), 2))
Bgenes[:] = np.nan
for ig, ginter in enumerate(geneList):
        modelFile = "%s/GuoBestfit_%s.p" % (strSavePath, ginter)
        if os.path.isfile(modelFile):
            # Load model file
            print('Processing', ginter)
            mb = pickle.load(open("%s/GuoBestfit_%s.p" % (strSavePath, ginter), "rb"))
            BO = np.load("%s/GuoBestfitBO_%s.npy" % (strSavePath, ginter))
            assert BO.shape[1] == 5  # X=branching point and parameters, Y=objective
            iMin = np.argmin(BO[:, 4])
            objAtMin = BO[iMin, 4]
            assert np.allclose(mb.kern.branchkernelparam.Bv.value, BO[iMin, 0]),\
                "%s: Branching point does not match between BO and model" % ginter
            Bgenes[ig, 0] = mb.kern.branchkernelparam.Bv.value
            Bgenes[ig, 1] = objAtMin  # should match mb.objectiveFun() but faster
            assert Bgenes[ig, 0] >= 0 and Bgenes[ig, 0] <= 1  # valid pseudotime range

pt, Yall, Ygene, labels, labelLegend = LoadMouseQPCRData(0)
t = pt/100.

plt.ion()
plt.close('all')
with plt.style.context('seaborn-whitegrid'):
    fig = plt.figure(figsize=(10, 10))
    for ig, ginter in enumerate(geneList):
        modelFile = "%s/GuoBestfit_%s.p" % (strSavePath, ginter)
        if os.path.isfile(modelFile) and Bgenes[ig, 0] < 0.9:
            print(ginter, Bgenes[ig, 0], Bgenes[ig, 1])
            plt.scatter(Bgenes[ig, 0], Bgenes[ig, 1], s=80)
            plt.text(Bgenes[ig, 0], Bgenes[ig, 1], ginter, fontsize=45, color='orange')
    plt.xlabel('Branching point')
    plt.ylabel('objective function')
with plt.style.context('seaborn-whitegrid'):
    fig = plt.figure(figsize=(10, 10))
    plt.xlabel('Pseudotime')
    plt.ylabel('Principal GPLVM direction')
    plt.legend(loc='upper left')
    labelLegend = np.unique(labels)
    colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
    for lab, c in zip(labelLegend, colors):
        y1 = t[labels == lab]
        y2 = Yall[labels == lab, 0]  # principal GPLVM direction
        plt.scatter(y1, y2, label=lab, c=c, s=80)
        plt.text(np.median(y1), np.median(y2), lab, fontsize=45, color='blue')
    plt.legend(loc='upper left')
    for ig, ginter in enumerate(geneList):
        modelFile = "%s/GuoBestfit_%s.p" % (strSavePath, ginter)
        if os.path.isfile(modelFile):
            plt.scatter(Bgenes[ig, 0], Bgenes[ig, 1], s=80)
            plt.text(Bgenes[ig, 0], 0, ginter, fontsize=45, color='orange')

# Plot posterior or do test for significance?
