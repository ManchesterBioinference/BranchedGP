import pickle
import pods
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import scipy.stats


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
    N = labels.size
    stageCell = np.zeros(N)
    stageN = np.zeros(N)
    for i, l in enumerate(labels):
        stageCell[i] = int(l[:2])
        stageN[i] = np.log2(stageCell[i]) + 1

    return pt, Y, Ygene, labels, labelLegend, stageCell, stageN


def plotGene(t, g, labels):
    labelLegend = np.unique(labels)
    with plt.style.context('seaborn-whitegrid'):
        colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
        plt.figure(figsize=(10, 10))
        for lab, c in zip(labelLegend, colors):
            y1 = t[labels == lab]
            y2 = g[labels == lab]
            plt.scatter(y1, y2, label=lab, c=c, s=80)
            plt.text(np.median(y1), np.median(y2), lab, fontsize=45, color='blue')
        plt.legend(loc='upper left')


# evaluate roughness of pseudotime
def calcroughness(x, pt):
    '''
    This metric measures the smoothness of the gene expression profile by looking at the
    differences of consecutive measurements. Smaller values indicate a smoother response.
    '''
    x = np.atleast_2d(x)
    i = np.argsort(pt)
    x = x[:, i]
    N = x.shape[1]
    assert(N > 0)
    S = x.std(axis=1)
    return np.sqrt(np.sum(np.square(x[:, 0:(N-1)] - x[:, 1:N]), 1) / (N-1)) / S


def evaluatePseudoTime(pt, Ygene, labels, stageCell):
    '''
    this assesses how good a pseudotime is from the following perspectives:
    1. Roughness across the three cell fates (TE, EPI, PE)
    2. Rank correlation with capture time
    '''
    roughnessAcrossGenes = pd.DataFrame(np.zeros((Ygene.columns.size, 3)), index=Ygene.columns, columns=['TE', 'EPI', 'PE'])
    for ig, g in enumerate(Ygene):
        gd = Ygene[g]
        idx = np.logical_or(np.logical_or(labels == '32 TE', labels == '64 TE'), stageCell <= 16)
        assert np.unique(labels[idx]).size == 7  # just double check
        roughnessAcrossGenes.iloc[ig, 0] = calcroughness(gd[idx].values, pt[idx])
        # need EPI and PE paths
        idx = np.logical_or(np.logical_or(labels == '32 ICM', labels == '64 EPI'), stageCell <= 16)
        assert np.unique(labels[idx]).size == 7  # just double check
        roughnessAcrossGenes.iloc[ig, 1] = calcroughness(gd[idx].values, pt[idx])
        idx = np.logical_or(np.logical_or(labels == '32 ICM', labels == '64 PE'), stageCell <= 16)
        assert np.unique(labels[idx]).size == 7  # just double check
        roughnessAcrossGenes.iloc[ig, 2] = calcroughness(gd[idx].values, pt[idx])

    roughnessAcrossGenes.sort_values(ascending=True, axis=0, inplace=True, by=['TE', 'EPI', 'PE'])
    qs = roughnessAcrossGenes.quantile(q=[0.05, .5, .95])
    # This asseses how good the pseudotime rank correlates with capture time
    rankCor = scipy.stats.spearmanr(pt, stageCell)[0]
    return qs, rankCor, roughnessAcrossGenes
