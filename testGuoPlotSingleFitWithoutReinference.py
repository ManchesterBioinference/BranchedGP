# Generic libraries
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle
import pods
# Branching files


def plotVBCode(mV, lw=3., fs=10, fPlotVar=False):
    B = mV.kern.branchkernelparam.Bv.value.flatten()
    assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)
    pt = mV.t
    l = np.min(pt)
    u = np.max(pt)
    d = 0  # constraint code to be 1D for now
    for f in range(1, 4):
        if(f == 1):
            ttest = np.linspace(l, B, 100)[:, None]  # root
        else:
            ttest = np.linspace(B, u, 100)[:, None]
        Xtest = np.hstack((ttest, ttest * 0 + f))
        mu, var = mV.predict_f(Xtest)
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
        mean, = plt.plot(ttest, mu[:, d], linewidth=lw)
        col = mean.get_color()
        if(fPlotVar):
            plt.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            plt.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)

    v = plt.axis()
    plt.plot([B, B], v[-2:], '-m', linewidth=lw)


def plotGene(t,g):
    with plt.style.context('seaborn-whitegrid'):
        colors = cm.spectral(np.linspace(0, 1, len(labelLegend)))
        plt.figure(figsize=(10, 10))
        for lab, c in zip(labelLegend, colors):
            y1 = t[labels == lab]
            y2 = g[labels == lab]
            plt.scatter(y1,y2,label=lab, c=c,s=80)
            plt.text(np.median(y1),np.median(y2),lab, fontsize=45, color='blue')
        plt.legend(loc='upper left')


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

########################################
pt, Yall, Ygene, labels, labelLegend = LoadMouseQPCRData(0)
t = pt/100.

# get gene name
parser = argparse.ArgumentParser(description='Process genes..')
parser.add_argument('g', help='gene name')
args = vars(parser.parse_args())
ginter = args['g']
print('Processing gene %s' % ginter)
Y = Ygene[ginter].values[:, None]

strSavePath = '/Users/mqbssaby/Dropbox/BranchedGP/figs'
mb = pickle.load(open("%s/GuoBestfit_%s.p" % (strSavePath, ginter), "rb"))

plt.ion()
plt.close('all')
plotGene(t, Y)
plotVBCode(mb, lw=3., fs=10, fPlotVar=True)
plt.title('%s B=%g' % (ginter, mb.kern.branchkernelparam.Bv.value))

# 
# mb.optimize()
# VBHelperFunctions.plotVBCode(mb, labels=labels, figsizeIn=(5, 5), fPlotVar=True)
# plt.title('Optimised %s B=%g ll=%.2f' % (ginter, mb.kern.branchkernelparam.Bv.value, mb.objectiveFun()))
