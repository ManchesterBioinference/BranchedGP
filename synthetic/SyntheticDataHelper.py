# plotly
import plotly
plotly.offline.init_notebook_mode() # run at the start of every ipython notebook
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as ss
import pickle


def plotBranchModel(B, pt, Y, ttestl, mul, varl, Phi, figsizeIn=(5, 5), lw=3., fs=10, labels=None,
                    fPlotPhi=True, fPlotVar=False, ax=None):
    ''' Plotting code that does not require access to the model but takes as input predictions. '''
    if(ax is None):
        fig = plt.figure(figsize=figsizeIn)
        ax = fig.gca()
    else:
        fig = plt.gcf()
    d = 0  # constraint code to be 1D for now
    for f in range(3):
        mu = mul[f]
        var = varl[f]
        ttest = ttestl[f]
        mean, = ax.plot(ttest, mu[:, d], linewidth=lw)
        col = mean.get_color()
        if(fPlotVar):
            ax.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            ax.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
    v = ax.axis()
    ax.plot([B, B], v[-2:], '-m', linewidth=lw)
    # Plot Phi or labels
    if(fPlotPhi):
        gp_num = 1  # can be 0,1,2 - Plot against this
        PhiColor = ax.scatter(pt, Y[:, d], c=Phi[:, gp_num], vmin=0., vmax=1, s=40)
        plt.colorbar(PhiColor, label='GP {} assignment probability'.format(gp_num))
    return fig


def plotScatterMeanRanking(meanRank, title, Btry):
    ''' Function to do bubble plot of rank. Size of marker proportional to error'''
    traceCell = list()
    layout = go.Layout(showlegend=True, title='Mean ranking %s' % title,
                       annotations=list())
    for r in range(meanRank.shape[0]):
        traceCell.append(go.Scatter(
            x = Btry + 0.15*np.random.rand(len(Btry)),
            y = meanRank[r, :],
            mode='markers',
            name=r,
            text = r,
            marker={'size': 10*np.abs(meanRank[r,:] - np.array([1, 2, 3, 4]))}  # make size prop to error
            ))
    fig = go.Figure(data=traceCell, layout=layout)
    iplot(fig, filename='MeanRank%s' % title)
    
    
def GetRunData(strDataDir, fSparse, nrun, nTrueB, fPrint=True):
    assert nTrueB >= 0 and nTrueB <= 3, 'Should be 0 to 3'
    rallDescr = ['Full', 'Sparse']
    fullNamel = ['%s/runArrayJob_%s' % (strDataDir, rallDescr[0]),
                 '%s/runArrayJob_%s' % (strDataDir, rallDescr[1])]
    strfile = fullNamel[fSparse]+str(nrun)+'.p'
    if(fPrint):
        print('Open files %s' % strfile)
    r = pickle.load(open(strfile, "rb"))
    # Get objective functions and GP fits
    BgridSearch = r['BgridSearch']
    Btry = r['Btry']
    Btry[-1] = 1  # integrate GP as 1
    obj = r['mlist'][nTrueB]['obj']
    gridSearchData = r['mlist'][nTrueB]
    gridSearchGPs = r['mlist'][nTrueB]['mlocallist']
    assert len(obj) == len(gridSearchGPs), 'One GP per grid search pt'
    iMin = np.argmin(obj)  # we could also plot other GPs on the grid
    gpPlot = gridSearchGPs[iMin]  
    return obj, gridSearchData, gridSearchGPs, BgridSearch, Btry 


def GetPosteriorB(numExperiments, strDataDir, fSparse, fPrint=False):
    '''
    Return posterior on B for each experiment
    '''
    _, _, _, BgridSearch, Btry = GetRunData(strDataDir, fSparse, 1, 0, False)  # Get Bgrid and Btry. Experiments is 1-based
    posteriorB = np.zeros((numExperiments, len(Btry), len(BgridSearch))) # nexp X trueB X B grid src
    posteriorB[:] = np.nan    
    for ns in range(1, numExperiments+1):        
        for ib, b in enumerate(Btry):
            obj, gridSearchData, gridSearchGPs, BgridSearchI, BtryI = GetRunData(strDataDir, fSparse, ns, ib, False)
            assert set(BtryI) == set(Btry), 'Btry ust be the same or we are loading wrong file.'
            assert set(BgridSearchI) == set(BgridSearch), 'BgridSearch must be the same or we are loading wrong file.'
            # for each trueB calculate posterior over grid
            # ... in a numerically stable way
            o = -obj
            pn = np.exp(o - np.max(o))
            p = pn/pn.sum()
            assert np.any(~np.isnan(p)), 'Nans in p! %s' % str(p)
            assert np.any(~np.isinf(p)), 'Infinities in p! %s' % str(p)
            posteriorB[ns-1, ib,:] = p
            if(fPrint):
                print('%g:B=%s probs=' % (ns, b), np.round(p, 2))
    return posteriorB, Btry, BgridSearch

def GetMeanRank(posteriorB, Btry, BgridSearch):
    '''
    Return mean rank for synthetic experiment
    '''
    numExps = posteriorB.shape[0]
    numTrueB = posteriorB.shape[1]
    assert numTrueB == len(Btry)
    numGrid = posteriorB.shape[2]
    assert numGrid == len(BgridSearch)
    # for each experiment
    meanRank = np.zeros((numExps, numTrueB))  # nexp X num true B
    meanRank[:] = np.nan
    nMC = 100  # do Monte Carlo estimation of rank
    ranks = np.zeros((numExps, nMC, numTrueB))  # rank
    ranks[:] = np.nan
    samples = np.zeros((numExps, nMC, numTrueB))  # samples from Branching posterior
    samples[:] = np.nan
    for ns in range(numExps):
        for m in range(nMC):
            for ib, b in enumerate(Btry):
                # Sample from posterior for given branch pt
                samples[ns, m, ib] = np.random.choice(BgridSearch, p=posteriorB[ns, ib,:])
            # Rank each branch point
            ranks[ns, m,:] = ss.rankdata(samples[ns, m,:])  # only calculate rank if no errors
        # Calculate mean rank
        meanRank[ns,:] = np.mean(ranks[ns,:,:], 0)
        assert np.all(~np.isnan(meanRank[ns,:]))
    return meanRank, ranks, samples


