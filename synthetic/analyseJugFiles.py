import jug
import jug.task
from matplotlib import pyplot as plt
import numpy as np
import jugGPSamples


def plotBranchModel(B, pt, Y, ttestl, mul, varl, Phi, figsizeIn=(5, 5), lw=3., fs=10, labels=None,
                    fPlotPhi=True, fPlotVar=False):
    fig = plt.figure(figsize=figsizeIn)
    d = 0  # constraint code to be 1D for now
    for f in range(3):
        mu = mul[f]
        var = varl[f]
        ttest = ttestl[f]
        mean, = plt.plot(ttest, mu[:, d], linewidth=lw)
        col = mean.get_color()
        if(fPlotVar):
            plt.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
            plt.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
    v = plt.axis()
    plt.plot([B, B], v[-2:], '-m', linewidth=lw)
    # Plot Phi or labels
    if(fPlotPhi):
        gp_num = 1  # can be 0,1,2 - Plot against this
        plt.scatter(pt, Y[:, d], c=Phi[:, gp_num], vmin=0., vmax=1, s=40)
        plt.colorbar(label='GP {} assignment probability'.format(gp_num))
    return fig

if __name__ == "__main__":
    jug.init('jugGPSamples.py', 'jugGPSamples.jugdata')
    resultsFull = jug.task.value(jugGPSamples.runsFull)
    resultsSparse = jug.task.value(jugGPSamples.runsSpar)
    rall = [resultsFull, resultsSparse]
    rallDescr = ['Full', 'Sparse']
    plt.close('all')
    plt.ion()
    for ir, res in enumerate(rall):
        for r in res:
            '''{'errorInBranchingPt': errorInBranchingPt,
              'logLikelihoodRatio': logLikelihoodRatio,
              'Btry': Btry, 'BgridSearch': BgridSearch,
              'mlist': mlist, 'timingInfo': timingInfo} '''
            mlist = r['mlist']
            for ml in mlist:
                # Experiment for given B, we have one fit per search grid B
                ''' {'trueBStr': bs, 'bTrue': bTrue,
                      'pt': m.t, 'Y': m.Y.value, 'mlocallist': mlocallist}) '''
                print('trueB', ml['trueB'])
                mlocall = ml['mlocallist']
                for mlocal in mlocall:
                    ''' {'candidateB': b, 'obj': obj[ib], 'Phi': Phi,
                               'ttestl': ttestl, 'mul': mul, 'varl': varl} '''
                    plotBranchModel(mlocal['candidateB'], ml['pt'], ml['Y'],
                                    mlocal['ttestl'], mlocal['mul'], mlocal['varl'],
                                    mlocal['Phi'], fPlotPhi=True, fPlotVar=True)
                    plt.title('%s TrueB=%s b=%g NLL=%f' % (rallDescr[ir], ml['trueB'],
                                                           mlocal['candidateB'], mlocal['obj']))
