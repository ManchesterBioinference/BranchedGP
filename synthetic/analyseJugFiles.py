import jug
import jug.task
from matplotlib import pyplot as plt
import numpy as np
from BranchedGP import VBHelperFunctions as v


# name of both python code and jugdata directory
# strJugFile = 'jugGPSamplesTestRun2'
strJugFile = 'jugGPSamples'
jugCodeFile = __import__(strJugFile)
if __name__ == "__main__":
    # Control settings
    fPlotAllFits = False
    fPlotBestFit = False
    fSparse = False
    # Collections across all samples
    BtryAll = None
    BgridSearchAll = None
    objAll = None
    errorInBranchingPtAll = None
    logLikelihoodRatioAll = None
    timingInfoAll = None
    # Load data and go
    jug.init('%s.py' % strJugFile, '%s.jugdata' % strJugFile)
    resultsFull = jugCodeFile.runsFull
    if(fSparse):
        resultsSparse = jugCodeFile.runsSpar
        rall = [resultsFull, resultsSparse]
    else:
        rall = [resultsFull]
    rallDescr = ['Full', 'Sparse']
    plt.close('all')
    plt.ion()
    for ir, res in enumerate(rall):
        print('res', res)
        for ns, rt in enumerate(res):  # get task
            print('sample', ns)
            if(rt.can_load()):
                r = jug.task.value(rt)  # extract data
            else:
                print('Task %s cannot be loaded' % str(r))
                r = None
                continue
            '''{'errorInBranchingPt': errorInBranchingPt,
              'logLikelihoodRatio': logLikelihoodRatio,
              'Btry': Btry, 'BgridSearch': BgridSearch,
              'mlist': mlist, 'timingInfo': timingInfo} '''
            BgridSearch = r['BgridSearch']
            Btry = r['Btry']
            if(BtryAll is None):
                # First run
                BtryAll = Btry
                BgridSearchAll = BgridSearch
                objAll = np.zeros((len(Btry), len(BgridSearch), len(res)))  # trueB X cand B X nSamples
                errorInBranchingPtAll = np.zeros((len(Btry), len(res)))  # trueB X nSamples
                logLikelihoodRatioAll = np.zeros((len(Btry), len(res)))  # trueB X nSamples
                timingInfoAll = np.zeros((len(Btry), len(BgridSearch), len(res)))  
            else:
                assert np.all(BgridSearchAll == BgridSearch)
                assert len(BtryAll) == len(Btry)  # cannot test nan equality
            # Update collection for this sample
            errorInBranchingPtAll[:, ns] = r['errorInBranchingPt']
            logLikelihoodRatioAll[:, ns] = r['logLikelihoodRatio']
            timingInfoAll[:, :, ns] = r['timingInfo']
            mlist = r['mlist']
            assert len(mlist) == len(Btry), 'list size is %g - should be %g' % (len(mlist), len(Btry))
            for iml, ml in enumerate(mlist):
                # Experiment for given B, we have one fit per search grid B
                ''' {'trueBStr': bs, 'bTrue': bTrue,
                      'pt': m.t, 'Y': m.Y.value, 'mlocallist': mlocallist}) '''
                print('trueB', ml['trueBStr'])
                mlocall = ml['mlocallist']
                obj = []
                # Find best fit?
                for mlocal in mlocall:
                    obj.append(mlocal['obj'])
                iMin = np.argmin(obj)
                objAll[iml, :, ns] = obj
                ''' {'candidateB': b, 'obj': obj[ib], 'Phi': Phi,  'ttestl': ttestl, 'mul': mul, 'varl': varl} '''
                for im, mlocal in enumerate(mlocall):
                    print('mlocal', im)
                    if((fPlotBestFit and iMin == im) or fPlotAllFits):
                        v.plotBranchModel(mlocal['candidateB'], ml['pt'], ml['Y'],
                                          mlocal['ttestl'], mlocal['mul'], mlocal['varl'],
                                          mlocal['Phi'], fPlotPhi=True, fPlotVar=True)
                        plt.title('%s TrueB=%s b=%g NLL=%f' % (rallDescr[ir], ml['trueBStr'],
                                                               mlocal['candidateB'], mlocal['obj']))
                    else:
                        pass
        # Plot objective function for full/sparse
        assert len(BtryAll) == 4 or len(BtryAll) == 2, 'for plotting assume 4 or 2 real branching locations'
        f, axarr = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(10, 10))
        plt.title(rallDescr[ir])
        ax = axarr.flatten()
        for ib, btrue in enumerate(BtryAll):
            if(np.isnan(btrue)):
                stb = 'Ind'
            else:
                stb = str(btrue)
#             plt.boxplot(objAll[ib, :, :].T, labels=BgridSearchAll)
            ax[ib].plot(BgridSearchAll, objAll[ib, :, :])
            ax[ib].set_title('%s: Likelihood samples %g TrueB=%s' % (rallDescr[ir], len(res), stb))
            if(~np.isnan(btrue)):
                vax = ax[ib].axis()
                ax[ib].plot([btrue, btrue], vax[-2:], '--g', linewidth=3)
        '''
        Plot errors in branching point identification
        bTrue - S[im, 0] except for integrated where we show Sim[im, 0]
        So the larger the value the bigger the error.
        The more positive, the more we underestimate the branching pt
        The more negative the more we overshoot
        '''
        f, axarr = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(10, 10))
        ax = axarr.flatten()
        for ib, btrue in enumerate(BtryAll):
            if(np.isnan(btrue)):
                stb = 'Ind'
            else:
                stb = str(btrue)
            ax[ib].hist(errorInBranchingPtAll[ib, :])
            ax[ib].set_title('%s: Histogram of branching error samples %g TrueB=%s' % (rallDescr[ir], len(res), stb))
            vax = ax[ib].axis()
            ax[ib].plot([0, 0], vax[-2:], '--g', linewidth=3)  # the reference line
        # Plot evidence for branching
        f, axarr = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(10, 10))
        ax = axarr.flatten()
        for ib, btrue in enumerate(BtryAll):
            if(np.isnan(btrue)):
                stb = 'Ind'
            else:
                stb = str(btrue)
            ax[ib].hist(logLikelihoodRatioAll[ib, :])
            ax[ib].set_title('%s: Histogram of log likelihood ratios %g TrueB=%s' % (rallDescr[ir], len(res), stb))
            vax = ax[ib].axis()
            ax[ib].plot([0, 0], vax[-2:], '--g', linewidth=3)  # the further negative the more evidence for branching
        # can plot timing info

# explicitly plot the samples
p = jug.task.value(resultsFull)
f, axarr = plt.subplots(len(p), 2, sharex=False, sharey=False, figsize=(10, 10))
for pii, pi in enumerate(p):  # through samples
    mlist = pi['mlist']
    for iml, ml in enumerate(mlist):
        print('bTrue', ml['bTrue'])  # could be none
        X = ml['pt']
        Y = ml['Y']
        axarr[pii, iml].scatter(X, Y)
        axarr[pii, iml].set_title('Sample %g True B %s' % (pii, str(ml['bTrue'])))
