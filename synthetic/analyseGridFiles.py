from matplotlib import pyplot as plt
from BranchedGP import VBHelperFunctions as v
import pickle
import glob

if __name__ == "__main__":
    # iterate over all files and make list of all results - so code between jug and here same
    strDataDir = '.'  # Where data files reside
    rallDescr = ['Full', 'Sparse']
    fullNamel = ['runArrayJob_%s' % rallDescr[0], 'runArrayJob_%s' % rallDescr[1]]
    rall = list()
    for fullName in fullNamel:
        rFullOrSparse = list()
        for file in glob.glob("%s*.p" % fullName):
            r = pickle.load(open(file, "rb"))
            # get seed
            seed = int(file[len(fullName):-2])
            r['seed'] = seed  # add seed
            rFullOrSparse.append(r)
        rall.append(rFullOrSparse)
    plt.close('all')
    plt.ion()
    for ir, res in enumerate(rall):
        for r in res:
            '''{'errorInBranchingPt': errorInBranchingPt,
              'logLikelihoodRatio': logLikelihoodRatio,
              'Btry': Btry, 'BgridSearch': BgridSearch,
              'mlist': mlist, 'timingInfo': timingInfo} 'seed': seed'''
            mlist = r['mlist']
            for ml in mlist:
                # Experiment for given B, we have one fit per search grid B
                ''' {'trueBStr': bs, 'bTrue': bTrue,
                      'pt': m.t, 'Y': m.Y.value, 'mlocallist': mlocallist}) '''
                print('trueB', ml['trueBStr'])
                mlocall = ml['mlocallist']
                for mlocal in mlocall:
                    ''' {'candidateB': b, 'obj': obj[ib], 'Phi': Phi,
                               'ttestl': ttestl, 'mul': mul, 'varl': varl} '''
                    v.plotBranchModel(mlocal['candidateB'], ml['pt'], ml['Y'],
                                      mlocal['ttestl'], mlocal['mul'], mlocal['varl'],
                                      mlocal['Phi'], fPlotPhi=True, fPlotVar=True)
                    plt.title('%s seed=%g TrueB=%s b=%g NLL=%f' %
                              (rallDescr[ir], r['seed'], ml['trueBStr'],
                               mlocal['candidateB'], mlocal['obj']))
