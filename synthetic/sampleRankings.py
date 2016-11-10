#         objAll = np.zeros((len(Btry), len(BgridSearch), len(res)))  # trueB X cand B X nSamples
import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt


strDataDir = '/home/mqbssaby/transfer/syn'
# d = np.load(strDataDir+'analyseGridFilesFull.npz')  # ned to run analysGridFiles to get this file
d = np.load(strDataDir+'analyseGridFilesSparse.npz')  # ned to run analysGridFiles to get this file
objAll = d['objAll']
BgridSearchAll = d['BgridSearchAll']
BtryAll = d['BtryAll']
objIntAll = d['objIntAll']
assert len(BgridSearchAll) == objAll.shape[1]
assert len(BtryAll) == objAll.shape[0]
# for each experiment
meanRank = np.zeros((objAll.shape[2], len(BtryAll)))
meanRank[:] = np.nan
for ns in range(objAll.shape[2]):
    print('Experiment %g ======================================= ' % ns)
    # do Monte Carlo estimation of rank
    nMC = 100
    r = np.zeros((nMC, len(BtryAll)))  # rank
    r[:] = np.nan
    s = np.zeros((nMC, len(BtryAll)))  # samples from Branching posterior
    s[:] = np.nan
    for m in range(nMC):
        for ib, b in enumerate(BtryAll):
            if(np.isnan(b)):
                bs = 'Int'
            else:
                bs = str(b)
            # for each trueB calculate posterior over grid
            objSamples = objAll[ib, :, ns]
            o = np.clip(objSamples, -600, 2000)  # for numerical stability
            # normalize and make positive
            p = np.exp(-o)
            assert np.all(~np.isinf(p)), 'infinities in p %s, obj=%s' % (str(p), str(o))
            p = p/p.sum()
            assert np.any(~np.isnan(p)), 'Nans in p! %s' % str(p)
            if(m == 0):
                print('p=', np.round(p, 2), ' Obj=\n', objSamples, 'clipped=\n', o)
            # Sample from posterior
            s[m, ib] = np.random.choice(BgridSearchAll, p=p)
        # Now go do MCMC sample from posterior
        r[m, :] = ss.rankdata(s[m, :])  # only calculate rank if no errors
    # Calculate mean rank
    meanRank[ns, :] = np.mean(r, 0)
    assert np.all(~np.isnan(meanRank[ns, :]))
    print('Mean rank', meanRank[ns, :])

plt.ion()
plt.figure(figsize=(10, 10))
plt.boxplot(meanRank, labels=['Early', 'Med', 'Late', 'Int'])
