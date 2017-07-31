'''
Run BGP on synthetic data. Will save results to pickle file which
can then be examined in notebook.
'''
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import BranchedGP

datafile='syntheticdata/synthetic20.csv'
data = pd.read_csv(datafile, index_col=[0])
G = data.shape[1] - 2  # all data - time columns - state column
Y = data.iloc[:, 2:]
for i in range(G):
    assert 'Y' == Y.columns[i][:1], 'Bad gene column name %s. Should start with Y.' % Y.columns[i]
trueBranchingTimes = np.array([float(Y.columns[i][-3:]) for i in range(G)])
assert np.all(trueBranchingTimes>0) and np.all(trueBranchingTimes <= 1.1), 'Branching time should be in [0,1.1]'
plt.ion()
f, ax = plt.subplots(5, 8, figsize=(10, 8))
ax = ax.flatten()
for i in range(G):
    for s in np.unique(data['MonocleState']):
        idxs = (s == data['MonocleState'].values)
        ax[i].scatter(data['Time'].loc[idxs], Y.iloc[:, i].loc[idxs])
        ax[i].set_title(Y.columns[i])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
f.suptitle('Branching genes, location=1.1 indicates no branching')

for fixInducingPoints in [True, False]:
    print('fixInducingPoints', fixInducingPoints)
    tallstart = time.time()
    for g in [0, 10, 30]:
        t = time.time()
        Bsearch = [0.1, 0.2, 0.3, 0.5, 0.8, 1.1]  # set of candidate branching points
        GPy = Y.iloc[:, i][:, None]
        GPt = data['Time'].values
        globalBranching = data['MonocleState'].values.astype(int)
        d = BranchedGP.FitBranchingModel.FitModel(Bsearch, GPt, GPy, globalBranching, maxiter=200, fixInducingPoints=fixInducingPoints,
                                                  fDebug=False, M=30)
        print(trueBranchingTimes[g], 'BGP Maximum at b=%.2f' % Bsearch[np.argmax(d['loglik'])], 'inference completed in %.1f seconds.' %  (time.time()-t))

    tend = time.time()
    print('Done - total time %.1f secs' % (tend-tallstart))


# TODO: we could do one step optimisation?
