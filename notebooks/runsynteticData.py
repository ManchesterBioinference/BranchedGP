"""
Run BGP on synthetic data. Will save results to pickle file which
can then be examined in notebook.
"""
import pickle
import time

import numpy as np
import pandas as pd

import BranchedGP

datafile = "syntheticdata/synthetic20.csv"
data = pd.read_csv(datafile, index_col=[0])
G = data.shape[1] - 2  # all data - time columns - state column
Y = data.iloc[:, 2:]
for i in range(G):
    assert "Y" == Y.columns[i][:1], (
        "Bad gene column name %s. Should start with Y." % Y.columns[i]
    )
trueBranchingTimes = np.array([float(Y.columns[i][-3:]) for i in range(G)])
assert np.all(trueBranchingTimes > 0) and np.all(
    trueBranchingTimes <= 1.1
), "Branching time should be in [0,1.1]"


M = 10  # number of inducing points. Increase for better accuracy but at increased computational cose.
maxiter = 20  # maximum number of optimisation. Increase for better parameter estimation
tallstart = time.time()
gpmodels = []
Bsearch = [0.1, 0.2, 0.3, 0.5, 0.8, 1.1]  # set of candidate branching points
for g in range(G):
    t = time.time()
    GPy = Y.iloc[:, g][:, None]
    GPt = data["Time"].values
    globalBranching = data["MonocleState"].values.astype(int)
    gpmodels.append(
        BranchedGP.FitBranchingModel.FitModel(
            Bsearch, GPt, GPy, globalBranching, maxiter=maxiter, M=M
        )
    )
    bmode = Bsearch[np.argmax(gpmodels[g]["loglik"])]
    print(
        trueBranchingTimes[g],
        "BGP Maximum at b=%.2f" % bmode,
        "inference completed in %.1f seconds." % (time.time() - t),
    )
    _ = gpmodels[g].pop("model")  # remove model which cannot be pickled
pickle.dump(
    {"gpmodels": gpmodels, "Bsearch": Bsearch, "M": M, "maxiter": maxiter},
    open("syntheticdata/syntheticDataRun.p", "wb"),
)
tend = time.time()
print("Done - total time %.1f secs" % (tend - tallstart))
