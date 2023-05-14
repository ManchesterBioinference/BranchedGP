# ---
# jupyter:
#   anaconda-cloud: {}
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.5
# ---

# %% [markdown]
# Branching GP Regression on synthetic data
# --
#
# *Alexis Boukouvalas, 2017*
#
# Branching GP regression with Gaussian noise on the hematopoiesis data described in the paper "BGP: Gaussian processes for identifying branching dynamics in single cell data".
#
# This notebook shows how to build a BGP model and plot the posterior model fit and posterior branching times.

# %%
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BranchedGP import VBHelperFunctions as bplot

plt.style.use("ggplot")
# %matplotlib inline

# %% [markdown]
# ### Load the data
# 1. Monocle has already been run on the data. The first columns contains the state assigned by the DDRTree algorithm to each cell.
# 1. Second column is the gene time.
# 1. All other columns are the 40 genes. The first 10 branch early, then 20 branch late and 10 do not branch.

# %%
datafile = "syntheticdata/synthetic20.csv"
data = pd.read_csv(datafile, index_col=[0])
G = data.shape[1] - 2  # all data - time columns - state column
Y = data.iloc[:, 2:]
trueBranchingTimes = np.array([float(Y.columns[i][-3:]) for i in range(G)])

# %%
data.head()

# %% [markdown]
# # Plot the data

# %%
f, ax = plt.subplots(5, 8, figsize=(10, 8))
ax = ax.flatten()
for i in range(G):
    for s in np.unique(data["MonocleState"]):
        idxs = s == data["MonocleState"].values
        ax[i].scatter(data["Time"].loc[idxs], Y.iloc[:, i].loc[idxs])
        ax[i].set_title(Y.columns[i])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
f.suptitle("Branching genes, location=1.1 indicates no branching")

# %% [markdown] scrolled=true
# # Run the BGP model
# Run script `runsyntheticData.py` to obtain a pickle file with results.
# This script can take ~10 to 20 minutes depending on your hardware.
# It performs a gene-by-gene branch model fitting.

# %% [markdown]
# # Plot BGP posterior fit
# Plot posterior fit.

# %% scrolled=true
r = pickle.load(open("syntheticdata/syntheticDataRun.p", "rb"))

# %%
r.keys()

# %% scrolled=true
# plot fit for a gene
g = 0
GPy = Y.iloc[:, g][:, None]
GPt = data["Time"].values
globalBranching = data["MonocleState"].values.astype(int)
bmode = r["Bsearch"][np.argmax(r["gpmodels"][g]["loglik"])]
print("True branching time", trueBranchingTimes[g], "BGP Maximum at b=%.2f" % bmode)
_ = bplot.PlotBGPFit(GPy, GPt, r["Bsearch"], r["gpmodels"][g])

# %% [markdown]
# We can also plot with the predictive uncertainty of the GP.
# The dashed lines are the 95% confidence intervals.

# %%
g = 0
bmode = r["Bsearch"][np.argmax(r["gpmodels"][g]["loglik"])]
pred = r["gpmodels"][g]["prediction"]  # prediction object from GP
_ = bplot.plotBranchModel(
    bmode,
    GPt,
    GPy,
    pred["xtest"],
    pred["mu"],
    pred["var"],
    r["gpmodels"][g]["Phi"],
    fPlotPhi=True,
    fColorBar=True,
    fPlotVar=True,
)

# %% [markdown]
# # Plot posterior
# Plotting the posterior alongside the true branching location.

# %%
fs, ax = plt.subplots(1, 1, figsize=(5, 5))
for g in range(G):
    bmode = r["Bsearch"][np.argmax(r["gpmodels"][g]["loglik"])]
    ax.scatter(bmode, g, s=100, color="b")  # BGP mode
    ax.scatter(trueBranchingTimes[g] + 0.05, g, s=100, color="k")  # True

# %%
