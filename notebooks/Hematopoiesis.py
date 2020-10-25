# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Branching GP Regression on hematopoietic data
# --
#
# *Alexis Boukouvalas, 2017*
#
# **Note:** this notebook is automatically generated by [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html), see the REAMDME for instructions on working with it.
#
# test change
#
# Branching GP regression with Gaussian noise on the hematopoiesis data described in the paper "BGP: Gaussian processes for identifying branching dynamics in single cell data".
#
# This notebook shows how to build a BGP model and plot the posterior model fit and posterior branching times.

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline

# %% [markdown]
# ### Read the hematopoiesis data. This has been simplified to a small subset of 23 genes found to be branching.
# We have also performed Monocle2 (version 2.1) - DDRTree on this data. The results loaded include the Monocle estimated pseudotime, branching assignment (state) and the DDRTree latent dimensions.

# %%
Y = pd.read_csv('singlecelldata/hematoData.csv', index_col=[0])
monocle = pd.read_csv('singlecelldata/hematoMonocle.csv', index_col=[0])

# %%
Y.head()

# %%
monocle.head()

# %%
# Plot Monocle DDRTree space
genelist = ['FLT3','KLF1','MPO']
f, ax = plt.subplots(1, len(genelist), figsize=(10, 5), sharex=True, sharey=True)
for ig, g in enumerate(genelist):
        y = Y[g].values
        yt = np.log(1+y/y.max())
        yt = yt/yt.max()
        h = ax[ig].scatter(monocle['DDRTreeDim1'], monocle['DDRTreeDim2'],
                       c=yt, s=50, alpha=1.0, vmin=0, vmax=1)
        ax[ig].set_title(g)

# %%
import BranchedGP, time
def PlotGene(label, X, Y, s=3, alpha=1.0, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for li in np.unique(label):
        idxN = (label == li).flatten()
        ax.scatter(X[idxN], Y[idxN], s=s, alpha=alpha, label=int(np.round(li)))
    return fig, ax


# %% [markdown]
# ### Fit BGP model
# Notice the cell assignment uncertainty is higher for cells close to the branching point.
#

# %%
def FitGene(g, ns=20): # for quick results subsample data
    t = time.time()
    Bsearch = list(np.linspace(0.05, 0.95, 5)) + [1.1]  # set of candidate branching points
    GPy = (Y[g].iloc[::ns].values - Y[g].iloc[::ns].values.mean())[:, None]  # remove mean from gene expression data
    GPt = monocle['StretchedPseudotime'].values[::ns]
    globalBranching = monocle['State'].values[::ns].astype(int)
    d = BranchedGP.FitBranchingModel.FitModel(Bsearch, GPt, GPy, globalBranching)
    print(g, 'BGP inference completed in %.1f seconds.' %  (time.time()-t))
    # plot BGP
    fig,ax=BranchedGP.VBHelperFunctions.PlotBGPFit(GPy, GPt, Bsearch, d, figsize=(10,10))
    # overplot data
    f, a=PlotGene(monocle['State'].values, monocle['StretchedPseudotime'].values, Y[g].values-Y[g].iloc[::ns].values.mean(),
                  ax=ax[0], s=10, alpha=0.5)
    # Calculate Bayes factor of branching vs non-branching
    bf = BranchedGP.VBHelperFunctions.CalculateBranchingEvidence(d)['logBayesFactor']

    fig.suptitle('%s log Bayes factor of branching %.1f' % (g, bf))
    return d, fig, ax
d, fig, ax = FitGene('MPO')

# %%
d_c, fig_c, ax_c = FitGene('CTSG')

# %%

a = 1