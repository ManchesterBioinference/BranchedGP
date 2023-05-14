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
# Branching GP Regression: Sampling from the model
# --
#
# *Alexis Boukouvalas, 2017*
#
# This notebook shows how to sample from a BGP model

# %%
import gpflow
import numpy as np
from matplotlib import pyplot as plt

from BranchedGP import BranchingTree as bt
from BranchedGP import branch_kernParamGPflow as bk

plt.style.use("ggplot")
# %matplotlib inline

# %% [markdown]
# ### Create the tree
# Specify where the branching point is

# %%
branchingPoint = 0.5
tree = bt.BinaryBranchingTree(
    0, 10, fDebug=False
)  # set to true to print debug messages
tree.add(None, 1, branchingPoint)  # single branching point
(fm, fmb) = tree.GetFunctionBranchTensor()

# %% [markdown]
# Specify where to evaluate the kernel

# %%
t = np.linspace(0.01, 1, 10)
(XForKernel, indicesBranch, Xtrue) = tree.GetFunctionIndexList(t, fReturnXtrue=True)

# %% [markdown]
# Specify the kernel and its hyperparameters
# These determine how smooth and variable the branching functions are

# %%
Bvalues = np.expand_dims(np.asarray(tree.GetBranchValues()), 1)
KbranchParam = bk.BranchKernelParam(gpflow.kernels.RBF(1), fm, b=Bvalues)
KbranchParam.kern.lengthscales = 2
KbranchParam.kern.variance = 1

# %% [markdown] scrolled=true
# Sample the kernel

# %% scrolled=true
samples = bk.SampleKernel(KbranchParam, XForKernel)

# %% [markdown]
# Plot the sample

# %%
bk.PlotSample(XForKernel, samples)

# %% [markdown] collapsed=true
# You can rerun the same code as many times as you want and get different sample paths

# %% [markdown]
# We can also sample independent functions. This is the assumption in the overlapping mixtures of GPs model (OMGP) discussed in the paper.

# %%
indKernel = bk.IndKern(gpflow.kernels.RBF(1))
samples = bk.SampleKernel(indKernel, XForKernel)
bk.PlotSample(XForKernel, samples)

# %%
