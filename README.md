# BranchedGP

BranchedGP is a package for building Branching Gaussian process models in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow). 
The model is described in the paper

["BGP: Branched Gaussian processes for identifying gene-specific branching dynamics in single cell data", 
Alexis Boukouvalas, James Hensman, Magnus Rattray, bioRxiv, 2017.](http://www.biorxiv.org/content/early/2017/08/01/166868).

This is now published in [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1440-2).
[![Build Status](https://travis-ci.org/ManchesterBioinference/BranchedGP.svg?branch=master)](https://travis-ci.org/ManchesterBioinference/BranchedGP)
[![codecov](https://codecov.io/gh/ManchesterBioinference/BranchedGP/branch/master/graph/badge.svg)](https://codecov.io/gh/ManchesterBioinference/BranchedGP)

The package builds on the [GPFlow](https://github.com/GPflow/GPflow) project which allows for
scalable inference of Gaussian processes using TensorFlow.

# Example
An example of what the model can provide is shown below.
   1. The posterior cell assignment is shown in top subpanel: each cell is assigned a probability of belonging to a  branch.
   1. In the bottom subpanel the posterior branching time is shown: the probability of branching at a particular pseudotime.
<img src="images/VAMP5_BGPAssignmentProbability.png" width="400" height="400" align="middle"/>

# Install
To install Branched GP package and all its requirements.
```
git clone https://github.com/ManchesterBioinference/BranchedGP
cd BranchedGP
pip install -r requirements.txt
python setup.py install
```

or do `pip install -e .` for a development setup where the 
the current directory is used for installation rather than copied.

# Quick start
For a quick introduction see the `notebooks/Hematopoiesis.ipynb` notebook.
Therein we demonstrate how to fit the model and compute
the log Bayes factor for two genes.

The Bayes factor in particular is calculated by calling `CalculateBranchingEvidence`
after fitting the model using `FitModel`.

This notebook should take a total of 6 minutes to run.

# Tests
To run the tests takes about 10 minutes.
```
cd BranchedGP/testing
nosetests --pdb-failures --pdb --with-timer
```


# List of notebooks
To run the notebooks
```
cd BranchedGP/notebooks
jupyter notebook
```

| File <br> name | Description | 
| --- | --- | 
| Hematopoiesis       | Application of BGP to hematopoiesis data. |
| SyntheticData       | Application of BGP to synthetic data. |
| SamplingFromTheModel| Sampling from the BGP model. |


# Comparison to monocle-BEAM

In the paper we compare the BGP model to the BEAM method proposed
in monocle 2. In ```monocle/runMonocle.R``` the R script for performing
Monocle and BEAM on the hematopoiesis data is included.
# List of python library files
| File <br> name | Description | 
| --- | --- | 
| FitBranchingModel.py | Main file for user to call BGP fit, see function FitModel | 
| pZ_construction_singleBP.py | Construct prior on assignments; use by variational code. |
| assigngp_dense.py | Variational inference code to infer function labels. |
| assigngp_denseSparse.py | Sparse inducing point variational inference code to infer function labels. |
| branch_kernParamGPflow.py | Branching kernels. Includes independent kernel as used in the overlapping mixture of GPs and a hardcoded branch kernel for testing. |
| BranchingTree.py | Code to generate branching tree. |
| VBHelperFunctions.py | Plotting code. |

# Running in a cluster
When running BranchingGP in a cluster it may be useful to constrain the number of cores used. To do this insert this code at the beginning of your script.
```
from gpflow import settings
settings.session.intra_op_parallelism_threads = NUMCORES
settings.session.inter_op_parallelism_threads = NUMCORES
```



