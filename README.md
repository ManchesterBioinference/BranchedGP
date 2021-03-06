# BranchedGP

BranchedGP is a package for building Branching Gaussian process models in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow).
You can install it via `pip install BranchedGP`.

The model is described in the paper

["BGP: Branched Gaussian processes for identifying gene-specific branching dynamics in single cell data",
Alexis Boukouvalas, James Hensman, Magnus Rattray, bioRxiv, 2017.](http://www.biorxiv.org/content/early/2017/08/01/166868).

This is now published in [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1440-2).
[![CI](https://github.com/ManchesterBioinference/BranchedGP/workflows/CI/badge.svg)](https://github.com/ManchesterBioinference/BranchedGP/workflows/CI)
[![codecov](https://codecov.io/gh/ManchesterBioinference/BranchedGP/branch/master/graph/badge.svg)](https://codecov.io/gh/ManchesterBioinference/BranchedGP)

# Example
An example of what the model can provide is shown below.
   1. The posterior cell assignment is shown in top subpanel: each cell is assigned a probability of belonging to a  branch.
   1. In the bottom subpanel the posterior branching time is shown: the probability of branching at a particular pseudotime.
<img src="images/VAMP5_BGPAssignmentProbability.png" width="400" height="400" align="middle"/>


# Quick start
For a quick introduction see the `notebooks/Hematopoiesis.ipynb` notebook.
Therein we demonstrate how to fit the model and compute
the log Bayes factor for two genes.

The Bayes factor in particular is calculated by calling `CalculateBranchingEvidence`
after fitting the model using `FitModel`.

This notebook should take a total of 6 minutes to run.

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


# Development setup

This project requires Python3.7 or earlier (TensorFlow 1 requirement).
Create a virtual environment, activate it and run `make install`.

## Common tasks

* Tests: `make test`
* Install dependencies (into an active virtual environment): `make install`
* Format code: `make format`
* Run a jupyter notebook server: `make jupyter_server`

# Contributing

We welcome any and all contributions to the BranchedGP repo.
Feel free to create issues or PRs into the repo and someone will
take a look and review.

## Notebooks

We use [Jupytext](https://github.com/mwouts/jupytext) to help version
Jupyter notebooks.
Each notebook corresponds to a Python script, which is easy to review.
See also the Jupytext documentation on
[paired notebooks](https://jupytext.readthedocs.io/en/latest/#paired-notebooks).

Note that Jupytext should be automatically installed in
your virtual environment if you follow the instructions above.

### Updating an existing notebook

We want our notebooks to always work.
Therefore, before committing any changes to a notebook,
we ask contributors to re-run the notebook from scratch.

The Jupytext extension should automatically sync the notebook
to the paired script.
If you're unsure, you can always check
via `make check_notebooks_synced`
and manually run `make sync_notebooks` if needed.

### Adding a new notebook

Follow your usual procedure, but run `make pair_notebooks` afterwards.
This will produce the paired script
(or notebook if you're starting from a script).
Commit both the notebook as well as the paired notebook.

### Syncing notebooks

If Jupyter shows you a warning about the notebook being
out of sync with the master script, run `make sync_notebooks`.

## Formatting code

We automatically check that all contributions are formatted
according to the recommendations by
[black](https://black.readthedocs.io/en/stable/) and
[isort](https://pycqa.github.io/isort/).
If your changes fail these checks, all you need to do is run
`make format` and commit the changes.

## Static checks

We automatically check our code conforms to the coding standards
enforced by [flake8](https://flake8.pycqa.org/en/latest/manpage.html) and 
[MyPy](http://mypy-lang.org/).
You can check if your changes conform with these checks via `make static_checks`.