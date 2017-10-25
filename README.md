# BranchedGP

BranchedGP is a package for building Branching Gaussian process models in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow). 
The model is described in the paper

["BGP: Branched Gaussian processes for identifying gene-specific branching dynamics in single cell data", 
Alexis Boukouvalas, James Hensman, Magnus Rattray, bioRxiv, 2017.](http://www.biorxiv.org/content/early/2017/08/01/166868).

[![Build Status](https://travis-ci.org/ManchesterBioinference/BranchedGP.svg?branch=master)](https://travis-ci.org/ManchesterBioinference/BranchedGP)
[![codecov](https://codecov.io/gh/ManchesterBioinference/BranchedGP/branch/master/graph/badge.svg)](https://codecov.io/gh/ManchesterBioinference/BranchedGP)

# Example
An example of what the model can provide is shown below.
   1. The posterior cell assignment is shown in top subpanel: each cell is assigned a probability of belonging to a  branch.
   1. In the bottom subpanel the posterior branching time is shown: the probability of branching at a particular pseudotime.
<img src="images/VAMP5_BGPAssignmentProbability.png" width="400" height="400" align="middle"/>

# Install
If you have any problems with installation see the script at the bottom of the page for a detailed setup guide from a new python environment. 

   - Install tensorflow
```
pip install tensorflow
```
   - Install GPflow
```
git clone https://github.com/GPflow/GPflow.git
cd GPflow    
python setup.py install
cd
```
    
See [GPFlow](https://github.com/GPflow/GPflow) page for more detailed instructions.

   - Installed Branched GP package
```
git clone https://github.com/ManchesterBioinference/BranchedGP
cd BranchedGP
python setup.py install
cd
```

# Tests
To run the tests should takes < 3min.
```
pip install nose
pip install nose-timer
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
from GPflow import settings
settings.session.intra_op_parallelism_threads = NUMCORES
settings.session.inter_op_parallelism_threads = NUMCORES
```

# Python 2.7 Compatibility testing
This gives a detailed breakdown of all the steps requires. It sets up a conda environment but direct installation is also possible.
```
conda create --yes -n condaenv2test python=2.7
conda install --yes -n condaenv2test pip
conda install --yes -n condaenv2test scipy
conda install --yes -n condaenv2test nose
source activate condaenv2test
pip install tensorflow
pip install pandas
pip install nose-timer
pip install matplotlib
pip install nbformat
pip install nbconvert
pip install jupyter_client
mkdir ~/python2test
cd ~/python2test
git clone https://github.com/GPflow/GPflow.git
cd GPflow    
python setup.py install
cd ~/python2test
git clone https://github.com/ManchesterBioinference/BranchedGP
cd BranchedGP
python setup.py install
cd ~/python2test
cd BranchedGP/testing
conda install --yes nb_conda
conda install --yes ipykernel
nosetests --pdb-failures --pdb --with-timer
source deactivate
```


