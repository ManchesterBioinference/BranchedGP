# BranchedGP

BranchedGP is a package for building Branching Gaussian process models in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow). 
It has been created by Alexis Boukouvalas, James Hensman and Magnus Rattray. 

[![Build Status](https://travis-ci.org/ManchesterBioinference/BranchedGP.svg?branch=master)](https://travis-ci.org/ManchesterBioinference/BranchedGP)
[![codecov](https://codecov.io/gh/ManchesterBioinference/BranchedGP/branch/master/graph/badge.svg)](https://codecov.io/gh/ManchesterBioinference/BranchedGP)

# Install
If you have any problems with installation see the script at the bottom of the page for a detailed setup guide from a branch new python environment. 

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
| Hematopoiesis.ipynb | Application of BGP to hematopoiesis data. |

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


=======
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


