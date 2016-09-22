# GPBranch

GPBranch is a package for building Gaussian process models in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow). 
It has been created by Alexis Boukouvalas, James Hensman and Magnus Rattray. 

# Install

## 1) Install Tensorflow fork and GPflow.
See [GPFlow](https://github.com/GPflow/GPflow) page for instructions.


# List of notebooks
| File <br> name | Description | 
| --- | --- | 
| OptimizingBranching* | Demonstrate how a single branching can be identified on a synthetic GP sample data where the ground truth is known. |
| BranchingGPTutorial | Toy example demonstrating the library interface. |

# List of python library files
| File <br> name | Description | 
| --- | --- | 
| pZ_construction_singleBP.py | Construct prior on assignments; use by variational code. |
| assigngp_dense.py | Variational inference code to infer function labels. |
| branch_kernParamGPflow.py | Branching kernels. Includes independent kernel as used in the overlapping mixture of GPs and a hardcoded branch kernel for testing. |
| BranchingTree.py | Code to generate branching tree. |
| AssignGPGibbsSingleLoop.py | MAP/Gibbs code to that infers function labels. | 
| tests.py | nosetests |
