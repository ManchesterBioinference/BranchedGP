# GPBranch

GPBranch is a package for building Gaussian process models in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow). 
It has been created by Alexis Boukouvalas, James Hensman and Magnus Rattray. 

# Install

## 1) Install Tensorflow fork and GPflow.

   - Install tensorflow
```pip install tensorflow```
   - Install GPflow
    ```git clone https://github.com/GPflow/GPflow.git
    
    cd GPflow
    
    python setup.py install
    ```
    
See [GPFlow](https://github.com/GPflow/GPflow) page for more detailed instructions.


# List of notebooks
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



