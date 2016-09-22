#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd                   # Run job from directory where submitted
#$ -V                     # Inherit environment (modulefile) settings
##### Select the required backend node
#$ -l k40

# Load modulefiles....
module load apps/gcc/tensorflow/0.9.0rc0-py27-gpu
module load tools/env/proxy

# export to load correct mkl library - see https://github.com/BVLC/caffe/issues/3884

# Run whatever commands you want
export runNode='GPU'
python $1

