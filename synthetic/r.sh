#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd                   # Run job from directory where submitted
#$ -V                     # Inherit environment (modulefile) settings
# Load modulefiles....
module load apps/gcc/tensorflow/0.10.0-py34-cpu
module load tools/env/proxy
# Run whatever commands you want
# exec jug execute $1
python $1
