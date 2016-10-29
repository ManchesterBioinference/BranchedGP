#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd                   # Run job from directory where submitted
#$ -V                     # Inherit environment (modulefile) settings
#$ -pe smp.pe 16          # Number of cores on a single compute node. Can be 2-24.

# Load modulefiles....
module load apps/gcc/tensorflow/0.10.0-py34-cpu
module load tools/env/proxy

# Run whatever commands you want
exec jug execute $1

