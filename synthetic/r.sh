#!/bin/bash -l
#$ -S /bin/bash
#$ -cwd                   # Run job from directory where submitted
#$ -V                     # Inherit environment (modulefile) settings
# Load modulefiles....
# Running jobs
module load tools/env/proxy
module load tools/gcc/git/2.8.2
module load apps/binapps/anaconda/3/4.1.1
export PATH=/opt/gridware/apps/gcc/tensorflow/fixes:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages
# Run 
LD_PRELOAD="/usr/lib64/librt.so:/opt/gridware/apps/gcc/tensorflow/fixes/stubs/mylibc.so:/opt/gridware/compilers/gcc/4.8.2/lib64/libstdc++.so.6" /opt/gridware/apps/binapps/anaconda/3/2.3.0/bin/python "$@" $1
