from jug import TaskGenerator
import os
import tensorflow as tf
import fitSampleGP
'''
Script to test jug parallelism and also grid setup (GPflow, Tensorflow)

Jug cheat sheet: https://jug.readthedocs.io/en/latest/faq.html
jug status primes.py         Status
jug execute primes.py &      Run
jug shell primes.py          Look at results using p = value(primes100)
'''

@TaskGenerator
def runSampleGPFull(seedpr):
    # Get number of cores reserved by the batch system (NSLOTS is automatically set)
    NSLOTS = os.environ.get("NSLOTS")
    if(NSLOTS is None):
        NUMCORES = 1
    else:
        NUMCORES = int(NSLOTS)
    print("Using", NUMCORES, "core(s)")
    # Create TF session using correct number of cores
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                                            intra_op_parallelism_threads=NUMCORES))
    with sess.as_default():
        # build the GPR object
        return fitSampleGP.GetSampleGPFitBranchingModel(seedpr, fTesting=fTesting)

@TaskGenerator
def runSampleGPSparse(seedpr):
    # Get number of cores reserved by the batch system (NSLOTS is automatically set)
    NSLOTS = os.environ.get("NSLOTS")
    if(NSLOTS is None):
        NUMCORES = 1
    else:
        NUMCORES = int(NSLOTS)
    print("Using", NUMCORES, "core(s)")
    # Create TF session using correct number of cores
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                                            intra_op_parallelism_threads=NUMCORES))
    with sess.as_default():
        # build the GPR object
        return fitSampleGP.GetSampleGPFitBranchingModel(seedpr, fTesting=fTesting, nsparseGP=21)


# in this configuration, full and sparse should be run on exactly the same data
fTesting = False
if(fTesting):
    NSamples = 2
else:
    NSamples = 100
runsFull = [runSampleGPFull(n) for n in range(NSamples)]
runsSpar = [runSampleGPSparse(n) for n in range(NSamples)]
