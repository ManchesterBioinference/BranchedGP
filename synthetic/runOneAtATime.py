import os
import tensorflow as tf
import pickle
import fitSampleGP
'''
Script to run single array job
'''


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
        return fitSampleGP.GetSampleGPFitBranchingModel(seedpr, fTesting=fTesting, N=N)


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
        return fitSampleGP.GetSampleGPFitBranchingModel(seedpr, fTesting=fTesting,
                                                        N=N, nsparseGP=21)


if __name__ == '__main__':
    '''
    Differences from jugscript:
    Use N=100 points
    We have also changed fitSampleGP in non-testing case to use 11 points in  B grid search
    '''
    # in this configuration, full and sparse should be run on exactly the same data
    fTesting = False
    if(fTesting):
        N = 30
    else:
        N = 100
    # Run single job at a time
    for taskId in range(1, 101):
        # Run both full and sparse versions
        print('runArrayJob: Running full branching GP with seed %g' % taskId)
        r = runSampleGPFull(taskId)
        pickle.dump(r, open("runArrayJob_Full%g.p" % taskId, "wb"))
        print('runArrayJob: Running sparse branching GP with seed %g' % taskId)
        r = runSampleGPSparse(taskId)
        pickle.dump(r, open("runArrayJob_Sparse%g.p" % taskId, "wb"))
