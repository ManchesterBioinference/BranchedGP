import tensorflow as tf
import numpy as np
import BranchingTree as bt
import GPflow
import branch_kernParamGPflow  as bk

def testParamKernel():
    tf.reset_default_graph()
    tree = bt.BinaryBranchingTree(0,10,fDebug=False) # set to true to print debug messages
    tree.add(None,1,5) # single branching point
    tree.add(1,2,7) # single branching point
    (fm, fmb) = tree.GetFunctionBranchTensor()
    #print fmb
    
    tree.printTree()  
    print fm
    #print fmb
    t = np.linspace(0.01,10,100)
    (XForKernel, indicesBranch,Xtrue) = tree.GetFunctionIndexList(t,fReturnXtrue=True)
    # GP flow kernel
    D = 2
    Xs = tf.placeholder("float64", shape=[None, D])
    Ys = tf.placeholder("float64", shape=[None, D])
    
    parameterVector = tf.placeholder("float64") # Needed by GPflow
    kernInsideBranch = GPflow.kernels.RBF(D-1) # 1 dimension for labels
               
    
    # Hardcoded kernel
    Kbranch = bk.BranchKernelHardcoded( kernInsideBranch, fmb)
    Kbranch.kern.lengthscales = 2
    Kbranch.kern.variance = 1
    with Kbranch.tf_mode():
        Kbranch.make_tf_array(parameterVector)
        Kbranch_s = Kbranch.K(Xs,Ys)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())  # Not sure this is needed
        Kbranch_values = sess.run(Kbranch_s, feed_dict={parameterVector:Kbranch.get_free_state(), Xs:Xtrue, Ys:Xtrue.copy() })
    
    
    # Param kernel
    BvaluesInit = np.ones((2,1)) # initial values
    
    KbranchParamGPflow = bk.BranchKernelParam( kernInsideBranch, fm, BvInitial=BvaluesInit)
    Bvalues = np.expand_dims(np.asarray(tree.GetBranchValues()),1)
    print 'Initialised Kernel'
    print KbranchParamGPflow
    KbranchParamGPflow.Bv = Bvalues
    print 'After update'
    print KbranchParamGPflow
    print 'free state'
    KbranchParamGPflow.get_free_state()
    KbranchParamGPflow.kern.lengthscales = 2
    KbranchParamGPflow.kern.variance = 1
    with KbranchParamGPflow.tf_mode():
        KbranchParamGPflow.make_tf_array(parameterVector)
        KbranchParam_s = KbranchParamGPflow.K(Xs,Ys)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())  # Not sure this is needed
        KbranchParamGPflow_values = sess.run(KbranchParam_s, feed_dict={parameterVector:KbranchParamGPflow.get_free_state(), Xs:Xtrue, Ys:Xtrue.copy() }) 
    
    print np.allclose(Kbranch_values,KbranchParamGPflow_values)
    assert np.allclose(Kbranch_values,KbranchParamGPflow_values)
    
def testpZConstruction():
    tf.reset_default_graph()
    import pZ_construction_singleBP
    
    #X = np.random.rand(10, 1)
    N = 4
    X = np.linspace(0, 1, N, dtype=float)[:,None]
    X = np.sort(X, 0)
    BP = tf.placeholder(tf.float64, shape=[])
    pZ = tf.Session().run(pZ_construction_singleBP.make_matrix(X, BP), feed_dict={BP: 0.5})

    pZShouldBe = np.zeros((N,3*N),dtype=float)
    pZShouldBe[0,0] = 1
    pZShouldBe[1,3] = 1
    pZShouldBe[2,7:9] = 0.5
    pZShouldBe[3,10:] = 0.5
        
    print np.allclose(pZShouldBe,pZ)
    assert np.allclose(pZShouldBe,pZ, atol=1.e-5), 'they are not close pZ=' +str(pZ) 
    
def testSingleBranchPointInference():
    tf.reset_default_graph()
    import AssignGPGibbsSingleLoop
    import assigngp_dense
    seed = 43
    
    np.random.seed(seed=seed) # easy peasy reproducibeasy
    tf.set_random_seed(seed)
    
    # Create data set - see BranchingGPTutorial for plots
    N = 20
    t = np.linspace(0,1,N)
    print t
    Bv = 0.5
    Y = np.zeros( (N,1) )
    idx = np.nonzero(t > 0.5)[0]
    idxA = idx[::2]
    idxB = idx[1::2]
    print idx
    print idxA
    print idxB
    Y[idxA,0] = 2*t[idxA]
    Y[idxB,0] = -2*t[idxB]
    
    # Initialise model
    # kernel
    Bvalue = np.ones((1,1)) * 0.5 
    BvalueInit = np.ones((1,1)) * 999 
    print Bvalue
    
    tree = bt.BinaryBranchingTree(0,1,fDebug=False)
    tree.add(None,1,Bvalue) 
    (fm, _) = tree.GetFunctionBranchTensor()
    
    print fm
    
    Kbranch = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, BvInitial=Bvalue) + GPflow.kernels.White(1)
    varianceLik = .001
    Kbranch.white.variance = varianceLik # controls the discontinuity magnitude, the gap at the branching point
    Kbranch.branchkernelparam.kern.lengthscales = 5
    Kbranch.branchkernelparam.kern.variance = 1
    Kbranch.branchkernelparam.Bv = Bvalue
    Kbranch.branchkernelparam.Bv.fixed = True
    
    m = AssignGPGibbsSingleLoop.AssignGPGibbsFast(t, Y, Kbranch)
    m.likelihood.variance = 1
    m.CompileAssignmentProbability(fDebug=True,fMAP=True) 
    
    print m.XExpanded.shape
    print len(m.indices)
    #print XSampleGeneral
    
    mV = assigngp_dense.AssignGP(t, m.XExpanded, Y, Kbranch)
    mV.likelihood.variance = varianceLik
    
    mV._compile() # creates objective function
    
    randomAssignment = AssignGPGibbsSingleLoop.GetRandomInit(t,Bvalue,m.indices)
    print randomAssignment
    print m.XExpanded[randomAssignment,:]

    # MAP inference    
    numMCMCsteps = 10
    #bestAssignment = list(randomAssignment)
    #chainState = np.ones(10)
    (chainState, bestAssignment) = m.InferenceGibbsMAP(fDebug=True,maximumNumberOfSteps=numMCMCsteps,\
                                                                 startingAssignment=randomAssignment)
    
    allocationOfPoints = m.XExpanded[bestAssignment,:]
    print 'MAP Allocation of points'
    print allocationOfPoints
    print 'MAP Allocation of points function A'
    print allocationOfPoints[idxA,:]
    print 'MAP Allocation of points function B'
    print allocationOfPoints[idxB,:]
    assert np.unique(allocationOfPoints[idxA,1]) == 2 # seed fixed so should not - all should be same label 
    assert np.unique(allocationOfPoints[idxB,1]) == 3
    
    # VB code
    # Variational bound - no recomputing
    print 'Variational kernel branch value ' + str(mV.kern.branchkernelparam.Bv._array.flatten())
    # Set state for assignments
    phiInitial =  np.zeros((N,3*N))
    for i,n in enumerate(bestAssignment):
        phiInitial[i,n] = 10
    mV.logPhi= phiInitial
    # Could also optimize!
    VBbound = mV._objective(mV.get_free_state())[0] # this is -log of bound
    print VBbound
    
    print mV
    
    df=tree.GetFunctionDomains()

    # Prediction 
    # UNDONE - this just exercises code, no validation of results done
    for b in range(1, df.shape[0]+1):
        ttest = np.linspace(df[b-1][0], df[b-1][1], 10)[:,None]
        Xtest = np.hstack((ttest, ttest*0+b))
        
        mu, var = mV.predict_f(Xtest)
        
        assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
        assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)

    
    print 'test serialisation of VB object...'
    print mV 
    strSaveFile='testSingleBranchPointInference'
    np.save(strSaveFile + 'VBmodel',mV.get_free_state())
    KbranchNew = bk.BranchKernelParam(GPflow.kernels.Matern32(1), fm, BvInitial=BvalueInit) + GPflow.kernels.White(1)
    KbranchNew.branchkernelparam.Bv.fixed = True # not part of the free state
    mNew = AssignGPGibbsSingleLoop.AssignGPGibbsFast(t, Y, KbranchNew)
    mVNew = assigngp_dense.AssignGP(t, mNew.XExpanded, Y, KbranchNew)

    assert np.allclose( mV.get_free_state(), mVNew.get_free_state() ) == False, 'No assignment yet.'
    mVNew.set_state( np.load(strSaveFile + 'VBmodel.npy'))
    # note this will not copy and check the fixed parameters such as branching value
    print 'Loading object has state...'
    print mVNew
    assert np.allclose( mV.get_free_state(), mVNew.get_free_state() ) == True, 'Free state should match.'
    
    print 'Finished testSingleBranchPointInference test.'

def TestModelSelection():
    tf.reset_default_graph()
    import mouseQPCRModelSelection
    pt,Y = mouseQPCRModelSelection.LoadMouseQPCRData(subsetSelection=3)
    m,mV = mouseQPCRModelSelection.InitModels(pt,Y)
    logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m,mV,Bpossible=np.array([25.]), \
        strSaveState='nosetest', \
        fSoftVBAssignment=True, fOptimizeHyperparameters = False, fReestimateMAPZ=True,\
        numMAPsteps = 10)
    print logVBBound, logLike
    logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m,mV,Bpossible=np.array([25.]), \
        strSaveState='nosetest', \
        fSoftVBAssignment=False, fOptimizeHyperparameters = False, fReestimateMAPZ=True,\
        numMAPsteps = 10)
    logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m,mV,Bpossible=np.array([25.]), \
        strSaveState='nosetest', \
        fSoftVBAssignment=False, fOptimizeHyperparameters = True, fReestimateMAPZ=True,\
        numMAPsteps = 10)  
    logVBBound, logLike = mouseQPCRModelSelection.DoModelSelectionRuns(m,mV,Bpossible=np.array([25.]), \
        strSaveState='nosetest', \
        fSoftVBAssignment=True, fOptimizeHyperparameters = True, fReestimateMAPZ=True,\
        numMAPsteps = 10)      
    
if __name__ == '__main__':
    #testSingleBranchPointInference()
    TestModelSelection()
