# coding: utf-8

import GPflow
import numpy as np
import tensorflow as tf
import sys


def GetRandomInit(t, Bv, indicesBranchGeneral):
    randomAssignment = []
    for i, ind in enumerate(indicesBranchGeneral):
        if(t[i] <= Bv):
            # root
            # print 'Set to ind[0] x= ' + str(t[i])
            randomAssignment.append(ind[0])
        else:
            # one of two branches
            # print 'Set to ind[1,2] x= ' + str(t[i])
            # print ind
            # print ind[1:]
            randomAssignment.append(np.random.choice(ind[1:]))
    return randomAssignment


def sample_assignment(logp):
    """
    logp is a vector of log probabilities
    """
    return np.argmax(logp - np.log(-np.log(np.random.rand(logp.size))))


def sample_assignment_tf(logp):
    """
    logp is a vector of log probabilities
    see https://github.com/tensorflow/tensorflow/issues/456
    """
    matrix_U = tf.random_uniform(tf.shape(logp), minval=0, maxval=1, dtype=tf.float64)
    p = tf.log(tf.neg(tf.log(matrix_U)))
    final_number = tf.argmax(tf.sub(logp, p), 0)
    return final_number


def checkIndices(indices, XForKernel, Xtrue):
    '''# Check indices produces XForKernel[] -> same as X (N size) and in addition all rows are taken from X'''
    assert Xtrue.shape[1] == XForKernel.shape[1]  # same number of columns
    Xt = Xtrue[:, 0]  # first column
    Xr = np.zeros(Xt.shape)
    for i, ind in enumerate(indices):
        Xr[i] = XForKernel[ind[0], 0]  # take the first entry
    assert np.allclose(Xr, Xt)


def CreateAssignmentVector(indices, functionIndex, XForKernel, Xtrue):
    ''' Create assignment vector by looking at functionIndex, which specifies which function generated sample. '''
    assert len(indices) == len(functionIndex)
    checkIndices(indices, XForKernel, Xtrue)
    assignmentCreated = np.zeros(len(indices), int)

    for i, ind in enumerate(indices):
        f = functionIndex[i]  # must be 1 based as this is what we use in XForKernel
        assert f > 0, 'Must be 1-based function counting ' + str(f)
        print(f)
        whichind = np.flatnonzero(XForKernel[ind, 1] == f)
        assert len(whichind) == 1, 'Index big ' + str(whichind)
        assignmentCreated[i] = ind[whichind]

    return assignmentCreated


def CreateTrueAssignmentVector(indices, XForKernel, Xtrue):
    ''' Return assingment vector reflecting true allocation '''
    checkIndices(indices, XForKernel, Xtrue)
    trueAssignment = np.zeros(Xtrue.shape[0], int)
    for i, ind in enumerate(indices):
        whichind = np.flatnonzero(XForKernel[ind, 1] == Xtrue[i, 1])  # which entry ind is correct?
        assert len(whichind) == 1
        trueAssignment[i] = ind[whichind]

    return trueAssignment


def checkAssignmentVector(indices, assignCheck, Xtrue):
    assert len(indices) == Xtrue.shape[0]

    # Check assignment vector
    for i, ia in enumerate(assignCheck):
        # check corresponding index has this element
        if(ia not in indices[i]):
            print('Assignment vector ' + str(assignCheck))
            print('Index ' + str(indices[i]))
            strError = 'checkAssignmentVector: assignment vector: Element ' + \
                str(i) + ' has value ' + str(ia) + ' but should be one of ' + str(indices[i])
            raise NameError(strError)
            assert False, strError


def GetFunctionIndexListGeneral(Xin):
    ''' Function to return index list  and input array X repeated as many time as each possible function '''
    # limited to one dimensional X for now!
    assert Xin.shape[0] == np.size(Xin)
    indicesBranch = []

    XSample = np.zeros((Xin.shape[0], 2), dtype=float)
    Xnew = []
    inew = 0
    functionList = list(range(1, 4))  # can be assigned to any of root or subbranches, one based counting

    for ix, x in enumerate(Xin):
        XSample[ix, 0] = Xin[ix]
        XSample[ix, 1] = np.random.choice(functionList)
        # print str(ix) + ' ' + str(x) + ' f=' + str(functionList) + ' ' +  str(XSample[ix,1])
        idx = []
        for f in functionList:
            Xnew.append([x, f])  # could have 1 or 0 based function list - does kernel care?
            idx.append(inew)
            inew = inew + 1
        indicesBranch.append(idx)
    Xnewa = np.array(Xnew)
    return (Xnewa, indicesBranch, XSample)


class AssignGPGibbsFast(GPflow.gpr.GPR):

    def __init__(self, X, Y, kern, Z=None):
        GPflow.gpr.GPR.__init__(self, X[:, None], Y, kern)  # ugly hack to make sure they have N data
        # randomly assign to start.
        assert X.shape[0] == Y.shape[0], "Must have enlarged X matrix. Dont use this function with only 1 function!  size of X is %g, Y is %g" % (X.shape[
                                                                                                                                                  0], Y.shape[0])
        assert X.ndim == 1, 'Must have one-dimensional input'

        (XForKernelGeneral, indicesBranchGeneral, _) = GetFunctionIndexListGeneral(X)

        self.indices = indicesBranchGeneral
        self.KChol = None  # until compilation
        self.XExpanded = XForKernelGeneral
        self.t = X
        self.X = None  # includes
        self.ZExpanded = None
        if(Z is not None):
            assert Z.ndim == 1, 'Must have one dimensional Z'
            (ZExpanded, _, _) = GetFunctionIndexListGeneral(Z)
            self.ZExpanded = ZExpanded  # inducing points for sparse GP, optional. Same format as XExpanded

    def OptimizeK(self, assignment, fDebug=False, max_iters=1000):
        checkAssignmentVector(self.indices, assignment, self.XExpanded[assignment, :])  # check we haven't messed up

        self.kern.branchkernelparam.Bv.fixed = True  # dont optimize here as it leads to bad solutions
        self.X = self.XExpanded[assignment, :]
        self._compile()  # creates objective function
        self.optimize(max_iters=max_iters)

        # compile KChol on expanded kernel given current kernel hyperparameters
        self.CompileAssignmentProbability(fDebug=fDebug)

    def CompileAssignmentProbability(self, fDebug=False, fMAP=True):
        """
        modified version of model _compile
        Compiles the assignment probability function "self._assignProbFunction" for likelihood evaluation
        """
        self.X = None
        # Precompile covariance kernel K
        self.make_tf_array(self._free_vars)

        with self.tf_mode():
            M = self.XExpanded.shape[0]
            if self.ZExpanded is None:  # No sparse GP
                with tf.name_scope('CholeskyPrecomputation'):
                    K = self.kern.K(self.XExpanded) + GPflow.tf_hacks.eye(M) * 1e-6
                    L = tf.cholesky(K)
                    self.KChol = self._session.run(L, feed_dict={self._free_vars: self.get_free_state()})
            else:  # sparse GP
                M = self.ZExpanded.shape[0]
                with tf.name_scope('CholeskyPrecomputationSparseGP'):
                    K = self.kern.K(self.ZExpanded) + GPflow.tf_hacks.eye(M) * 1e-6
                    L = tf.cholesky(K)
                    Kuf = self.kern.K(self.ZExpanded, self.XExpanded)
                    Kdiag = self.kern.Kdiag(self.XExpanded)
                    self.KChol, self.Kuf, self.Kdiag = self._session.run(
                        [L, Kuf, Kdiag], feed_dict={self._free_vars: self.get_free_state()})
        if(fDebug):
            print('Compiling assignment probability tensorflow function... KChol ' + str(self.KChol.shape))
            sys.stdout.flush()
        # build tensorflow functions for computing the likelihood and predictions

        # Compile update expression
        NumberOfOverlappinFunctions = 2  # TODO hardcoded #self.tree.GetNumberOfBranchPts()+1
        if(fDebug):
            print('NumberOfOverlappinFunctions ' + str(NumberOfOverlappinFunctions))

        N = self.Y.shape[0]

        # Placeholders inputs
        assignments_tf = tf.placeholder(tf.int32)
        idxTF = tf.placeholder(tf.int32, name='idxTF')
        pointToUpdate = tf.placeholder(tf.int32, name='iPointToUpdate', shape=[1])
        # Expression to compute MAP or Gibbs
        probVector = tf.Variable(
            np.zeros(
                NumberOfOverlappinFunctions,
                dtype=float) * np.NaN,
            tf.float64,
            name='probVector')
        probVector = probVector.assign(probVector.initialized_value())

        for f in range(NumberOfOverlappinFunctions):
            # k = idx[f]
            with tf.name_scope('ComputeCoin_f_' + str(f)):
                # k'th function
                # bestAssignment[i] = k
                assignments_ind_tf = tf.Variable(
                    tf.zeros(
                        [N],
                        tf.int32),
                    name='assind_' +
                    str(f))  # this better be int32
                assignments_ind_tf = assignments_ind_tf.assign(assignments_tf)  # initialise

                # bestAssignment[i] = k
                assignments_ind_tf = tf.scatter_update(
                    assignments_ind_tf, pointToUpdate, tf.expand_dims(
                        idxTF[f], 0), name='initialass_' + str(f))

                if(fDebug):
                    assignments_ind_tf = tf.Print(assignments_ind_tf,
                                                  [pointToUpdate,
                                                   tf.shape(assignments_ind_tf),
                                                      assignments_ind_tf],
                                                  message='point - assignments_after update=',
                                                  name='assignments_ind_tfddebug',
                                                  summarize=10)  # will print message

                # lik = tf.constant([2.],tf.float64) # for testing
                with self.tf_mode():
                    lik = self.build_likelihoodAssignment(assignments_ind_tf, fDebug=False)
                if(fDebug):
                    lik = tf.Print(lik, [pointToUpdate, tf.shape(lik), lik], message='point - lik=',
                                   name='likeddebug', summarize=10)  # will print message

                # without locking but ok to do in parallel but ik unique
                probVector = tf.scatter_update(probVector, [f], lik, name='probVector' + str(f))
                tf.histogram_summary('probvector', tf.squeeze(probVector))
        # print probability vector of all assignments
        if(fDebug):
            probVector = tf.Print(probVector,
                                  [pointToUpdate,
                                   tf.shape(probVector),
                                      probVector],
                                  message='point - probVector=',
                                  name='probVectordebug',
                                  summarize=5)  # will print message

        if(fMAP):
            with tf.name_scope('MAPdecision'):
                # MAP decision
                si = tf.arg_max(probVector, 0, name='argmax')  # MAP
                si = tf.expand_dims(si, 0, name='MAPexpanddims')
                if(fDebug):
                    si = tf.Print(si, [pointToUpdate, tf.shape(si), si], message='siMAP point=',
                                  name='siMAPdebug', summarize=3)  # will print message
                lastlik = tf.slice(probVector, si, np.asarray([1], int), name='lastlik')

                if(fDebug):
                    tf.scalar_summary("LikelihoodWinning", tf.squeeze(lastlik))  # show the winning likelihood
                    tf.merge_all_summaries()
                    tf.train.SummaryWriter("logs", self._session.graph)
                init = tf.initialize_all_variables()

                def assignmentProbabilityMAP(x, z, idx, iP):
                    self._session.run(init)
                    # return index of winner, like of winner, all likelihoods
                    return self._session.run((si, lastlik, probVector), feed_dict={
                                             self._free_vars: x, assignments_tf: z, idxTF: idx, pointToUpdate: iP})
                    # print 'Summary x%g=%s'%(iP[0], result[0])
                    #writer.add_summary(result[0], iP[0])
                    # return result[1:]
                    # return result
                self._assignProbFunction = assignmentProbabilityMAP
                self.fMAP = True
        else:
            with tf.name_scope('GibbsDecision'):
                # Gibbs sample
                si = sample_assignment_tf(probVector)
                si = tf.expand_dims(si, 0, name='Gibbsexpanddims')
                if(fDebug):
                    si = tf.Print(si, [pointToUpdate, tf.shape(si), si], message='siGibbs point=',
                                  name='siGibbsdebug', summarize=3)  # will print message
                lastlik = tf.slice(probVector, si, np.asarray([1], int), name='lastlik')
                init = tf.initialize_all_variables()

                def assignmentProbabilityGibbs(x, z, idx, iP):
                    self._session.run(init)
                    return self._session.run((si, lastlik, probVector), feed_dict={
                                             self._free_vars: x, assignments_tf: z, idxTF: idx, pointToUpdate: [iP]})
                self._assignProbFunction = assignmentProbabilityGibbs
                self.fMAP = False

    def InferenceGibbsMAP(self, fReturnAssignmentHistory=False, maximumNumberOfSteps=1,
                          startingAssignment=None, fDebug=False, tol=1e-2):
        ''' MAP Gibbs MCMC. Could restart algorithm by passing in last assignment vector. Gives point estimate at convergence'''
        self.X = None  # to ensure kernel hyperparameters are not optimized on old X
        N = len(self.indices)  # true number of datapoints (not exploded)
        chainState = []

        Bv = self.kern.branchkernelparam.Bv._array.flatten()  # we hardcode location of branching kernel, can we do better?
        assert Bv.size == 1, 'Only one dimensional one B currently supported'
        if(self.fMAP):
            print('Performing MAP inference with B=' + str(Bv))
        else:
            print('Performing Gibbs inference with B=' + str(Bv))

        if(startingAssignment is None):
            # randomly assign to start.
            bestAssignment = GetRandomInit(self.t, Bv, self.indices)
        else:
            checkAssignmentVector(
                self.indices,
                startingAssignment,
                self.XExpanded[
                    startingAssignment,
                    :])  # check we haven't messed up
            bestAssignment = list(startingAssignment)  # make a copy

        # If we change B, we should recompile!
        assignmentHistory = []
        condProbs = []
        assignmentHistory.append(bestAssignment)
        for c in range(maximumNumberOfSteps):
            for i in range(N):  # could randomly permute data
                ind = self.indices[i]  # which entries this point is represented from
                assert len(ind) == 3, 'Must have 3 possible functions, root and two branches. Have ' + str(ind)
                if(self.t[i] > Bv):
                    # We have a choice of branching functions
                    if(fDebug):
                        print('Processing point ' + str(i) + ' with inds ' + str(ind))
                    (si, lastlik, sliceOut) = self._assignProbFunction(
                        self.get_free_state(), bestAssignment, ind[1:], [i])

                    condProbs.append(sliceOut)
                    bestAssignment[i] = ind[np.asscalar(si) + 1]  # skip the 0th entry
                    assignmentHistory.append(bestAssignment)
                    chainState.append(lastlik)  # last likelihood
                    if(fDebug):
                        checkAssignmentVector(
                            self.indices, bestAssignment, self.XExpanded[
                                bestAssignment, :])  # check we haven't messed up

            if(fDebug):
                print('Iteration %g/%g, lik=%.3f' % (c, maximumNumberOfSteps, chainState[-1]))
                sys.stdout.flush()
            # check convergence
            if(c > 1 and np.abs(chainState[-1] - chainState[-2]) < tol):
                break

        # Done - return state vector
        print('Converged after %g iterations, lik=%.3f' % (c, chainState[-1]))
        if(fReturnAssignmentHistory):
            return (chainState, bestAssignment, assignmentHistory, condProbs)
        else:
            return (chainState, bestAssignment)

    def build_prior_assign(self):
        """
        the prior for assignments
        """
        return 0.  # TODO

    #@GPflow.model.AutoFlow(tf.placeholder(tf.float64))
    def K(self, foo):
        return self.kern.K(self.XExpanded) + foo

    # should be renamed as prior on assignments is included?
    def build_likelihoodAssignment(self, assignments_tf, fDebug=False):
        if self.ZExpanded is None:  # No sparse GP
            with tf.name_scope('Likelihood_full'):
                return self.build_likelihoodAssignment_full(assignments_tf, fDebug)
        else:
            with tf.name_scope('Likelihood_Sparse'):
                return self.build_likelihoodAssignment_sparse(assignments_tf, fDebug)

    # should be renamed as prior on assignments is included?
    def build_likelihoodAssignment_full(self, assignments_tf, fDebug=False):
        N = self.Y.shape[0]
        M = self.XExpanded.shape[0]
        L = self.KChol

        tau = 1. / self.likelihood.variance
        if(fDebug):
            snl = 3  # how many entries to print
            L = tf.Print(L, [tf.shape(L), L], message='L=', name='Ldebug', summarize=snl)  # will print message
            assignments = tf.Print(assignments_tf,
                                   [tf.shape(assignments_tf),
                                    assignments_tf],
                                   message='assign=',
                                   name='assigndebug',
                                   summarize=snl)  # will print message
        else:
            assignments = assignments_tf

        p = tf.Variable(tf.zeros((self.XExpanded.shape[0],), tf.float64))
        p.assign(np.zeros((self.XExpanded.shape[0],)))  # make sure we overwrite previous run

        p = tf.scatter_add(p, assignments, np.ones(self.Y.shape[0]), name='scatteradd_p')
        if(fDebug):
            p = tf.Print(p, [tf.shape(p), p], message='p=', name='pdebug', summarize=snl)  # will print message

        LTA = tf.transpose(L) * tf.sqrt(p)
        if(fDebug):
            LTA = tf.Print(LTA, [tf.shape(LTA), LTA], message='LTA=',
                           name='LTAdebug', summarize=snl)  # will print message

        P = tf.matmul(LTA, tf.transpose(LTA)) * tau + GPflow.tf_hacks.eye(M)
        if(fDebug):
            P = tf.Print(P, [tf.shape(P), P], message='P=', name='Pdebug', summarize=snl)  # will print message

        R = tf.cholesky(P)

        PhiY = tf.Variable(tf.zeros((self.XExpanded.shape[0], self.Y.shape[1]), tf.float64))
        PhiY = tf.scatter_add(PhiY, assignments, self.Y)
        LPhiY = tf.matmul(tf.transpose(L), PhiY)
        RiLPhiY = tf.matrix_triangular_solve(R, LPhiY, lower=True)

        D = self.Y.shape[1]
        if(fDebug):
            tauD = tf.Print(tau, [tf.shape(tau), tau], message='tau=',
                            name='taudebug', summarize=snl)  # will print message
            RD = tf.Print(R, [tf.shape(R), R], message='R=', name='Rdebug', summarize=snl)  # will print message
            RiLPhiYD = tf.Print(RiLPhiY, [tf.shape(RiLPhiY), RiLPhiY], message='RiLPhiY=',
                                name='RiLPhiYdebug', summarize=snl)  # will print message
            tauD = tf.Print(tau, [tf.shape(tau), tau], message='tau=',
                            name='taudebug', summarize=snl)  # will print message

            res =  -0.5 * N * D * tf.log(2 * np.pi / tauD)\
                   - 0.5 * D * tf.reduce_sum(tf.log(tf.square(tf.diag_part(RD))))\
                   - 0.5 * tau * tf.reduce_sum(tf.square(self.Y))\
                + 0.5 * tf.reduce_sum(tf.square(tauD * RiLPhiYD))\
                + self.build_prior_assign()
            return tf.Print(res, [tf.shape(res), res], message='lik=',
                            name='likdebug', summarize=snl)  # will print message

        else:

            return -0.5 * N * D * tf.log(2 * np.pi / tau)\
                - 0.5 * D * tf.reduce_sum(tf.log(tf.square(tf.diag_part(R))))\
                - 0.5 * tau * tf.reduce_sum(tf.square(self.Y))\
                + 0.5 * tf.reduce_sum(tf.square(tau * RiLPhiY))\
                + self.build_prior_assign()

    def build_likelihoodAssignment_sparse(self, assignments_tf, fDebug=False):
        M = self.KChol.shape[0]
        with tf.name_scope('prepare'):
            tau = 1. / self.likelihood.variance
            L = self.KChol
            W = tf.matrix_triangular_solve(L, self.Kuf)

            # copmute the 'probability' of assignment \in {0, 1}
            p = tf.Variable(tf.zeros((self.XExpanded.shape[0],), tf.float64))
            p.assign(np.zeros((self.XExpanded.shape[0],)))  # make sure we overwrite previous run
            p = tf.scatter_add(p, assignments_tf, np.ones(self.Y.shape[0]), name='scatteradd_p')

            LTA = W * tf.sqrt(p)
        with tf.name_scope('prepare2'):
            P = tf.matmul(LTA, tf.transpose(LTA)) * tau + GPflow.tf_hacks.eye(M)
        with tf.name_scope('prepare3'):
            traceTerm = -0.5 * tau * (tf.reduce_sum(self.Kdiag * p) - tf.reduce_sum(tf.square(LTA)))
        with tf.name_scope('prepare4'):
            R = tf.cholesky(P)
        with tf.name_scope('prepare5'):
            PhiY = tf.Variable(tf.zeros((self.XExpanded.shape[0], self.Y.shape[1]), tf.float64))
        with tf.name_scope('prepare6'):
            PhiY = tf.scatter_add(PhiY, assignments_tf, self.Y)  # N X 1
        with tf.name_scope('prepare7'):
            KufPhiY = tf.matmul(self.Kuf, PhiY)  # I = ind pt X 1
        with tf.name_scope('prepare8'):
            LPhiY = tf.matmul(tf.transpose(L), KufPhiY)
        with tf.name_scope('prepare9'):
            RiLPhiY = tf.matrix_triangular_solve(R, LPhiY, lower=True)

        with tf.name_scope('a'):
            a = traceTerm + 0.5 * self.Y.size * tf.log(tau)
        with tf.name_scope('b'):
            b = - 0.5 * self.Y.shape[1] * tf.reduce_sum(tf.log(tf.square(tf.diag_part(R))))
        with tf.name_scope('c'):
            c = - 0.5 * tau * tf.reduce_sum(tf.square(self.Y))
        with tf.name_scope('d'):
            d = + 0.5 * tf.reduce_sum(tf.square(tau * RiLPhiY))

        return a + b + c + d
