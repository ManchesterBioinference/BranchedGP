''' Module to replace branch_kern with parameterised version'''

import GPflow
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def PlotSample(D, X, M, samples, B=None, lw=3., fs=10, figsizeIn=(12, 16), title=None, mV=None):
    f, ax = plt.subplots(D, 1, figsize=figsizeIn, sharex=True, sharey=True)
    nb = len(B)  # number of branch points
    for d in range(D):
        for i in range(1, M + 1):
            t = X[X[:, 1] == i, 0]
            y = samples[X[:, 1] == i, d]
            if(t.size == 0):
                continue
            if(D != 1):
                p = ax.flatten()[d]
            else:
                p = ax

            p.plot(t, y, '.', label=i, markersize=2 * lw)
            p.text(t[t.size / 2], y[t.size / 2], str(i), fontsize=fs)
        # Add vertical lines for branch points
        if(title is not None):
            p.set_title(title + ' Dim=' + str(d), fontsize=fs)

        if(B is not None):
            v = p.axis()
            for i in range(nb):
                p.plot([B[i], B[i]], v[-2:], '--r')
        if(mV is not None):
            assert B.size == 1, 'Code limited to one branch point, got ' + str(B.shape)

            pt = mV.t
            l = np.min(pt)
            u = np.max(pt)
            for f in range(1, 4):
                if(f == 1):
                    ttest = np.linspace(l, B.flatten(), 100)[:, None]  # root
                else:
                    ttest = np.linspace(B.flatten(), u, 100)[:, None]
                Xtest = np.hstack((ttest, ttest * 0 + f))
                mu, var = mV.predict_f(Xtest)
                assert np.all(np.isfinite(mu)), 'All elements should be finite but are ' + str(mu)
                assert np.all(np.isfinite(var)), 'All elements should be finite but are ' + str(var)
                mean, = p.plot(ttest, mu[:, d], linewidth=lw)
                col = mean.get_color()
                # print 'd='+str(d)+ ' f='+str(f) + '================'
                # variance is common for all outputs!
                p.plot(ttest.flatten(), mu[:, d] + 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)
                p.plot(ttest, mu[:, d] - 2 * np.sqrt(var.flatten()), '--', color=col, linewidth=lw)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#PlotSample(D,m.XExpanded[bestAssignment, : ],3,Y,Bcrap,lw=5.,fs=30, mV=mV, figsizeIn=(D*10,D*7),title='Posterior B=%.1f -loglik= %.2f VB= %.2f'%(b,-chainState[-1],VBbound))


def SampleBranchGP(kern, X, M, D=1, lw=3., fs=10, tol=1e-5):
    ''' Sample kernel on X. Can generate D independent samples
    X input
    M number of functions
    tol tolerance to add such that matrix is invertible
    lw line width
    fs font size
    kern has the branching point specified along with the other GP hyperparameters
    '''
    x_free = tf.placeholder(tf.float64)
    kern.make_tf_array(x_free)
    with kern.tf_mode():
        K = kern.K(X)
    with tf.Session() as sess:
        K = sess.run(K, feed_dict={x_free: kern.get_free_state()})

    L = np.linalg.cholesky(K + np.eye(K.shape[0]) * tol)
    samples = L.dot(np.random.randn(L.shape[0], D))

    return (samples, L, K)


class BranchKernelParam(GPflow.kernels.Kern):

    def __init__(self, base_kern, branchPtTensor, BvInitial, fDebug=False):
        ''' branchPtTensor is tensor of branch points of size F X F X B where F the number of
        functions and B the number of branching points '''

        GPflow.kernels.Kern.__init__(self, input_dim=base_kern.input_dim + 1)
        self.kern = base_kern
        self.fm = branchPtTensor
        self.fDebug = fDebug
        assert self.fm.shape[0] == self.fm.shape[1]
        assert self.fm.shape[2] > 0
#         # make list of Params for branching pts
#         self.Bs = GPflow.param.ParamList()
#         for _ in range(numBranchingPts):
#             self.Bs.append(GPflow.param.Param(1))
#     then need to do indexing self.Bs[bnan] in K
        assert BvInitial.dtype == np.float
        self.Bv = GPflow.param.Param(BvInitial)

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64))
    def eval_K(self, X):
        return self.K(X)

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64), tf.placeholder(tf.float64))
    def eval_K_x(self, X, Z):
        return self.K(X, Z)

    def K(self, X, Y=None):
        if Y is None:
            Y = X  # hack to avoid duplicating code below

        t1s = tf.expand_dims(X[:, 0], 1)  # N X 1
        t2s = tf.expand_dims(Y[:, 0], 1)
        i1s_r = tf.expand_dims(X[:, 1], 1)
        i2s_r = tf.expand_dims(Y[:, 1], 1)
        if(self.fDebug):
            snl = 10  # how many entries to print
            i1s = tf.Print(i1s_r, [tf.shape(i1s_r), i1s_r], message='i1s=',
                           name='i1sdebug', summarize=snl)  # will print message
            i2s = tf.Print(i2s_r, [tf.shape(i2s_r), i2s_r], message='i2s=',
                           name='i2sdebug', summarize=snl)  # will print message
        else:
            i1s = i1s_r
            i2s = i2s_r

        i1s_matrix = tf.tile(i1s, tf.reverse(tf.shape(i2s), [True]))
        i2s_matrix = tf.tile(i2s, tf.reverse(tf.shape(i1s), [True]))
        i2s_matrixT = tf.transpose(i2s_matrix)

        Ktts = self.kern.K(t1s, t2s)  # N*M X N*M
        with tf.name_scope("kttscope"):  # scope
            same_functions = tf.equal(i1s_matrix, tf.transpose(i2s_matrix), name='FiEQFj')
            K_s = tf.select(same_functions, Ktts, Ktts, name='selectFiEQFj')  # just setup matrix with block diagonal

        m = self.fm.shape[0]
        for fi in range(m):
            for fj in range(m):
                if (fi != fj):
                    with tf.name_scope("f" + str(fi) + "f" + str(fj)):  # scope
                        # much easier to remove nans before tensorflow
                        bnan = self.fm[fi, fj, ~np.isnan(self.fm[fi, fj, :])]
                        fi_s = tf.constant(fi + 1, tf.int32, name='function' + str(fi))
                        fj_s = tf.constant(fj + 1, tf.int32, name='function' + str(fj))

                        i1s_matrixInt = tf.cast(i1s_matrix, tf.int32, name='casti1s')
                        i2s_matrixTInt = tf.cast(i2s_matrixT, tf.int32, name='casti2s')

                        fiFilter = fi_s * tf.ones_like(i1s_matrixInt, tf.int32, name='fiFilter')
                        fjFilter = fj_s * tf.ones_like(i2s_matrixTInt, tf.int32, name='fjFilter')  # must be transpose

                        f1F = tf.equal(i1s_matrixInt, fiFilter, name='indexF' + str(fi))
                        f2F = tf.equal(i2s_matrixTInt, fjFilter, name='indexF' + str(fj))

                        t12F = tf.logical_and(f1F, f2F, name='F' + str(fi) + 'andF' + str(fj))

                        # Get the actual values of the Bs = B[index of relevant branching points]
                        bint = bnan.astype(int)  # convert to int - set of indexes
                        Bs = ((tf.concat(0, [tf.slice(self.Bv, [i - 1, 0], [1, 1]) for i in bint])))

                        kbb = self.kern.K(Bs) + tf.diag(tf.ones(tf.shape(Bs)[:1], dtype=tf.float64)) * 1e-6
                        Kbbs_inv = tf.matrix_inverse(kbb, name='invKbb')  # B X B
                        Kb1s = self.kern.K(t1s, Bs)  # N*m X B
                        Kb2s = self.kern.K(t2s, Bs)  # N*m X B

                        a = tf.matmul(Kb1s, Kbbs_inv)
                        K_crosss = tf.matmul(a, tf.transpose(Kb2s), name='Kt1_Bi_invBB_KBt2')

                        K_s = tf.select(t12F, K_crosss, K_s, name='selectIndex')
        return K_s

    def Kdiag(self, X):
        return tf.diag_part(self.kern.K(X))  # diagonal is just single point no branch point relevant

    # an independent output kernel


class IndKern(GPflow.kernels.Kern):

    def __init__(self, base_kern):
        GPflow.kernels.Kern.__init__(self, input_dim=base_kern.input_dim + 1)
        self.kern = base_kern

    def K(self, X, Y=None):
        if Y is None:
            t1 = X[:, :-1]
            i1 = X[:, -1:]
            Ktt = self.kern.K(t1)

            i1_matrix = tf.tile(i1, tf.reverse(tf.shape(i1), [True]))

            same_functions = tf.equal(i1_matrix, tf.transpose(i1_matrix))
            K_s = tf.select(same_functions, Ktt, tf.zeros_like(Ktt))
            return K_s
        else:
            t1 = tf.expand_dims(X[:, 0], 1)
            t2 = tf.expand_dims(Y[:, 0], 1)
            i1 = tf.expand_dims(X[:, 1], 1)
            i2 = tf.expand_dims(Y[:, 1], 1)

            Ktt = self.kern.K(t1, t2)

            i1_matrix = tf.tile(i1, tf.reverse(tf.shape(i2), [True]))
            i2_matrix = tf.tile(i2, tf.reverse(tf.shape(i1), [True]))

            same_functions = tf.equal(i1_matrix, tf.transpose(i2_matrix))
            K_s = tf.select(same_functions, Ktt, tf.zeros_like(Ktt))
            return K_s

    def Kdiag(self, X):
        return tf.diag_part(self.kern.K(X))


class SplitKernel(GPflow.kernels.Kern):

    def __init__(self, base_kern, breakpoint):
        GPflow.kernels.Kern.__init__(self, input_dim=base_kern.input_dim + 1)
        self.kern = base_kern
        self.B = np.ones((1, 1)) * breakpoint

    def K(self, X, Y=None):
        if Y is None:
            t = tf.expand_dims(X[:, 0], 1)
            i = tf.expand_dims(X[:, 1], 1)
            Kb = self.kern.K(t, self.B)  # N X 1
            Kbb = self.kern.K(self.B)
            Ktt = self.kern.K(t)
            K_cross = tf.matmul(Kb, tf.transpose(Kb)) / Kbb

            i_matrix = tf.tile(i, tf.reverse(tf.shape(i), [True]))

            same_functions = tf.equal(i_matrix, tf.transpose(i_matrix))
            K_s = tf.select(same_functions, Ktt, K_cross)  # N X M of selected entries
            return K_s
        else:
            t1 = tf.expand_dims(X[:, 0], 1)  # N X 1
            t2 = tf.expand_dims(Y[:, 0], 1)
            i1 = tf.expand_dims(X[:, 1], 1)
            i2 = tf.expand_dims(Y[:, 1], 1)
            Kb1 = self.kern.K(t1, self.B)  # N X 1
            Kb2 = self.kern.K(t2, self.B)
            Kbb = self.kern.K(self.B)
            Ktt = self.kern.K(t1, t2)
            K_crosss = tf.matmul(Kb1, tf.transpose(Kb2)) / Kbb

            i1_matrix = tf.tile(i1, tf.reverse(tf.shape(i2), [True]))
            i2_matrix = tf.tile(i2, tf.reverse(tf.shape(i1), [True]))

            same_functions = tf.equal(i1_matrix, tf.transpose(i2_matrix))
            K_s = tf.select(same_functions, Ktt, K_crosss)  # N X M of selected entries
            return K_s

    def Kdiag(self, X):
        return tf.diag_part(self.kern.K(X))


class BranchKernelHardcoded(GPflow.kernels.Kern):
    ''' Hardcoded branch location '''

    def __init__(self, base_kern, branchPtTensor, fDebug=False):
        ''' branchPtTensor is tensor of branch points of size F X F X B where F the number of
        functions and B the number of branching points '''

        GPflow.kernels.Kern.__init__(self, input_dim=base_kern.input_dim + 1)
        self.kern = base_kern
        self.fmb = branchPtTensor
        self.fDebug = fDebug
        assert self.fmb.shape[0] == self.fmb.shape[1]
        assert self.fmb.shape[2] > 0

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64))
    def eval_K(self, X):
        return self.K(X)

    @GPflow.model.AutoFlow(tf.placeholder(tf.float64), tf.placeholder(tf.float64))
    def eval_K_x(self, X, Z):
        return self.K(X, Z)

    def K(self, X, Y=None):
        if Y is None:
            Y = X  # hack to avoid duplicating code below

        t1s = tf.expand_dims(X[:, 0], 1)  # N X 1
        t2s = tf.expand_dims(Y[:, 0], 1)
        i1s_r = tf.expand_dims(X[:, 1], 1)
        i2s_r = tf.expand_dims(Y[:, 1], 1)
        if(self.fDebug):
            snl = 10  # how many entries to print
            i1s = tf.Print(i1s_r, [tf.shape(i1s_r), i1s_r], message='i1s=',
                           name='i1sdebug', summarize=snl)  # will print message
            i2s = tf.Print(i2s_r, [tf.shape(i2s_r), i2s_r], message='i2s=',
                           name='i2sdebug', summarize=snl)  # will print message
        else:
            i1s = i1s_r
            i2s = i2s_r

        i1s_matrix = tf.tile(i1s, tf.reverse(tf.shape(i2s), [True]))
        i2s_matrix = tf.tile(i2s, tf.reverse(tf.shape(i1s), [True]))
        i2s_matrixT = tf.transpose(i2s_matrix)

        Ktts = self.kern.K(t1s, t2s)  # N*M X N*M
        with tf.name_scope("kttscope"):  # scope
            same_functions = tf.equal(i1s_matrix, tf.transpose(i2s_matrix), name='FiEQFj')
            K_s = tf.select(same_functions, Ktts, Ktts, name='selectFiEQFj')  # just setup matrix with block diagonal

        m = self.fmb.shape[0]
        for fi in range(m):
            for fj in range(m):
                if (fi != fj):
                    with tf.name_scope("f" + str(fi) + "f" + str(fj)):  # scope
                        # much easier to remove nans before tensorflow
                        bnan = tf.constant(self.fmb[fi, fj, ~np.isnan(self.fmb[fi, fj, :])], tf.float64)
                        fi_s = tf.constant(fi + 1, tf.int32, name='function' + str(fi))
                        fj_s = tf.constant(fj + 1, tf.int32, name='function' + str(fj))

                        i1s_matrixInt = tf.cast(i1s_matrix, tf.int32, name='casti1s')
                        i2s_matrixTInt = tf.cast(i2s_matrixT, tf.int32, name='casti2s')

                        fiFilter = fi_s * tf.ones_like(i1s_matrixInt, tf.int32, name='fiFilter')
                        fjFilter = fj_s * tf.ones_like(i2s_matrixTInt, tf.int32, name='fjFilter')  # must be transpose

                        f1F = tf.equal(i1s_matrixInt, fiFilter, name='indexF' + str(fi))
                        f2F = tf.equal(i2s_matrixTInt, fjFilter, name='indexF' + str(fj))

                        t12F = tf.logical_and(f1F, f2F, name='F' + str(fi) + 'andF' + str(fj))

                        Bs = tf.expand_dims(bnan, 1)  # make a column vector

                        kbb = self.kern.K(Bs) + tf.diag(tf.ones(tf.shape(Bs)[:1], dtype=tf.float64)) * 1e-6
                        Kbbs_inv = tf.matrix_inverse(kbb, name='invKbb')  # B X B
                        Kb1s = self.kern.K(t1s, Bs)  # N*m X B
                        Kb2s = self.kern.K(t2s, Bs)  # N*m X B

                        K_crosss = tf.matmul(tf.matmul(Kb1s, Kbbs_inv), tf.transpose(Kb2s), name='Kt1_Bi_invBB_KBt2')

                        K_s = tf.select(t12F, K_crosss, K_s, name='selectIndex')
        return K_s

    def Kdiag(self, X):
        return tf.diag_part(self.kern.K(X))  # diagonal is just single point no branch point relevant
