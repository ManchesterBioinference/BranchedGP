import numpy as np
import tensorflow as tf
import gpflow

def expand_pZ0Zeros(pZ0, epsilon=1e-6):
    assert pZ0.shape[1] == 2, 'Should have exactly two cols got %g ' % pZ0.shape[1]
    num_columns = 3 * pZ0.shape[0]
    r = np.zeros((pZ0.shape[0], num_columns)) + epsilon
    count = 0
    for iz0, z0 in enumerate(pZ0):
        assert z0.sum() == 1, 'should sum to 1 is %s=%.3f' % (str(z0), z0.sum())
        r[iz0, count+1:count+3] = z0
        count += 3
    return r

def expand_pZ0PureNumpyZeros(eZ0, BP, X, epsilon=1e-6):
    N = X.size
    r = eZ0.copy()
    count = 0
    i = np.flatnonzero(X <= BP)
    # mark trunk as [1, 0, 0]
    r[i, i*3] = 1
    r[i, i * 3 + 1] = epsilon
    r[i, i * 3 + 2] = epsilon
    return r

def expand_pZ0(pZ0):
    assert pZ0.shape[1] == 2, 'Should have exactly two cols got %g ' % pZ0.shape[1]
    num_columns = 3 * pZ0.shape[0]
    r = np.ones((pZ0.shape[0], num_columns))
    count = 0
    for iz0, z0 in enumerate(pZ0):
        assert z0.sum() == 1, 'should sum to 1 is %s=%.3f' % (str(z0), z0.sum())
        r[iz0, count+1:count+3] = z0
        count += 3
    return r


def make_matrix(X, BP, eZ0, epsilon=1e-6):
    ''' Compute pZ which is N by N*3 matrix of prior assignment.
    This code has to be consistent with assigngp_dense.InitialiseVariationalPhi to where
        the equality is placed i.e. if x<=b trunk and if x>b branch or vice versa. We use the
         former convention.'''
    num_columns = 3 * tf.shape(X)[0]  # for 3 latent fns
    rows = []
    count = tf.zeros((1,), dtype=tf.int32)
    for x in X:
        # compute how many functions x may belong to
        # needs generalizing for more BPs
        n = tf.cast(tf.greater(x, BP), tf.int32) + 1
        # n == 1 when x <= BP
        # n == 2 when x > BP
        row = [tf.zeros(count + n - 1, dtype=gpflow.default_float()) + epsilon]  # all entries until count are zero
        # add 1's for possible entries
        probs = tf.ones(n, dtype=gpflow.default_float())
        row.append(probs)
        row.append(tf.zeros(2 - 2 * (n - 1), dtype=gpflow.default_float()) + epsilon)  # append zero
        count += 3
        row.append(tf.zeros(num_columns - count, dtype=gpflow.default_float()) + epsilon)
        # ensure things are correctly shaped
        row = tf.concat(row, 0, name='singleconcat')
        row = tf.expand_dims(row, 0)
        rows.append(row)
    return tf.multiply(tf.concat(rows, 0, name='multiconcat'), eZ0)

