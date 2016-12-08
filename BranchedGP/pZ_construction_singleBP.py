import numpy as np
import tensorflow as tf


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


def make_matrix(X, BP, eZ0, epsilon=1e-6, pZ0=None):
    num_columns = 3 * tf.shape(X)[0]  # for 3 latent fns
    rows = []
    count = tf.zeros((1,), dtype=tf.int32)
    for x in X:
        # compute how many functions x may belong to
        # needs generalizing for more BPs
        n = tf.cast(tf.greater(x, BP), tf.int32) + 1
        # n == 1 when x <= BP
        # n == 2 when x > BP
        row = [tf.zeros(count + n - 1, tf.float64) + epsilon]  # all entries until count are zero
        # add 1's for possible entries
        probs = tf.ones(n, tf.float64)
        row.append(probs)
        row.append(tf.zeros(2 - 2 * (n - 1), tf.float64) + epsilon)  # append zero
        count += 3
        row.append(tf.zeros(num_columns - count, tf.float64) + epsilon)
        # ensure things are correctly shaped
        row = tf.concat(0, row, name='singleconcat')
        row = tf.expand_dims(row, 0)
        rows.append(row)
    return tf.mul(tf.concat(0, rows, name='multiconcat'), eZ0)


if __name__ == "__main__":
    #X = np.random.rand(10, 1)
    np.set_printoptions(suppress=True,  precision=2)
    X = np.linspace(0, 1, 4, dtype=float)[:, None]
    X = np.sort(X, 0)
    BP_tf = tf.placeholder(tf.float64, shape=[])
    eZ0_tf = tf.placeholder(tf.float64, shape=(X.shape[0], X.shape[0]*3))
    pZ0 = np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5], [1, 0]])
    eZ0 = expand_pZ0(pZ0)
    BP = 0.2
    pZ = tf.Session().run(make_matrix(X, BP, eZ0_tf), feed_dict={BP_tf: BP, eZ0_tf: eZ0})
    print('pZ0', pZ0)
    print('eZ0', eZ0)
    print('pZ', pZ)
    for r, c in zip(range(0, X.shape[0]), range(0, X.shape[0]*3, 3)):
        print(r, c)
        print(X[r], pZ[r, c:c+3], pZ0[r, :])
        if(X[r] > BP):  # after branch point should be prior
            assert np.allclose(pZ[r, c+1:c+3], pZ0[r, :], atol=1e-6), 'must be the same! %s-%s' % (str(pZ[r, c:c+3]), str(pZ0[r, :]))
        else:
            assert np.allclose(pZ[r, c:c+3], np.array([1., 0., 0.]), atol=1e-6), 'must be the same! %s-%s' % (str(pZ[r, c:c+3]), str(pZ0[r, :]))
#     from matplotlib import pyplot as plt
#     plt.ion()
#     plt.matshow(pZ)
