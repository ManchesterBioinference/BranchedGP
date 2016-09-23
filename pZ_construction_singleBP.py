import numpy as np
import tensorflow as tf


def make_matrix(X, BP, epsilon=1e-6):
    num_columns = 3 * X.shape[0]  # for 3 latent fns
    rows = []
    count = tf.zeros((1,), dtype=tf.int32)
    for x in X:

        # compute how many functions x may belong to
        # needs generalizing for more BPs
        n = tf.cast(tf.greater(x, BP), tf.int32) + 1

        # n == 1 when x <= BP
        # n == 2 when x > BP

#         1 / 1
#         [1 1]/2

        # 1
        # 0 1/2 1/2

        # 1
        # 0 1/2 1/2
        row = [tf.zeros(count + n - 1, tf.float64) + epsilon]  # all entries until count are zero
        probs = tf.ones(n, tf.float64) / tf.cast(n, tf.float64)
        row.append(probs)

        # 2-2*n = 2 ,n=0
        #        0 ,n=1
        row.append(tf.zeros(2 - 2 * (n - 1), tf.float64) + epsilon)  # append zero

        count += 3  # TODO: change for number of latent fns
        row.append(tf.zeros(num_columns - count, tf.float64) + epsilon)

        # ensure things are correctly shaped
        row = tf.concat(0, row, name='singleconcat')
        row = tf.expand_dims(row, 0)
        rows.append(row)
    return tf.concat(0, rows, name='multiconcant')


if __name__ == "__main__":
    #X = np.random.rand(10, 1)
    X = np.linspace(0, 1, 4, dtype=float)[:, None]
    X = np.sort(X, 0)
    BP = tf.placeholder(tf.float64, shape=[])
    pZ = tf.Session().run(make_matrix(X, BP), feed_dict={BP: 0.5})

    from matplotlib import pyplot as plt
    plt.ion()
    plt.matshow(pZ)
