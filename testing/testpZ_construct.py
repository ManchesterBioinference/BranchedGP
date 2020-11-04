# Generic libraries
import unittest

import gpflow
import numpy as np
import tensorflow as tf

# Branching files
from BranchedGP import pZ_construction_singleBP


class TestpZ(unittest.TestCase):
    def test_pZ(self):
        np.set_printoptions(suppress=True, precision=2)
        X = np.linspace(0, 1, 4, dtype=float)[:, None]
        X = np.sort(X, 0)
        with tf.compat.v1.Session():
            BP_tf = tf.compat.v1.placeholder(gpflow.default_float(), shape=[])
            eZ0_tf = tf.compat.v1.placeholder(
                gpflow.default_float(), shape=(X.shape[0], X.shape[0] * 3)
            )
            pZ0 = np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5], [1, 0]])
            eZ0 = pZ_construction_singleBP.expand_pZ0(pZ0)
            for BP in [0, 0.2, 0.5, 1]:
                print("========== BP %.2f ===========" % BP)
                pZ = tf.compat.v1.Session().run(
                    pZ_construction_singleBP.make_matrix(X, BP, eZ0_tf),
                    feed_dict={BP_tf: BP, eZ0_tf: eZ0},
                )
                print("pZ0", pZ0)
                print("eZ0", eZ0)
                print("pZ", pZ)
                for r, c in zip(range(0, X.shape[0]), range(0, X.shape[0] * 3, 3)):
                    print(X[r], pZ[r, c : c + 3], pZ0[r, :])
                    if X[r] > BP:  # after branch point should be prior
                        self.assertTrue(
                            np.allclose(pZ[r, c + 1 : c + 3], pZ0[r, :], atol=1e-6),
                            "must be the same! %s-%s"
                            % (str(pZ[r, c : c + 3]), str(pZ0[r, :])),
                        )
                    else:
                        self.assertTrue(
                            np.allclose(
                                pZ[r, c : c + 3], np.array([1.0, 0.0, 0.0]), atol=1e-6
                            ),
                            "must be the same! %s-%s"
                            % (str(pZ[r, c : c + 3]), str(pZ0[r, :])),
                        )

    def test_further(self):
        np.set_printoptions(suppress=True, precision=6)
        # X = np.linspace(0, 1, 4, dtype=float)[:, None]
        X = np.array([0.1, 0.2, 0.3, 0.4])[:, None]
        with tf.compat.v1.Session():
            BP_tf = tf.compat.v1.placeholder(dtype=gpflow.default_float(), shape=[])
            eZ0_tf = tf.compat.v1.placeholder(
                dtype=gpflow.default_float(), shape=(X.shape[0], X.shape[0] * 3)
            )
            pZ0 = np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5], [0.85, 0.15]])
            eZ0 = pZ_construction_singleBP.expand_pZ0(pZ0)
            BP = 0.2
            pZ = tf.compat.v1.Session().run(
                pZ_construction_singleBP.make_matrix(X, BP_tf, eZ0_tf),
                feed_dict={BP_tf: BP, eZ0_tf: eZ0},
            )
            print("pZ0", pZ0)
            print("eZ0", eZ0)
            print("pZ", pZ)
            for r, c in zip(range(0, X.shape[0]), range(0, X.shape[0] * 3, 3)):
                print(r, c)
                print(X[r], pZ[r, c : c + 3], pZ0[r, :])
                if X[r] > BP:  # after branch point should be prior
                    assert np.allclose(
                        pZ[r, c + 1 : c + 3], pZ0[r, :], atol=1e-6
                    ), "must be the same! %s-%s" % (
                        str(pZ[r, c : c + 3]),
                        str(pZ0[r, :]),
                    )
                else:
                    assert np.allclose(
                        pZ[r, c : c + 3], np.array([1.0, 0.0, 0.0]), atol=1e-6
                    ), "must be the same! %s-%s" % (
                        str(pZ[r, c : c + 3]),
                        str(np.array([1.0, 0.0, 0.0])),
                    )

            eZ0z = pZ_construction_singleBP.expand_pZ0Zeros(pZ0)
            r = pZ_construction_singleBP.expand_pZ0PureNumpyZeros(eZ0z, BP, X)
            assert np.allclose(r, pZ, atol=1e-5)

            # try another
            pZ = tf.compat.v1.Session().run(
                pZ_construction_singleBP.make_matrix(X, BP_tf, eZ0_tf),
                feed_dict={BP_tf: 0.3, eZ0_tf: eZ0},
            )
            r = pZ_construction_singleBP.expand_pZ0PureNumpyZeros(eZ0z, 0.3, X)
            assert np.allclose(r, pZ, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
