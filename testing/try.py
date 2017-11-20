import tensorflow as tf
import gpflow
g = tf.Graph()
with gpflow.defer_build():
	kernel = gpflow.kernels.RBF(1)
	X = tf.placeholder(gpflow.settings.np_float)
    kernel.compile()
	gram_matrix = session.run(kernel.K(X), feed_dict={X: X_data})
	assert_allclose(gram_matrix, reference_gram_matrix)

with gpflow.defer_build():
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    kernel = gpflow.kernels.PeriodicKernel(1, period=1, variance=1, lengthscales=1)
    kernel.compile()

