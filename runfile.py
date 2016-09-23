
import GPflow
import numpy as np


def getData():
    rng = np.random.RandomState(1)
    N = 30
    X = rng.rand(N, 1)
    Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + rng.randn(N, 1) * 0.1 + 3
    return X, Y
if __name__ == '__main__':
    X, Y = getData()
    k = GPflow.kernels.Matern52(1)
    meanf = GPflow.mean_functions.Linear(1, 0)
    m = GPflow.gpr.GPR(X, Y, k, meanf)
    m.likelihood.variance = 0.01
    m._compile()
    m.kern.lengthscales = 2
    print("Here are the parameters before optimization")
    print(m)
    fs = m.get_free_state()
    np.save('saveModelState', fs)
    msstate = np.load('saveModelState.npy')
    m2 = GPflow.gpr.GPR(X, Y, k, meanf)
    m2.set_state(msstate)
    print('m2 state')
    print(m2)

    print('lengthscale ' + str(m2.kern.lengthscales._array.flatten()))
    assert m2.kern.lengthscales._array.flatten() == 2., 'This better be 2'
    #[mu,var] = m.predict_f(X)
    # print mu

    print('done')
