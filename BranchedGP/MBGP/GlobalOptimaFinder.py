import gpflow
import numpy as np

from .assigngp import AssignGP


def gradientSearch_geneBygene(m, phiPrior, Y, maxiter=1000):
    print("Inside gradientSearch_geneBygene")

    GPt = m.t
    XExpanded = m.X
    GPy = Y
    indices = m.indices
    logPhi = m.logPhi
    # print(GPt)

    ptb = 0.0001
    bConsider = [0.0001] * 2 + [0.001] + [0.01, 0.05, 0.1] + [0.0001]

    for dim in range(0, m.D):
        with gpflow.defer_build():
            m_new = AssignGP(
                GPt,
                XExpanded,
                GPy[:, dim][:, None],
                indices,
                phiPrior=phiPrior,
                logPhi=logPhi,
                multi=True,
            )
            m_new.logPhi.set_trainable(False)
        m_new.compile()

        ll = np.zeros(len(bConsider))
        models = list()
        branching_points = list()

        for ib, b in enumerate(bConsider):
            m_new.UpdateBranchingPoint(np.ones((1, 1)) * b)
            m_new = trainModel(m_new, maxiter)

            ll[ib] = m_new.compute_log_likelihood()
            branching_points.append(m_new.BranchingPoints.read_value())

            m_new.kern.kern.lengthscales.set_trainable(False)
            m_new.likelihood.variance.set_trainable(False)
            m_new.kern.kern.variance.set_trainable(False)
        print(ll)
        print(branching_points)
        iw = np.argmax(ll)
        print(ll[iw])
        print(branching_points[iw])
        del m_new


def gridSearch_geneBygene(m, phiPrior, Y):
    """

    :rtype: object
    """
    print("Inside grid search")
    print(m)
    # print(phiPrior)
    NN = 10
    testPoints = np.linspace(0.0, 1.0, NN)
    print(testPoints)

    with gpflow.defer_build():
        print("*" * 60, "Inside gridSearch_geneBygene")
        GPt = m.t
        XExpanded = m.X
        GPy = Y
        indices = m.indices
        logPhi = m.logPhi
        b = 0.0001
        print(GPt)
        m_new = AssignGP(
            GPt,
            XExpanded,
            GPy[:, 0][:, None],
            indices,
            phiPrior=phiPrior,
            logPhi=logPhi,
            multi=True,
        )

        # m_new.kern.kern.lengthscales = m.kern.kern.lengthscales
        # m_new.kern.kern.variance = m.kern.kern.variance
        # m_new.likelihood.variance = m.likelihood.variance
        #
        # m_new.kern.kern.lengthscales.set_trainable(False)
        # m_new.likelihood.variance.set_trainable(False)
        # m_new.kern.kern.variance.set_trainable(False)
        m_new.logPhi.set_trainable(False)
    m_new.compile()

    print("m_new is here")
    print(m_new)

    NN = 20
    testPoints = np.linspace(0.0, 1.0, NN)
    print(testPoints)
    b_points = m.BranchingPoints.read_value().flatten()

    for dim in range(0, m.D):
        m_new.Y = Y[:, dim][:, None]
        m_new.BranchingPoints.assign((np.ones((1, 1)) * b_points[dim]).flatten())
        ll = list()

        for j in range(0, NN):
            ll.append(m_new.compute_log_likelihood())
            m_new.BranchingPoints.assign((np.ones((1, 1)) * testPoints[j]).flatten())
        print(ll)
        print(np.argmax(ll))

    print("After grid search...")
    # print(m.compute_log_likelihood())
    # print(m.BranchingPoints.read_value())
    return m


def trainModel(gpflow_model, maxiter=100):
    # print('Inside train')
    try:
        gpflow.train.ScipyOptimizer().minimize(gpflow_model, maxiter=maxiter, disp=True)
        # print('Inside try block')
        return gpflow_model
    except:
        print("Failure", "Unexpected error:", sys.exc_info()[0])
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("Exception caused by model")
        print(gpflow_model)
        print("-" * 60)
        # return model so can inspect model
        return {
            "loglik": np.nan,
            "model": m,
            "Phi": np.nan,
            "Bmode": np.nan,
            "prediction": {"xtest": np.nan, "mu": np.nan, "var": np.nan},
            "hyperparameters": np.nan,
            "posteriorB": np.nan,
        }
