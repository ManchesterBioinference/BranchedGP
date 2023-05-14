import logging
import sys
from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import tabulate_module_summary

from . import VBHelperFunctions
from .assigngp import AssignGP, BranchKernelParam

LOG = logging.getLogger("mBGP")


def set_model_training_log_level(level: int) -> None:
    """
    Set logging level for this module.

    Use the constants from the logging module, for example,
        >>> set_model_training_log_level(logging.INFO)
    """
    LOG.setLevel(level)


def log_to_stdout() -> None:
    """Ensure that we log to stdout (that is, your terminal)."""
    logging.basicConfig(stream=sys.stdout)


def FitModel(
    bConsider,
    GPt,
    GPy,
    globalBranching,
    priorConfidence=0.80,
    M=10,
    likvar=1.0,
    kerlen=2.0,
    kervar=5.0,
    fDebug=False,
    maxiter=1000,
    fPredict=True,
    fixHyperparameters=False,
    optimisation_method="L-BFGS-B",
    kern: Optional[BranchKernelParam] = None,
):
    """
    Fit BGP model
    :param bConsider: list of candidate branching points
    :param GPt: pseudotime
    :param GPy: gene expression. Should be 0 mean for best performance.
    :param globalBranching: cell labels
    :param priorConfidence: prior confidence on cell labels
    :param M: number of inducing points
    :param likvar: initial value for Gaussian noise variance
    :param kerlen: initial value for kernel length scale
    :param kervar: initial value for kernel variance
    :param fDebug: Print debugging information
    :param maxiter: maximum number of iterations for optimisation
    :param fPredict: compute predictive mean and variance
    :param fixHyperparameters: should kernel hyperparameters be kept fixed or optimised?
    :param optimisation_method: Which optimisation method should we use?
        See https://docs.scipy.org/doc/scipy/reference/optimize.html for the available options.
    :param kern: the branching point kernel, see BranchKernelParam for details.
    :return: dictionary of log likelihood, GPflow model, Phi matrix, predictive set of points,
    mean and variance, hyperparameter values, posterior on branching time
    """
    assert isinstance(bConsider, list), "Candidate B must be list"
    assert GPt.ndim == 1
    assert GPy.ndim == 2
    assert (
        GPt.size == GPy.shape[0]
    ), "pseudotime and gene expression data must be the same size"
    assert (
        globalBranching.size == GPy.shape[0]
    ), "state space must be same size as number of cells"
    assert M >= 0, "at least 0 or more inducing points should be given"

    phiInitial, phiPrior = GetInitialConditionsAndPrior(
        globalBranching, priorConfidence, infPriorPhi=True
    )
    # phiInitial = phiPrior
    phiPrior = np.c_[
        np.zeros(phiPrior.shape[0])[:, None], phiPrior
    ]  # prepend 0 for trunk
    # it = np.argsort(GPt)
    # print('GPt and prior', np.hstack([GPt[it][:,None], phiPrior[it,:], phiInitial[it,:]]))
    # print(phiPrior[:5, :])
    XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(GPt)

    if M == 0:
        LOG.info("Experimental assign")
        m = AssignGP(
            GPt,
            XExpanded,
            GPy,
            indices,
            kern=kern,
            phiInitial=phiInitial,
            phiPrior=phiPrior,
            multi=True,
        )
        # m = AssignGP(GPt, XExpanded, GPy, fm, indices,
        #              phiInitial=phiInitial, phiPrior=None, multi=True)
        # ll = m.maximum_log_likelihood_objective()
        # print('log likelihood: %f',ll)
        # p_density = m.log_prior_density()
        # print('Prior density: ',p_density)
        # loss = ll + p_density
        # print('Loss: ',loss)
        # print('Training_loss',m.training_loss())
        # return
    else:
        raise NotImplementedError
        # ZExpanded = np.ones((M, 2))
        # ZExpanded[:, 0] = np.linspace(0, 1, M, endpoint=False)
        # ZExpanded[:, 1] = np.array([i for j in range(M) for i in range(1, 4)])[:M]
        # m = assigngp_denseSparse.AssignGPSparse(GPt, XExpanded, GPy, fm, indices,
        #                                         np.ones((1, 1)) * ptb, ZExpanded, phiInitial=phiInitial,
        #                                         phiPrior=phiPrior)

        # m.logPhi.set_trainable(False)
        # set_trainable(m.kernel.Bv, False)
        # m.kern.kern.lengthscales.prior = gpflow.priors.Gaussian(2., .1)
        # m.kern.kern.variance.prior = gpflow.priors.Gaussian(2, 1)
        # m.likelihood.variance.prior = gpflow.priors.Gaussian(0.1, .1)

    # optimization - for the gradient code this play the role of multiple restarts, for the
    # grid search code, this is performing grid search
    ll = np.zeros(len(bConsider))
    Phi_l, branching_points = list(), list()
    ttestl_l, mul_l, varl_l = list(), list(), list()
    hyps = list()
    models = list()

    for (ib, b), i in zip(enumerate(bConsider), range(len(bConsider))):
        m.UpdateBranchingPoint(np.ones((GPy.shape[1], 1)) * b)
        # m.UpdateBranchingPoint(np.ones((GPy.shape[1], 1)) * b, phiInitial)

        # Train the multivariate model
        trained_model = trainModel(m, maxiter, method=optimisation_method)

        # remember winning hyperparameter
        hyps.append({"likvar": trained_model.likelihood.variance.numpy()})
        ll[ib] = trained_model.log_posterior_density()
        models.append(trained_model)

        # branching value
        # branching_points.append(m.BranchingPoints.numpy().flatten())
        branching_points.append(trained_model.kernel.Bv.numpy().flatten())
        # prediction
        Phi = trained_model.GetPhi()
        Phi_l.append(Phi)
        if fPredict:
            ttestl, mul, varl = VBHelperFunctions.predictBranchingModel(trained_model)
            ttestl_l.append(ttestl), mul_l.append(mul), varl_l.append(varl)  # type: ignore
        else:
            ttestl_l.append([]), mul_l.append([]), varl_l.append([])  # type: ignore
            # TODO: help MyPy understand the above is fine

        # Train the Univariate model
        # logPhi = trained_model.logPhi.numpy()
        # bPoints = trained_model.kernel.Bv.numpy()
        # for (iGeneB, geneB) in enumerate(bPoints):
        #     print('%d %f'%(iGeneB, geneB))
        #     # fitSingleGene(bPoints[iGeneB], GPt, GPy[:, iGeneB], logPhi, globalBranching, priorConfidence,
        #     # M, likvar, kerlen, kervar, fDebug, maxiter, fPredict, fixHyperparameters)

        # set_trainable(m.kernel.kern.lengthscales, False)
        # # set_trainable(m.likelihood.variance, False)
        # set_trainable(m.kernel.kern.variance, False)
        # # if b > 0.001:
        # # m.logPhi.set_trainable(False)
        # set_trainable(m.kernel.Bv, True)

    # print(m.kern)
    iw = np.argmax(ll)
    postB = GetPosteriorB(ll, bConsider)
    # print('BGP Maximum at b=%.2f' % bConsider[iw], 'CI= [%.2f, %.2f]' %(postB['B_CI'][0], postB['B_CI'][1]))
    # print('!'*10, 'B exact=', branching_points[iw])

    # m = gridSearch_geneBygene(models[iw], phiPrior, GPy)
    # gradientSearch_geneBygene(models[iw], phiPrior, GPy)

    return {
        "loglik": ll,
        "Bmode": branching_points[iw],
        "Phi": Phi_l[iw],
        "model": models[iw],
        "prediction": {"xtest": ttestl_l[iw], "mu": mul_l[iw], "var": varl_l[iw]},
        "hyperparameters": hyps[iw],
        "posteriorB": postB,
        "globalB": branching_points[0],
    }


def trainModel(gpflow_model, maxiter=100, method="L-BFGS-B"):
    LOG.info(
        f"Starting training. Initial loss: {gpflow_model.training_loss()}. "
        f"Model summary:\n{tabulate_module_summary(gpflow_model, 'simple')}"
    )
    if method == "Adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=0.1,
        )
        opt.minimize(gpflow_model.training_loss, gpflow_model.trainable_variables)
    else:
        opt = gpflow.optimizers.Scipy()
        result = opt.minimize(
            gpflow_model.training_loss,
            variables=gpflow_model.trainable_variables,
            options=dict(maxiter=maxiter),
            method=method,
        )
        LOG.info(result)

    LOG.info(
        f"Training complete. Final loss: {gpflow_model.training_loss()}. "
        f"Model summary:\n{tabulate_module_summary(gpflow_model, 'simple')}"
    )

    return gpflow_model
    """
    except:
        print('Failure', "Unexpected error:", sys.exc_info()[0])
        print('-' * 60)
        traceback.print_exc(file=sys.stdout)
        print('Exception caused by model')
        print(gpflow_model)
        print('-' * 60)
        # return model so can inspect model
        return {'loglik': np.nan, 'model': m, 'Phi': np.nan, 'Bmode': np.nan,
                'prediction': {'xtest': np.nan, 'mu': np.nan, 'var': np.nan},
                'hyperparameters': np.nan, 'posteriorB': np.nan}
    """


def GetPosteriorB(objUnsorted, BgridSearch, ciLimits=[0.01, 0.99]):
    """
    Return posterior on B for each experiment, confidence interval index, map index
    """
    # for each trueB calculate posterior over grid
    # ... in a numerically stable way
    assert objUnsorted.size == len(BgridSearch), "size do not match %g-%g" % (
        objUnsorted.size,
        len(BgridSearch),
    )
    gr = np.array(BgridSearch)
    isort = np.argsort(gr)
    gr = gr[isort]
    o = objUnsorted[isort].copy()  # sorted objective funtion
    imode = np.argmax(o)
    pn = np.exp(o - np.max(o))
    p = pn / pn.sum()
    assert np.any(~np.isnan(p)), "Nans in p! %s" % str(p)
    assert np.any(~np.isinf(p)), "Infinities in p! %s" % str(p)
    pb_cdf = np.cumsum(p)
    confInt = np.zeros(len(ciLimits), dtype=int)
    for pb_i, pb_c in enumerate(ciLimits):
        pb_idx = np.flatnonzero(pb_cdf <= pb_c)
        if pb_idx.size == 0:
            confInt[pb_i] = 0
        else:
            confInt[pb_i] = np.max(pb_idx)
    # if((imode+5) > 0 and imode < (len(BgridSearch)-5)):  # for modes at end points conf interval checks do not hold
    #     assert confInt[0] <= (imode-1), 'Lower confidence point bigger than mode! (%s)-%g' % (str(confInt), imode)
    #     assert confInt[1] >= (imode+1), 'Upper confidence point bigger than mode! (%s)-%g' % (str(confInt), imode)
    assert np.all(confInt < o.size), confInt
    B_CI = gr[confInt]
    Bmode = gr[imode]
    # return confidence interval as well as mode, and indexes for each
    return {
        "B_CI": B_CI,
        "Bmode": Bmode,
        "idx_confInt": confInt,
        "idx_mode": imode,
        "BgridSearch_sort": gr,
        "isort": isort,
    }


def GetInitialConditionsAndPrior(globalBranching, v, infPriorPhi):
    # Setting initial phi
    np.random.seed(42)  # UNDONE remove TODO
    assert isinstance(v, float), "v should be scalar is %s" % str(type(v))
    N = globalBranching.size
    phiInitial = np.ones((N, 2)) * 0.5  # don't know anything
    # SUMON
    # Sumon is commenting the following two lines
    phiInitial[:, 0] = np.random.rand(N)
    phiInitial[:, 1] = 1 - phiInitial[:, 0]
    # SUMON END
    phiPrior = np.ones_like(phiInitial) * 0.5  # don't know anything
    for i in range(N):
        iBranch = globalBranching[i] - 2  # is 1,2,3-> -1, 0, 1
        if iBranch == -1:
            # trunk - set all equal
            phiPrior[i, :] = 0.5
        else:
            if infPriorPhi:
                phiPrior[i, :] = 1 - v
                phiPrior[i, int(iBranch)] = v
            # SUMON - Can we get rid of following couple of lines
            # phiInitial[i, int(iBranch)] = v

            # TODO: Elvijs - slight cheating here? We're potentially telling the model quite a bit here
            phiInitial[i, int(iBranch)] = 0.5 + (
                np.random.random() / 2.0
            )  # number between [0.5, 1]
            phiInitial[i, int(iBranch) != np.array([0, 1])] = (
                1 - phiInitial[i, int(iBranch)]
            )
    assert np.allclose(
        phiPrior.sum(1), 1
    ), "Phi Prior should be close to 1 but got %s" % str(phiPrior)
    assert np.allclose(
        phiInitial.sum(1), 1
    ), "Phi Initial should be close to 1 but got %s" % str(phiInitial)
    assert np.all(~np.isnan(phiInitial)), "No nans please!"
    assert np.all(~np.isnan(phiPrior)), "No nans please!"
    return phiInitial, phiPrior
