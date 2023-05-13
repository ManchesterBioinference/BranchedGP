""" This module collects various helpers around model training. """
import abc
import dataclasses
import logging
from typing import Optional, Sequence, Tuple

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import tabulate_module_summary
from typing_extensions import Protocol

from . import VBHelperFunctions
from .assigngp import AssignGP, BranchKernelParam, get_branching_point_kernel
from .data_generation import ColumnVector, GeneExpressionData, ToyBranchedData, Vector
from .FitBranchingModel import GetInitialConditionsAndPrior

InitialAndPriorPhi = Tuple[np.ndarray, np.ndarray]

_DEFAULT_NUM_PREDICTIONS = 100
_EARLY_BP = 1e-4
LOG = logging.getLogger("MMBGP training")


class PhiConstructor(abc.ABC):
    """
    This interface abstracts the construction of the Phi matrix as defined
    in https://github.com/sumonahmedUoM/MMBGP/blob/master/docs/mmbgp.pdf.

    There are different ways this matrix could be initialised.
    This interface allows us to try out and compare various different initialisations.
    """

    @abc.abstractmethod
    def build(self) -> InitialAndPriorPhi:
        """Build and return initial and prior Phi matrices."""


class SimplePhiConstructor(PhiConstructor):
    """
    TODO
    """

    def __init__(
        self,
        gene_expression_data: GeneExpressionData,
        prior_confidence: float,
        allow_infs: bool = True,
    ) -> None:
        self._state = gene_expression_data.state
        self._prior_confidence = prior_confidence
        self._allow_infs = allow_infs

    def build(self) -> InitialAndPriorPhi:
        phi_initial, phi_prior = GetInitialConditionsAndPrior(
            self._state,
            self._prior_confidence,
            infPriorPhi=self._allow_infs,
        )

        phi_prior = np.c_[
            np.zeros(phi_prior.shape[0])[:, None], phi_prior
        ]  # prepend 0 for trunk
        return phi_initial, phi_prior


def get_funky_phi(
    global_branching_state: Vector,
    uninformative_until: float,
    informative_prior_confidence: float,
) -> InitialAndPriorPhi:
    if not 0 <= uninformative_until <= 1:
        raise ValueError(
            f"Expected uninformative_until to be in the range [0, 1]. "
            f"Instead got: {uninformative_until}."
        )
    if not 0 <= informative_prior_confidence <= 1:
        raise ValueError(
            f"Expected informative_prior_confidence to be in the range [0, 1]. "
            f"Instead got: {uninformative_until}."
        )

    (N,) = global_branching_state.shape
    # WARNING: assumes uniformly spaced state on [0, 1].
    uninformative_until_idx = int(uninformative_until * N)
    print(f"Uninformative until idx: {uninformative_until_idx}")

    # Initialise phi_initial and phi_prior to 50% probability for each of the branches g, h
    # We will then update them
    phi_initial = np.ones((N, 2)) * 0.5
    phi_prior = np.ones((N, 2)) * 0.5

    # TODO: the following can probably be written as pure numpy array operations,
    #   but we're taking the easy way for speed.
    for i in range(N):
        not_trunk = global_branching_state[i] in {2, 3}

        if not_trunk:
            # The g, h states normally map to 2, 3, but here we want to map them to columns 0, 1
            true_branch_col = int(global_branching_state[i] - 2)
            if i > uninformative_until_idx:
                phi_prior[i, :] = 1 - informative_prior_confidence
                phi_prior[i, true_branch_col] = informative_prior_confidence
            # else leave at 0.5, 0.5

            # Set phi_initial to have true state label with a random probability in [0.5, 1]
            # This is the same as SimplePhiConstructor
            phi_initial[i, true_branch_col] = 0.5 + (np.random.random() / 2.0)
            phi_initial[i, true_branch_col != np.array([0, 1])] = (
                1 - phi_initial[i, true_branch_col]
            )

    assert np.allclose(phi_prior.sum(1), 1), (
        f"Phi Prior probability distribution should sum to 1 for each branch. "
        f"Instead got: {phi_prior.sum(1)}"
    )
    assert np.allclose(phi_initial.sum(1), 1), (
        f"Phi Initial probability distribution should sum to 1 for each branch. "
        f"Instead got: {phi_initial.sum(1)}"
    )
    assert np.all(
        ~np.isnan(phi_initial)
    ), "Found NaNs in phi_initial, something has gone badly wrong!"
    assert np.all(
        ~np.isnan(phi_prior)
    ), "Found NaNs in phi_prior, something has gone badly wrong!"
    return phi_initial, phi_prior


class FunkyPrior(PhiConstructor):
    """
    Uninformative prior for early cells, informative prior for late cells.
    """

    def __init__(
        self,
        gene_expression_data: GeneExpressionData,
        uninformative_until: float,
        informative_prior_confidence: float,
        allow_infs: bool = True,
    ) -> None:
        self._state = gene_expression_data.state
        self._uninformative_until = uninformative_until
        self._informative_prior_confidence = informative_prior_confidence
        self._allow_infs = allow_infs

    def build(self) -> InitialAndPriorPhi:
        phi_initial, phi_prior = get_funky_phi(
            global_branching_state=self._state,
            uninformative_until=self._uninformative_until,
            informative_prior_confidence=self._informative_prior_confidence,
        )

        phi_prior = np.c_[
            np.zeros(phi_prior.shape[0])[:, None], phi_prior
        ]  # prepend 0 for trunk
        return phi_initial, phi_prior


class AssignGPOptimiser(abc.ABC):
    @abc.abstractmethod
    def train(self, model: AssignGP) -> AssignGP:
        """Optimise the `model` hyperparameters and return the trained model."""


class ScipyOptimiser(AssignGPOptimiser):
    def __init__(self, method: str = "L-BFGS-B", maxiter: int = 100, **kwargs) -> None:
        """
        :param method: Which optimisation method should we use?
            See https://docs.scipy.org/doc/scipy/reference/optimize.html for the available options.
        :param maxiter: maximum number of iterations for optimisation
        :param kwargs: Any other Scipy-compatible kwargs that your selected optimisation method accepts.
        """
        self._method = method
        self._kwargs = kwargs

        if "maxiter" not in self._kwargs:
            self._kwargs["maxiter"] = maxiter
        else:
            assert maxiter == kwargs["maxiter"], (
                f"Two different values of maxiter provided. "
                f"Directly: {maxiter}, in kwargs: {kwargs['maxiter']}"
            )

    def train(self, model: AssignGP) -> AssignGP:
        LOG.info(
            f"Starting training. Initial loss: {model.training_loss()}. "
            f"Model summary:\n{gpflow.utilities.tabulate_module_summary(model, 'simple')}"
        )
        opt = gpflow.optimizers.Scipy()
        result = opt.minimize(
            model.training_loss,
            variables=model.trainable_variables,
            options=self._kwargs,
            method=self._method,
        )
        LOG.debug(f"Optimisation result: {result}")

        LOG.info(
            f"Training complete. Final loss: {model.training_loss()}. "
            f"Model summary:\n{tabulate_module_summary(model, 'simple')}"
        )

        return model


def uniform_bp(model: AssignGP, bp: float = _EARLY_BP) -> AssignGP:
    num_outputs = model.BranchingPoints.shape[0]
    ones = np.ones((num_outputs,))

    model.kernel.Bv.assign(ones * bp)
    return model


def fixed_bps(model: AssignGP) -> AssignGP:
    gpflow.set_trainable(model.kernel.Bv, False)
    return model


def trainable_bps(model: AssignGP) -> AssignGP:
    gpflow.set_trainable(model.kernel.Bv, True)
    return model


class ElvijsAmazingOptimiser(AssignGPOptimiser):
    def __init__(
        self,
        base_optimiser: AssignGPOptimiser = ScipyOptimiser(),
        # TODO: we may want to sample differently
        branching_points: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
        initial_bp: float = _EARLY_BP,
    ) -> None:
        """TODO"""
        self._base_optimiser = base_optimiser
        self._bps = branching_points
        self._initial_bp = initial_bp

    def train(self, model: AssignGP) -> AssignGP:
        # TODO: Add logging
        model = fixed_bps(uniform_bp(model, bp=self._initial_bp))
        model_with_reasonable_params = trainable_bps(self._base_optimiser.train(model))

        best_loss = model.training_loss()
        best_model = gpflow.utilities.deepcopy(model)

        for bp in self._bps:
            new_model = gpflow.utilities.deepcopy(model_with_reasonable_params)
            new_model = uniform_bp(new_model, bp)

            trained_model = self._base_optimiser.train(new_model)

            if trained_model.training_loss() < best_loss:
                best_loss = trained_model.training_loss()
                best_model = trained_model

        return best_model


class ElvijsRandomisingOptimiser(AssignGPOptimiser):
    """TODO"""

    def __init__(
        self,
        base_optimiser: AssignGPOptimiser = ScipyOptimiser(),
        num_samples: int = 5_000,
        initial_bp: float = 1e-4,
    ):
        self._num_samples = num_samples
        self._base_optimiser = base_optimiser
        self._initial_bp = initial_bp

    def train(self, model: AssignGP) -> AssignGP:
        # TODO: add nice logging
        num_outputs = model.kernel.Bv.shape[0]

        ones = np.ones((num_outputs,))

        model.kernel.Bv.assign(ones * self._initial_bp)
        gpflow.set_trainable(model.kernel.Bv, False)

        model_with_reasonable_params = self._base_optimiser.train(model)
        gpflow.set_trainable(model_with_reasonable_params.kernel.Bv, True)

        @tf.function
        def sample(num_samples, n_outputs, _model):
            best_loss = _model.training_loss()
            best_bps = _model.kernel.Bv

            for _ in tf.range(num_samples):
                sampled_bp = tf.random.uniform(
                    shape=(n_outputs,), dtype=gpflow.default_float()
                )

                _model.kernel.Bv.assign(sampled_bp)
                loss = _model.training_loss()

                if tf.greater(best_loss, loss):
                    best_loss = loss
                    best_bps = sampled_bp

            _model.kernel.Bv.assign(best_bps)

        sample(self._num_samples, num_outputs, model_with_reasonable_params)

        trained_model = self._base_optimiser.train(model_with_reasonable_params)
        return trained_model


class OptimiserCallback(Protocol):
    def __call__(self, model: AssignGP) -> None:
        """
        Optimiser callback is any function operating on a model.
        It should NOT mutate the model.

        To be used with the CompositeOptimiser in order to track
        """


class CompositeOptimiser(AssignGPOptimiser):
    def __init__(self, *optimisers: AssignGPOptimiser) -> None:
        self._optimisers = optimisers

    def train(self, model: AssignGP) -> AssignGP:
        for optimiser in self._optimisers:
            model = optimiser.train(model)

        return model


def construct_assigngp_model(
    gene_expression: GeneExpressionData,
    phi_constructor: PhiConstructor,
    initial_branching_points: Sequence[float],
    kern: Optional[BranchKernelParam] = None,
) -> AssignGP:
    """
    Construct an MMBGP model consistent with the provided gene expression data.
    :param gene_expression: gene expression data you want to model
    :param initial_branching_points: initial branching points, one for each gene.
        Order should correspond to the order used in gene_expression.Y
    :param phi_constructor: a constructor for phi initial and prior values
    :param kern: the branching point kernel, see BranchKernelParam for details.
    :return: dictionary of log likelihood, GPflow model, Phi matrix, predictive set of points,
    mean and variance, hyperparameter values, posterior on branching time
    """
    phi_initial, phi_prior = phi_constructor.build()
    x_expanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(
        gene_expression.t
    )

    if not kern:
        kern = get_branching_point_kernel(branching_points=initial_branching_points)
    elif kern:
        # TODO: add a good error message
        np.testing.assert_array_equal(
            kern.Bv.numpy(), np.array(initial_branching_points)
        )

    model = AssignGP(
        gene_expression.t,
        x_expanded,
        gene_expression.Y,
        indices,
        kern=kern,
        phiInitial=phi_initial,
        phiPrior=phi_prior,
        multi=True,
    )
    return model


def construct_and_fit_assigngp_model(
    gene_expression: GeneExpressionData,
    phi_constructor: PhiConstructor,
    initial_branching_points: Sequence[float],
    optimiser: AssignGPOptimiser,
    kern: Optional[BranchKernelParam] = None,
) -> AssignGP:
    """
    Fit an MMBGP model to the provided gene expression data.
    :param gene_expression: gene expression data you want to model
    :param initial_branching_points: initial branching points, one for each gene.
        Order should correspond to the order used in gene_expression.Y
    :param phi_constructor: a constructor for phi initial and prior values
    :param optimiser: the optimiser to use for training
    :param kern: the branching point kernel, see BranchKernelParam for details.
    :return: dictionary of log likelihood, GPflow model, Phi matrix, predictive set of points,
    mean and variance, hyperparameter values, posterior on branching time
    """
    model = construct_assigngp_model(
        gene_expression=gene_expression,
        phi_constructor=phi_constructor,
        initial_branching_points=initial_branching_points,
        kern=kern,
    )
    trained_model = optimiser.train(model)

    return trained_model


@dataclasses.dataclass
class GaussianPredictions:
    """
    Return the key statistics for a branching model with a Gaussian posterior.

    Let the branching process be governed by latent GP f on the pre-branching point and g, h post-branching.
    Then we return the following quantities.
    * test points x,
    * mean prediction for f, g and h at x - in this specific order,
    * variance prediction for f, g, and h at x - in this specific order.
    """

    x: ColumnVector
    y_mean: Tuple[ColumnVector, ColumnVector, ColumnVector]
    y_var: Tuple[ColumnVector, ColumnVector, ColumnVector]


@dataclasses.dataclass
class TrainingOutcome:
    model: AssignGP
    predictions: GaussianPredictions
    learned_branching_points: ColumnVector


def get_predictions(model: AssignGP) -> GaussianPredictions:
    grid_on_unit_interval = np.linspace(0, 1, _DEFAULT_NUM_PREDICTIONS)

    means, variances = [], []

    for f in range(1, 4):
        # TODO: comment
        grid_on_unit_interval_as_column = grid_on_unit_interval.reshape(
            _DEFAULT_NUM_PREDICTIONS, 1
        )
        # TODO: what's going on here, surely we can simplify?
        test_x = np.hstack(
            (grid_on_unit_interval_as_column, grid_on_unit_interval_as_column * 0 + f)
        )

        mean, variance = model.predict_y(test_x)

        means.append(mean.numpy())
        variances.append(variance.numpy())

    return GaussianPredictions(
        x=grid_on_unit_interval,
        y_mean=tuple(means),  # type: ignore  # TODO: tell MyPy this is actually fine
        y_var=tuple(variances),  # type: ignore  # TODO: tell MyPy this is actually fine
    )


def get_training_outcome(trained_model: AssignGP) -> TrainingOutcome:
    return TrainingOutcome(
        model=trained_model,
        predictions=get_predictions(model=trained_model),
        learned_branching_points=trained_model.kernel.Bv.numpy(),
    )


def get_assigngp_with_target_bps(
    bps: Sequence[float], lengthscale: float, noise_variance: float
) -> AssignGP:
    """A simple wrapper that tweaks some key model construction parameters."""
    data = ToyBranchedData(bps, N=100)  # not used; need data to construct a model
    m = construct_assigngp_model(
        gene_expression=data,
        phi_constructor=SimplePhiConstructor(data, prior_confidence=0.65),
        initial_branching_points=[0.5] * data.num_genes,
    )

    # assign reasonable initial values for the BPs
    m.kernel.Bv.assign(np.array(bps))
    m.kernel.kern.lengthscales.assign(lengthscale)
    m.likelihood.variance.assign(noise_variance)
    return m
