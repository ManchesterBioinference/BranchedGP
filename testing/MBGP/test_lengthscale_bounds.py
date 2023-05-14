""" Tests that check the lengthscales can be bounded via priors and transforms. """
from typing import Sequence

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from BranchedGP.MBGP.assigngp import get_branching_point_kernel
from BranchedGP.MBGP.data_generation import ToyWigglyBranchedData
from BranchedGP.MBGP.FitBranchingModel import FitModel

_TRUE_BP: float = 0.5
_NUM_DATA_POINTS: int = 50
_WIGGLE_FREQUENCY: float = 10.0
_BP_SEARCH_STARTING_POINTS: Sequence[float] = [0.5]


def test_lengthscale_prior_respected() -> None:
    # First we train without the prior and check that we learn a sufficiently small lengthscale
    data = ToyWigglyBranchedData(
        [_TRUE_BP], num_data_points=_NUM_DATA_POINTS, wiggle_frequency=_WIGGLE_FREQUENCY
    )
    kern = get_branching_point_kernel(branching_points=_BP_SEARCH_STARTING_POINTS)
    fit_details = FitModel(
        _BP_SEARCH_STARTING_POINTS, data.t, data.Y, data.state, kern=kern, M=0
    )

    # The magic constant was obtained by lengthscale inspection
    assert fit_details["model"].kernel.kern.lengthscales.numpy() < 0.3

    # Now construct a fresh kernel with a prior on the lengthscales and
    # check that we learn a longer lengthscale
    lengthscale_lower_bound = 0.5
    kern = get_branching_point_kernel(branching_points=_BP_SEARCH_STARTING_POINTS)
    kern.kern.lengthscales.prior = tfp.distributions.Uniform(
        tf.constant(lengthscale_lower_bound, dtype=gpflow.default_float()),
        tf.constant(1.5, dtype=gpflow.default_float()),
    )

    fit_details = FitModel(
        _BP_SEARCH_STARTING_POINTS, data.t, data.Y, data.state, kern=kern, M=0
    )
    # The magic constant is
    assert (
        fit_details["model"].kernel.kern.lengthscales.numpy() >= lengthscale_lower_bound
    )


def test_lengthscale_transformation_applied() -> None:
    # First we train without the transfork and check that we learn a sufficiently small lengthscale
    data = ToyWigglyBranchedData(
        [_TRUE_BP], num_data_points=_NUM_DATA_POINTS, wiggle_frequency=_WIGGLE_FREQUENCY
    )
    kern = get_branching_point_kernel(branching_points=_BP_SEARCH_STARTING_POINTS)
    fit_details = FitModel(
        _BP_SEARCH_STARTING_POINTS, data.t, data.Y, data.state, kern=kern, M=0
    )

    # The magic constant was obtained by lengthscale inspection
    assert fit_details["model"].kernel.kern.lengthscales.numpy() < 0.3

    # Now construct a fresh kernel with a prior on the lengthscales and
    # check that we learn a longer lengthscale
    lengthscale_lower_bound = 0.5
    base_kernel = gpflow.kernels.SquaredExponential()
    base_kernel.lengthscales = gpflow.Parameter(
        tf.Variable(1.0, dtype=gpflow.default_float()),
        transform=tfp.bijectors.Sigmoid(
            low=tf.constant(0.5, dtype=gpflow.default_float()),
            high=tf.constant(1.5, dtype=gpflow.default_float()),
        ),
    )
    kern = get_branching_point_kernel(
        branching_points=_BP_SEARCH_STARTING_POINTS, base_kernel=base_kernel
    )

    fit_details = FitModel(
        _BP_SEARCH_STARTING_POINTS, data.t, data.Y, data.state, kern=kern, M=0
    )
    # The magic constant is
    assert (
        fit_details["model"].kernel.kern.lengthscales.numpy() >= lengthscale_lower_bound
    )
