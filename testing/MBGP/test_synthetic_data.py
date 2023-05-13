"""
Tests to check that we correctly infer the branching time for simple cases with synthetic data.
"""
import numpy as np
import pytest
import tensorflow as tf

from BranchedGP.MBGP.assigngp import get_branching_point_kernel
from BranchedGP.MBGP.data_generation import ToyBranchedData
from BranchedGP.MBGP.FitBranchingModel import FitModel
from BranchedGP.MBGP.training_helpers import (
    AssignGPOptimiser,
    ElvijsAmazingOptimiser,
    ElvijsRandomisingOptimiser,
    ScipyOptimiser,
    SimplePhiConstructor,
    construct_and_fit_assigngp_model,
    get_training_outcome,
)

_TRUE_BRANCHING_POINTS = [0.2, 0.4, 0.5, 0.6]
_NUM_DATA_POINTS = 30
_BRANCHING_SEARCH_STARTING_POINTS = [
    0.01,
    0.15,
    0.4,
    0.55,
]  # Random selection; note how .4 is a true branching point
_MODEL_FIT_CONFIG = dict(
    M=0, maxiter=100, priorConfidence=0.65, kervar=1.0, kerlen=1.0, likvar=0.01
)
_TOLERANCE = 0.01


@pytest.mark.parametrize("num_inputs", range(1, 5))
@pytest.mark.parametrize(
    "method", ["L-BFGS-B"]
)  # The tolerance has to be relaxed in order for CG to pass
def test_demo(num_inputs: int, method: str) -> None:
    """A test to automate some of the checks done in `notebooks/demo_simple_synthetic.py`."""
    data = ToyBranchedData(B=_TRUE_BRANCHING_POINTS[0:num_inputs], N=_NUM_DATA_POINTS)
    kern = get_branching_point_kernel(_BRANCHING_SEARCH_STARTING_POINTS[0:num_inputs])
    fit_details = FitModel(
        _BRANCHING_SEARCH_STARTING_POINTS,
        data.t,
        data.Y,
        data.state,
        **_MODEL_FIT_CONFIG,
        optimisation_method=method,
        kern=kern,
    )

    true_branching_points = _TRUE_BRANCHING_POINTS[0:num_inputs]
    inferred_branching_points = fit_details["Bmode"][0:num_inputs]

    for true_bp, inferred_bp in zip(true_branching_points, inferred_branching_points):
        assert (
            abs(true_bp - inferred_bp) < _TOLERANCE
        ), f"True branching points: {true_branching_points}. Inferred branching points: {inferred_branching_points}"


@pytest.mark.parametrize(
    "optimiser",
    [
        ScipyOptimiser(),
        ElvijsAmazingOptimiser(),
        ElvijsRandomisingOptimiser(num_samples=1),
    ],
)
@pytest.mark.parametrize("num_inputs", [1, 2, 4])
def test_demo__using_other_training_methods(
    optimiser: AssignGPOptimiser, num_inputs: int
) -> None:
    """A test to automate some of the checks done in `notebooks/demo_simple_synthetic.py`."""

    true_branching_points = _TRUE_BRANCHING_POINTS[0:num_inputs]

    data = ToyBranchedData(B=true_branching_points, N=_NUM_DATA_POINTS)

    trained_model = construct_and_fit_assigngp_model(
        data,
        phi_constructor=SimplePhiConstructor(data, prior_confidence=0.65),
        initial_branching_points=_BRANCHING_SEARCH_STARTING_POINTS[0:num_inputs],
        optimiser=ElvijsRandomisingOptimiser(base_optimiser=ScipyOptimiser()),
    )

    training_outcomes = get_training_outcome(trained_model)

    true_bps = np.array(true_branching_points)

    abs_bp_differences = np.abs(true_bps - training_outcomes.learned_branching_points)
    tolerance = _TOLERANCE * np.ones_like(true_bps)

    assert np.less(abs_bp_differences, tolerance).all(), (
        f"True branching points: {true_branching_points}. "
        f"Inferred branching points: {training_outcomes.learned_branching_points}"
    )


def test_equivalence() -> None:
    true_branching_points = _TRUE_BRANCHING_POINTS[0:2]
    data = ToyBranchedData(B=true_branching_points, N=_NUM_DATA_POINTS)

    initial_bps = np.ones((data.num_genes, 1)) * 0.4
    trained_model1 = construct_and_fit_assigngp_model(
        data,
        phi_constructor=SimplePhiConstructor(data, prior_confidence=0.65),
        initial_branching_points=initial_bps,  # type: ignore  # numpy can be consumed as a sequence
        optimiser=ScipyOptimiser(),
    )

    kern = get_branching_point_kernel(_BRANCHING_SEARCH_STARTING_POINTS[0:2])
    fit_details = FitModel(
        [0.4],
        data.t,
        data.Y,
        data.state,
        **_MODEL_FIT_CONFIG,
        kern=kern,
    )
    trained_model2 = fit_details["model"]

    np.testing.assert_array_equal(
        trained_model1.BranchingPoints, trained_model2.BranchingPoints
    )


@pytest.fixture(autouse=True, scope="function")
def fixed_random_seed() -> None:
    tf.random.set_seed(42)
