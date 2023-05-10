import numpy as np
import pytest

from BranchedGP.MBGP.assigngp import AssignGP
from BranchedGP.MBGP.data_generation import ToyBranchedData
from BranchedGP.MBGP.plotting_helpers import (
    plot_detailed_fit,
    plot_model_snapshot,
    plot_samples,
)
from BranchedGP.MBGP.training_helpers import (
    SimplePhiConstructor,
    construct_assigngp_model,
    get_training_outcome,
)
from BranchedGP.MBGP.VBHelperFunctions import GetFunctionIndexListGeneral


def test_plot_samples__is_not_smoking(assigngp) -> None:
    x_new = np.linspace(0, 1, 50)
    x_expanded, indices, _ = GetFunctionIndexListGeneral(x_new)

    samples = assigngp.predict_f_samples(x_expanded, num_samples=43)
    plot_samples(x_expanded, samples=samples.numpy(), BPs=assigngp.BranchingPoints)


def test_plot_detailed_fit__is_not_smoking(
    assigngp: AssignGP, toy_data: ToyBranchedData
) -> None:
    outcomes = get_training_outcome(assigngp)
    plot_detailed_fit(outcomes, toy_data)


def test_plot_model_snapshot__is_not_smoking(
    assigngp: AssignGP, toy_data: ToyBranchedData
) -> None:
    plot_model_snapshot(assigngp, toy_data)


@pytest.fixture(name="toy_data")
def _toy_data() -> ToyBranchedData:
    return ToyBranchedData(B=[0.1, 0.5, 0.9], N=50)


@pytest.fixture(name="assigngp")
def _assigngp(toy_data: ToyBranchedData) -> AssignGP:
    model = construct_assigngp_model(
        gene_expression=toy_data,
        phi_constructor=SimplePhiConstructor(toy_data, prior_confidence=0.65),
        initial_branching_points=[0.5, 0.5, 0.5],
    )
    return model
