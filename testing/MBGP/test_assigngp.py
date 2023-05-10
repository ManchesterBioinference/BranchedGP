""" Simple tests for some specific AssignGP behaviours. """
import numpy as np
import pytest
import tensorflow_probability as tfp

from BranchedGP.MBGP import VBHelperFunctions
from BranchedGP.MBGP.assigngp import AssignGP
from BranchedGP.MBGP.data_generation import ToyBranchedData
from BranchedGP.MBGP.FitBranchingModel import GetInitialConditionsAndPrior
from BranchedGP.MBGP.sampling_helpers import sample_prior_as_branched_data

_TRUE_BRANCHING_POINTS = [0.1, 0.3, 0.5, 0.7, 0.9]
_NUM_DATA_POINTS = 50
_PRIOR_CONFIDENCE = 0.65


def test_assigngp_defaults__to_kern_with_sigmoid_transform(
    simple_assigngp: AssignGP,
) -> None:
    """
    We test that the default construction of AssignGP constrains branching points to [0, 1] via the Sigmoid transform.
    """
    assert isinstance(simple_assigngp.kernel.Bv.transform, tfp.bijectors.Sigmoid)


# x_extended rows will always be divisible by 3, so let's check for unwanted broadcasting by adding 3 samples to the mix
@pytest.mark.parametrize("num_samples", [1, 43, 3])
@pytest.mark.parametrize("full_cov", [False, True])
def test_f_samples__is_not_smoking(
    num_samples: int, full_cov: bool, simple_assigngp: AssignGP
) -> None:
    x_new = np.linspace(0, 1, 100)
    x_expanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(x_new)

    samples = simple_assigngp.predict_f_samples(
        x_expanded, num_samples, full_cov=full_cov
    )
    assert samples.shape == (
        num_samples,
        x_expanded.shape[0],
        len(_TRUE_BRANCHING_POINTS),
    )


# x_extended rows will always be divisible by 3, so let's check for unwanted broadcasting by adding 3 samples to the mix
@pytest.mark.parametrize("num_samples", [1, 43, 3])
def test_prior_samples__is_not_smoking(
    num_samples: int, simple_assigngp: AssignGP
) -> None:
    x_new = np.linspace(0, 1, 100)
    x_expanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(x_new)

    samples = simple_assigngp.sample_prior(x_expanded, num_samples)
    assert samples.shape == (
        num_samples,
        x_expanded.shape[0],
        len(_TRUE_BRANCHING_POINTS),
    )


@pytest.mark.parametrize("num_data_points", [1, 10, 43, 100])
def test_prior_samples_as_branched_data__is_not_smoking(
    num_data_points: int, simple_assigngp: AssignGP
) -> None:
    branched_data = sample_prior_as_branched_data(simple_assigngp, num_data_points)
    assert branched_data.num_genes == len(_TRUE_BRANCHING_POINTS)
    np.testing.assert_array_equal(
        branched_data.branching_points, simple_assigngp.BranchingPoints
    )
    assert branched_data.Y.shape == (num_data_points, len(_TRUE_BRANCHING_POINTS))


@pytest.fixture(name="simple_assigngp")
def _simple_assigngp() -> AssignGP:
    data = ToyBranchedData(B=_TRUE_BRANCHING_POINTS, N=_NUM_DATA_POINTS)
    # The variable setup is taken from MMBGP.FitBranchingModel.FitModel
    phi_initial, phi_prior = GetInitialConditionsAndPrior(
        data.state, _PRIOR_CONFIDENCE, infPriorPhi=True
    )
    phi_prior = np.c_[np.zeros(phi_prior.shape[0])[:, None], phi_prior]
    x_expanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(data.t)
    m = AssignGP(
        data.t,
        x_expanded,
        data.Y,
        indices,
        phiInitial=phi_initial,
        phiPrior=phi_prior,
        multi=True,
    )
    return m
