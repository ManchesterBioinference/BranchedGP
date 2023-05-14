import pytest

from BranchedGP.MBGP.data_generation import GeneExpressionData, ToyBranchedData
from BranchedGP.MBGP.training_helpers import FunkyPrior


@pytest.mark.parametrize("uninformative_until", [0, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("confidence", [0, 0.3, 0.5, 0.7, 1.0])
def test_funky_prior__is_not_smoking(
    dummy_data: GeneExpressionData,
    uninformative_until: float,
    confidence: float,
) -> None:
    """If uninformative_until is set to 0, then it should be equivalent to SimplePhiConstructor"""
    FunkyPrior(
        dummy_data,
        uninformative_until=uninformative_until,
        informative_prior_confidence=confidence,
    ).build()


@pytest.fixture
def dummy_data() -> GeneExpressionData:
    return ToyBranchedData(B=(0.5,), N=5)
