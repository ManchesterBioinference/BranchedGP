""" Tests for the data classes. """
from typing import Optional, Sequence

import numpy as np
import pytest

from BranchedGP.MBGP.data_generation import (
    BranchedData,
    GeneExpressionData,
    GeneExpressionDataValidationError,
    Matrix,
    ToyWigglyBranchedData,
    Vector,
)


@pytest.mark.parametrize("gene_labels", [None, ["0"]])
def test_can_construct_data(gene_labels: Optional[Sequence[str]]) -> None:
    GeneExpressionData(
        t=np.array([1.0]),
        Y=np.array([[2.0]]),
        state=np.array([1]),
        gene_labels=None,
    )


@pytest.mark.parametrize(
    "t, Y, state, gene_labels",
    [
        # shapes of t and Y inconsistent
        (
            np.array([1.0, 2.0]),
            np.array([[3.0]]),
            np.array([1]),
            None,
        ),
        # shapes of t and state inconsistent
        (
            np.array([1.0]),
            np.array([[3.0]]),
            np.array([1, 2]),
            None,
        ),
        # t not a column array
        (
            np.array([[1.0]]),
            np.array([[2.0]]),
            np.array([3]),
            None,
        ),
        # Y not a matrix
        (
            np.array([1.0]),
            np.array([2.0]),
            np.array([3]),
            None,
        ),
        # state not a column array
        (
            np.array([1.0]),
            np.array([2.0]),
            np.array([[3]]),
            None,
        ),
    ],
)
def test_validation(
    t: Vector,
    Y: Matrix,
    state: Vector,
    gene_labels: Optional[Sequence[str]],
) -> None:
    """Test that inconsistent construction arguments are caught and raise the appropriate exception."""
    with pytest.raises(GeneExpressionDataValidationError):
        GeneExpressionData(t=t, Y=Y, state=state, gene_labels=gene_labels)


def test_gene_expression_plot__is_not_smoking() -> None:
    data = GeneExpressionData(
        t=np.array([1.0]),
        Y=np.array([[2.0]]),
        state=np.array([1]),
        gene_labels=None,
    )
    fig, axa = data.plot()
    data.plot_rolling_means(fig, axa)


def test_gene_expression_plot() -> None:
    """Let's check that the right genes are plotted in the right locations."""
    fig, axa = GeneExpressionData(
        t=np.arange(10),
        Y=np.arange(100).reshape(10, 10),
        state=np.ones((10,)),
        gene_labels=None,
    ).plot()

    for row in range(2):
        for column in range(5):
            ax = axa[row, column]
            assert ax.title.get_text() == f"gene {row * 5 + column}", (
                f"Plotting on a (2, 5) grid. Expected gene {row * 5 + column} in "
                f"the ({row}, {column}) position. Instead got {ax.title.get_text()}"
            )


def test_branching_data_plot__is_not_smoking() -> None:
    data = BranchedData(
        t=np.array([1.0]),
        Y=np.array([[2.0]]),
        state=np.array([1]),
        branching_points=[0.5],
        gene_labels=None,
    )
    fig, axa = data.plot()
    data.plot_rolling_means(fig, axa)


def test_branched_data_plot() -> None:
    """Let's check that the right genes are plotted in the right locations."""
    fig, axa = BranchedData(
        t=np.arange(10),
        Y=np.arange(100).reshape(10, 10),
        state=np.ones((10,)),
        branching_points=[0.1] * 10,
        gene_labels=None,
    ).plot()

    for row in range(2):
        for column in range(5):
            ax = axa[row, column]
            assert ax.title.get_text() == f"gene {row * 5 + column}", (
                f"Plotting on a (2, 5) grid. Expected gene {row * 5 + column} in "
                f"the ({row}, {column}) position. Instead got {ax.title.get_text()}"
            )


def test_can_construct__toy_wiggly_data() -> None:
    ToyWigglyBranchedData(branching_points=[0.1, 0.5], num_data_points=50)
