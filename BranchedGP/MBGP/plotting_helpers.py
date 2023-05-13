from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .assigngp import AssignGP
from .data_generation import DEFAULT_COLOURS, BranchedData, Colours, GeneExpressionData
from .gene_expression_api import GeneExpressionModel
from .training_helpers import TrainingOutcome, get_training_outcome
from .VBHelperFunctions import plotBranchModel


def plot_samples(
    X,
    samples,
    BPs,
    fig: Optional[Figure] = None,
    axa: Optional[Sequence[Axes]] = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """
    X is in the XExpanded format, see MMBGP.VBHelperFunctions.GetFunctionIndexListGeneral,
    shape (N, 2).
    Samples have shape (num_samples, N, num_outputs).
    BPs have shape (num_outputs,).
    """
    num_samples, num_x, num_outputs = samples.shape
    assert X.shape[0] == num_x
    assert BPs.shape == (num_outputs,)

    if fig is None or axa is None:
        fig, axa = plt.subplots(num_outputs, 1, sharex=True, sharey=True)
        if num_outputs == 1:
            axa = [axa]

    M = 3  # number of functions
    for d in range(num_outputs):
        ax = axa[d]  # type: ignore  # TODO: tell MyPy this is actually fine
        b = BPs[d]

        for s in range(num_samples):
            for i in range(1, M + 1):  # pick out trunk, branch1, branch2
                t = X[X[:, 1] == i, 0]
                y = samples[s, X[:, 1] == i, d]

                # Cut off the trunk GP after the BP and the two branches before the BP
                idx = t < b if i == 1 else t >= b

                colour = DEFAULT_COLOURS[i - 1]
                ax.plot(t[idx], y[idx], "x", color=colour)

        # Add vertical lines for branch points
        v = ax.axis()
        ax.plot([b, b], v[-2:], "--r")

    return fig, axa  # type: ignore  # TODO: tell MyPy this is actually a sequence


def plot_detailed_fit(
    outcomes: TrainingOutcome,
    genes: GeneExpressionData,
    alpha: float = 0.01,
    title: bool = True,
    axa_per_row: int = 4,
) -> Tuple[Figure, Sequence[Axes]]:
    fig, axa = genes.plot(
        max_samples_per_gene=5000, alpha=alpha, axa_per_row=axa_per_row
    )
    if title:
        fig.suptitle(
            f"ELBO: {outcomes.model.training_loss():.2f}, "
            f"lengthscale: {outcomes.model.kernel.kern.lengthscales.numpy():.2f}, "
            f"variance: "
            f"{outcomes.model.kernel.kern.variance.numpy() + outcomes.model.likelihood.variance.numpy():.2f}, "
        )

    for ib, ax in enumerate(axa.flatten()):
        try:
            plotBranchModel(
                outcomes.learned_branching_points[ib],
                None,  # type: ignore  # TODO: this is actually fine, but need to convince MyPy
                None,
                genes.t,
                genes.Y,
                np.vstack((outcomes.predictions.x,) * 3),
                outcomes.predictions.y_mean,  # type: ignore  # TODO: this is actually fine, but need to convince MyPy
                outcomes.predictions.y_var,  # type: ignore  # TODO: this is actually fine, but need to convince MyPy
                outcomes.model.logPhi,
                fPlotVar=False,
                d=ib,
                ax=ax,
                fColorBar=False,
                fPlotPhi=False,
                show_legend=False,
            )
        except IndexError:
            # Work around empty axes in the axa object
            pass
    return fig, axa  # type: ignore  # TODO: tell MyPy this is actually a sequence


def plot_model_snapshot(
    model: AssignGP,
    genes: GeneExpressionData,
    alpha: float = 0.01,
    title: bool = True,
    axa_per_row: int = 4,
) -> Tuple[Figure, Sequence[Axes]]:
    details = get_training_outcome(model)
    fig, axa = plot_detailed_fit(
        details, genes, alpha=alpha, title=title, axa_per_row=axa_per_row
    )
    return fig, axa


def plot_gene_expression_model(
    data: BranchedData,  # Includes true branching points
    model: GeneExpressionModel,
    axa_per_row: int = 4,
    alpha: float = 0.01,
    linewidth: float = 3.0,
    colorarray: Optional[Colours] = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """Plotting code that does not require access to the model but takes as input predictions."""
    fig, axa = data.plot(
        max_samples_per_gene=5000, alpha=alpha, axa_per_row=axa_per_row
    )
    colorarray = colorarray or DEFAULT_COLOURS

    msg = "The model has been trained on different data than we're plotting."
    np.testing.assert_array_equal(data.t, model.data.t, err_msg=msg)
    np.testing.assert_array_equal(data.Y, model.data.Y, err_msg=msg)

    t = data.t  # TODO: we may want to pass in new test locations at some point
    predictions = model.predictions(t.reshape(-1, 1))

    for gene_idx in range(data.num_genes):
        ax = axa.flatten()[gene_idx]
        learned_branching_pt = model.branching_times[gene_idx]
        true_branching_pt = data.branching_points[gene_idx]

        for f in range(3):
            col = colorarray[f]
            state_predictions = predictions[f]

            if f == 0:
                idx = np.flatnonzero(t < learned_branching_pt)
            else:
                idx = np.flatnonzero(t >= learned_branching_pt)

            ax.plot(
                t[idx], state_predictions[idx, gene_idx], linewidth=linewidth, color=col
            )

        y_bounds = ax.axis()[-2:]

        ax.plot(
            [learned_branching_pt, learned_branching_pt],
            y_bounds,
            "--m",
            linewidth=linewidth,
            label="Estimated BP",
        )
        ax.plot(
            [true_branching_pt, true_branching_pt],
            y_bounds,
            "--b",
            linewidth=linewidth,
            label="True BP",
        )

    return fig, axa  # type: ignore  # TODO: tell MyPy this is actually a sequence
