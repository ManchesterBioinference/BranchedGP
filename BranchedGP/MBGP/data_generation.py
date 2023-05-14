"""
Toy function to generate toy data
"""
import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

Colours = Tuple[str, str, str]
# 0th colour corresponds to the pre-branching point colour
# 1st colour corresponds to the post-branching data points with state value 2
# 2nd colour corresponds to the post-branching data points with state value 3

DEFAULT_BP_COLOUR = "darkolivegreen"
DEFAULT_COLOURS: Colours = (DEFAULT_BP_COLOUR, "peru", "mediumvioletred")

Vector = np.ndarray  # shape: (n, )
ColumnVector = np.ndarray  # shape: (n, 1)
Matrix = np.ndarray  # shape: (n, k)
BranchingMatrix = np.ndarray  # shape: (3, n). Row correspondence:
# 0 is the pre-branching GP, 1 is one of the post-branching GPs and 2 is the other one
AxesMatrix = np.ndarray  # Just a helper type to indicate plots arranged in a matrix


class GeneExpressionDataValidationError(ValueError):
    """Raised when data provided to construct BranchedData is internally inconsistent."""


@dataclasses.dataclass(frozen=True)
class GeneExpressionData:
    t: Vector  # Pseudotime. Shape (N, ).
    Y: Matrix  # Gene expressions. Shape (N, K), where K is the number of genes.
    state: Vector  # State indicator array. Shape (N, ). Values in {1, 2, 3}..
    gene_labels: Optional[Sequence[str]]  # Length of sequence is K.

    def __post_init__(self) -> None:
        self._validate()

    @property
    def num_genes(self) -> int:
        return self.Y.shape[1]

    def _validate(self) -> None:
        if len(self.t.shape) != 1:
            raise GeneExpressionDataValidationError(
                f"Expected pseudotime to have shape (N, ), instead got {self.t.shape}"
            )

        if len(self.Y.shape) != 2:
            raise GeneExpressionDataValidationError(
                f"Expected gene expressions to have shape (N, K), instead got {self.Y.shape}"
            )

        if self.gene_labels and len(self.gene_labels) != self.Y.shape[1]:
            raise GeneExpressionDataValidationError(
                f"Expected {self.Y.shape[1]} gene labels, instead got {len(self.gene_labels)}"
            )

        if self.t.shape[0] != self.Y.shape[0]:
            raise GeneExpressionDataValidationError(
                f"Pseudotime and gene expression should have the same number of rows, "
                f"instead got {self.t.shape[0]} and {self.Y.shape[0]}"
            )

        if len(self.state.shape) != 1:
            raise GeneExpressionDataValidationError(
                f"Expected state array to be a vector, "
                f"instead got {self.state.shape}"
            )

        if self.state.shape[0] != self.t.shape[0]:
            raise GeneExpressionDataValidationError(
                f"Pseudotime and state array should have the same number of rows, "
                f"instead got {self.t.shape[0]} and {self.state.shape[0]}"
            )

        # TODO: we could also double-check that the branching points do not disagree
        #  with what the state labels say

    def plot(
        self,
        fig: Optional[Figure] = None,
        axa: Optional[AxesMatrix] = None,
        colours: Colours = DEFAULT_COLOURS,
        alpha: float = 0.4,
        marker_size_in_points: int = 40,
        max_samples_per_gene: int = 100,
        axa_per_row: int = 5,
        y_limits: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Figure, AxesMatrix]:
        """
        Plot gene expressions on the provided axa and return the axa.

        If no axa are provided, create them.
        """
        if axa is None or fig is None:
            num_subplots = self.num_genes
            fig, axa = self._get_axa_grid(
                num_subplots=num_subplots, axa_per_row=axa_per_row
            )

        gene_labels = self.gene_labels or (f"gene {i}" for i in range(self.num_genes))

        if self.t.size > max_samples_per_gene:
            row_idx = np.random.choice(self.t.size, max_samples_per_gene)
        else:
            row_idx = np.arange(self.t.size)

        for gene_idx, gene_label in enumerate(gene_labels):
            ax = self._get_ax(gene_idx, axa, axa_per_row=axa_per_row)

            ax.set_title(gene_label)
            if y_limits:
                y_min, y_max = y_limits
                ax.set_ylim(bottom=y_min, top=y_max)

            # different colours for different states
            for i in range(1, 4):
                idx = np.logical_and(row_idx, self.state == i)

                colour = colours[i - 1]

                ax.scatter(
                    self.t[idx],
                    self.Y[idx, gene_idx],
                    s=marker_size_in_points,
                    c=colour,
                    alpha=alpha,
                )

        return fig, axa

    def plot_rolling_means(
        self,
        fig: Figure,
        axa: AxesMatrix,
        colours: Colours = DEFAULT_COLOURS,
        rolling_window_length: int = 10,
        line_width: int = 3,
    ) -> Tuple[Figure, AxesMatrix]:
        """Plot smoothed lines for the gene expressions."""
        gene_labels = self.gene_labels or (f"gene {i}" for i in range(self.num_genes))
        axa_per_row = axa.shape[1]

        def rolling_avg(arr: Vector, window_length: int) -> Vector:
            return scipy.ndimage.filters.uniform_filter1d(arr, size=window_length)

        for gene_idx, gene_label in enumerate(gene_labels):
            ax = self._get_ax(gene_idx, axa, axa_per_row=axa_per_row)

            ax.set_title(gene_label)

            # different colours for different states
            for i in range(1, 4):
                colour = colours[i - 1]
                state_mask = self.state == i
                sorted_idx = np.argsort(self.t[state_mask])

                ax.plot(
                    self.t[state_mask][sorted_idx],
                    rolling_avg(
                        self.Y[state_mask, gene_idx][sorted_idx],
                        window_length=rolling_window_length,
                    ),
                    c=colour,
                    linewidth=line_width,
                )

        return fig, axa

    @staticmethod
    def _get_ax(i: int, axa: AxesMatrix, axa_per_row: int) -> plt.Axes:
        if len(axa.shape) == 1:
            return axa[i]
        else:
            row_idx = i // axa_per_row
            col_idx = i % axa_per_row
            return axa[row_idx, col_idx]

    @staticmethod
    def _get_axa_grid(num_subplots: int, axa_per_row: int) -> Tuple[Figure, AxesMatrix]:
        nrows = 1 + ((num_subplots - 1) // axa_per_row)
        ncols = min([axa_per_row, num_subplots])
        scaling_factor = 3  # for making the default plot a reasonable size
        fig, axa = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * scaling_factor, nrows * scaling_factor),
            sharex=True,
            sharey=False,
        )

        # subplots returns an unwrapped element if only one subplot is requested
        if nrows == ncols == 1:
            return fig, np.array([[axa]])

        return fig, axa


@dataclasses.dataclass(frozen=True)
class BranchedData(GeneExpressionData):
    branching_points: Sequence[
        float
    ]  # True branching point locations. K elements expected.

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        super()._validate()

        if len(self.branching_points) != self.Y.shape[1]:
            raise GeneExpressionDataValidationError(
                f"There should be one branching point per gene, "
                f"instead got {len(self.branching_points)} branching points and {self.Y.shape[1]} genes."
            )

    def plot(  # type: ignore  # TODO: unify the interface with super() to satisfy MyPy
        self,
        fig: Optional[Figure] = None,
        axa: Optional[AxesMatrix] = None,
        colours: Colours = DEFAULT_COLOURS,
        alpha: float = 0.4,
        marker_size_in_points: int = 40,
        true_bp_location_line: str = "--b",
        max_num_genes: int = 100,
        max_samples_per_gene: int = 100,
        axa_per_row: int = 5,
        y_limits: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Figure, AxesMatrix]:
        """
        Plot gene expression on the provided axa, colour-code the different states and
        plot a line for the true BP location.

        The provided axa are expected to be an index-able matrix with `axa_per_row` columns.

        If no axa are provided, create them.
        """
        if axa is None or fig is None:
            num_subplots = min([max_num_genes, self.num_genes])
            fig, axa = self._get_axa_grid(
                num_subplots=num_subplots, axa_per_row=axa_per_row
            )

        gene_labels = self.gene_labels or (f"gene {i}" for i in range(self.num_genes))

        for gene_idx, gene_label in enumerate(gene_labels):
            ax = self._get_ax(gene_idx, axa, axa_per_row=axa_per_row)
            gene_branching_point = self.branching_points[gene_idx]

            for i in range(1, 4):
                col = colours[i - 1]
                ax.set_title(gene_label)

                if i == 1:  # the trunk is really indicated by the branching time
                    state_mask = self.t <= gene_branching_point
                else:
                    state_mask = np.logical_and(
                        self.state == i, self.t >= gene_branching_point
                    )

                ax.scatter(
                    self.t[state_mask],
                    self.Y[state_mask, gene_idx],
                    s=marker_size_in_points,
                    c=col,
                    alpha=alpha,
                )

                if y_limits:
                    y_min, y_max = y_limits
                    ax.set_ylim(bottom=y_min, top=y_max)

            # plot a vertical line for true branching point location
            y_bounds = ax.axis()[-2:]  # will equal y_limits if these are provided
            ax.plot(
                [gene_branching_point, gene_branching_point],
                y_bounds,
                true_bp_location_line,
                label="True BP",
            )

        return fig, axa


class ToyBranchedData(BranchedData):
    def __init__(
        self, B=(0.1, 0.5, 0.8), N=20, gene_labels: Optional[Sequence[str]] = None
    ):
        t = np.linspace(0, 1, N)
        Y = np.zeros((N, len(B)))

        # Create global branches
        state = np.ones(N, dtype=int)
        state[::2] = 2
        state[1::2] = 3
        for ib, b in enumerate(B):
            idx2 = np.logical_and(t > b, state == 2)
            idx3 = np.logical_and(t > b, state == 3)
            Y[idx2, ib] = t[idx2] ** 2 - b ** 2
            Y[idx3, ib] = -t[idx3] ** 2 + b ** 2

        super().__init__(
            t=t, Y=Y, branching_points=B, state=state, gene_labels=gene_labels
        )


class ToyGeneExpressionData(GeneExpressionData):
    def __init__(self, B=(0.1, 0.5, 0.8), N=20):
        t = np.linspace(0, 1, N)
        Y = np.zeros((N, len(B)))

        # Create global branches
        state = np.ones(N, dtype=int)
        state[::2] = 2
        state[1::2] = 3
        for ib, b in enumerate(B):
            idx2 = np.logical_and(t > b, state == 2)
            idx3 = np.logical_and(t > b, state == 3)
            Y[idx2, ib] = t[idx2] ** 2 - b ** 2
            Y[idx3, ib] = -t[idx3] ** 2 + b ** 2

        super().__init__(t=t, Y=Y, state=state, gene_labels=None)


class ToyWigglyBranchedData(BranchedData):
    """
    Toy synthetic dataset with predetermined branching points and
    sinusoidal base expression values.

    To get a sense for the dataset, use::
        >>> ToyWigglyBranchedData(branching_points=[0.2, 0.7]).plot()

    Change the default construction arguments to alter behaviour.
    """

    def __init__(
        self,
        branching_points: Sequence[float],
        num_data_points: int,
        wiggle_frequency: Optional[float] = 5.0 * 2 * np.pi,
        wiggle_amplitude: Optional[float] = 0.1,
        separation_amplitude: Optional[float] = 3.0,
    ) -> None:
        """
        :param branching_points: the location of the branching points
        :param num_data_points: the number of total data points (that is, how many pseudo-time steps)
        :param wiggle_frequency: the frequency of wiggles
        :param wiggle_amplitude: the amplitude of wiggles
        :param separation_amplitude: how quickly the branches move away from each other
        """
        for b in branching_points:
            assert (
                0 <= b <= 1
            ), f"Branching points should all be in [0, 1], instead got {branching_points}."

        # Data generation
        N = num_data_points
        num_output_dims = len(branching_points)
        t = np.linspace(0, 1, N)
        baseline_wiggles = np.sin(t * wiggle_frequency) * wiggle_amplitude  # type: ignore
        # The transposes ensure baseline wiggles are applied along the correct axis
        Y = (np.ones((N, num_output_dims)).T * baseline_wiggles).T

        # Create global branches
        state = np.ones(N, dtype=int)
        state[::2] = 2
        state[1::2] = 3

        for ib, b in enumerate(branching_points):
            idx2 = np.logical_and(t > b, state == 2)
            idx3 = np.logical_and(t > b, state == 3)

            Y[idx2, ib] = baseline_wiggles[idx2] - t[: sum(idx2)] * separation_amplitude  # type: ignore
            Y[idx3, ib] = baseline_wiggles[idx3] + t[: sum(idx3)] * separation_amplitude  # type: ignore

        super().__init__(
            t=t, Y=Y, state=state, gene_labels=None, branching_points=branching_points
        )


if __name__ == "__main__":
    ToyBranchedData(B=np.linspace(0.1, 0.9, 10)).plot()
    ToyGeneExpressionData().plot()
    ToyWigglyBranchedData([0.3, 0.5], 50).plot()
