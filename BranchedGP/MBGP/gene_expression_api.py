"""
This module defines the main interfaces for branching time and cell label predictors.

Just a wrapper for comparing different types of models.
"""
import abc
from typing import List, Sequence, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

from .assigngp import AssignGP
from .data_generation import ColumnVector, GeneExpressionData


class GeneExpressionModel(abc.ABC):
    """
    The main interface for gene expression modelling.
    """

    @property
    @abc.abstractmethod
    def data(self) -> GeneExpressionData:
        """
        The gene expression data that we want to model.
        N samples, K genes.

        Includes initial state estimates.
        """

    @property
    @abc.abstractmethod
    def branching_times(self) -> np.ndarray:
        """The branching times predicted by the model, shape (K,)."""

    @property
    @abc.abstractmethod
    def state(self) -> np.ndarray:
        """
        The state predicted by the model for each gene and each sample in the training set.
        Shape (N, K).

        State enumeration:
        1 - trunk
        2 - branch a
        3 - branch b
        """

    @abc.abstractmethod
    def predictions(self, t: ColumnVector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A triple of predictions for the (mean) of the branch value,
        one each for states 1, 2, and 3.

        If t has shape (M, ), then each prediction is of shape (M, K).
        """


class MBGP(GeneExpressionModel):
    """
    Multivariate branching Gaussian Process model.

    This is largely just an experimental wrapper for trained models,
    so can be created after training.
    """

    def __init__(self, model: AssignGP, data: GeneExpressionData) -> None:
        self._model = model
        self._data = data

    @property
    def data(self) -> GeneExpressionData:
        return self._data

    @property
    def branching_times(self) -> np.ndarray:
        return self._model.BranchingPoints

    @property
    def state(self) -> np.ndarray:
        # TODO: can probably rewrite the following as pure numpy code
        state_labels = np.zeros_like(self._data.Y)
        num_cells = self._data.t.shape[0]
        phi = self._model.GetPhi()

        for gene_idx in range(self._data.num_genes):
            learned_bp_idx = int(self.branching_times[gene_idx] * num_cells)

            for sample_idx in range(num_cells):
                if sample_idx < learned_bp_idx:  # Assumes ordered time
                    inferred_label = 1  # trunk
                else:
                    inferred_label_idx = np.argmax(phi[sample_idx])  # 0, 1 or 2
                    inferred_label = inferred_label_idx + 1  # type: ignore  # TODO: figure out how to convince MyPy

                state_labels[sample_idx, gene_idx] = inferred_label

        return state_labels

    def predictions(self, t: ColumnVector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = []

        for f in range(1, 4):
            # TODO: what's going on here, surely we can simplify?
            test_x = np.hstack((t, t * 0 + f))
            mean, _ = self._model.predict_y(test_x)

            means.append(mean.numpy())

        return means  # type: ignore  # TODO: rewrite to convince MyPy this is ok


class ManyBGPs(GeneExpressionModel):
    """
    Many branching Gaussian Process models for a multi-gene problem.

    This is largely just an experimental wrapper for trained models,
    so can be created after training.

    Each gene is modelled independently, which means there's potential for
    cell assignment inconsistencies (that is, a cell can be assigned
    to one branch on one gene, but a different branch on another gene,
    which does not make biological sense).
    """

    def __init__(self, models: Sequence[AssignGP], data: GeneExpressionData) -> None:
        assert (
            len(models) == data.num_genes
        ), f"Expected a BGP per gene, instead got {len(models)} and {data.num_genes} genes"
        self._models = models
        self._data = data

    @property
    def data(self) -> GeneExpressionData:
        return self._data

    @property
    def branching_times(self) -> np.ndarray:
        bps = []
        for model in self._models:
            model_bps = model.BranchingPoints
            assert len(model_bps) == 1, (
                f"Each Branching Gaussian Process should model one gene, "
                f"instead got {len(model_bps)} genes modelled by {model}"
            )
            bps.append(model_bps[0])

        return np.array(bps)

    @property
    def state(self) -> np.ndarray:
        # TODO: can probably rewrite the following as pure numpy code
        state_labels = np.zeros_like(self._data.Y)
        num_cells = self._data.t.shape[0]

        for gene_idx in range(self._data.num_genes):
            phi = self._models[gene_idx].GetPhi()
            learned_bp_idx = int(self.branching_times[gene_idx] * num_cells)

            for sample_idx in range(num_cells):
                if sample_idx < learned_bp_idx:  # Assumes ordered time
                    inferred_label = 1  # trunk
                else:
                    inferred_label_idx = np.argmax(phi[sample_idx])  # 0, 1 or 2
                    inferred_label = inferred_label_idx + 1  # type: ignore  # TODO: figure out how to convince MyPy

                state_labels[sample_idx, gene_idx] = inferred_label

        return state_labels

    def predictions(self, t: ColumnVector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means: Tuple[List[float], List[float], List[float]] = (
            [],
            [],
            [],
        )  # state 1, 2, 3 respectively

        for gene_idx in range(self._data.num_genes):
            model = self._models[gene_idx]

            for f in range(1, 4):
                # TODO: what's going on here, surely we can simplify?
                test_x = np.hstack((t, t * 0 + f))
                mean, _ = model.predict_y(test_x)

                means[f - 1].append(mean.numpy())

        return np.hstack(means[0]), np.hstack(means[1]), np.hstack(means[2])


SplineModel = Tuple[UnivariateSpline, UnivariateSpline, UnivariateSpline]
# The splines correspond to state 1 (trunk), 2 (branch a), and 3 (branch b) respectively


class SplineBEAM(GeneExpressionModel):
    """
    Spline-based BEAM from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5330805/.

    Each gene is modelled independently, which means there's potential for
    cell assignment inconsistencies (that is, a cell can be assigned
    to one branch on one gene, but a different branch on another gene,
    which does not make biological sense).
    """

    def __init__(self, data: GeneExpressionData, initial_bps: Sequence[float]) -> None:
        assert len(initial_bps) == data.num_genes, (
            f"Expected an initial branching point per gene, "
            f"instead got {len(initial_bps)} branching points and {data.num_genes} genes"
        )
        self._data = data

        self._models = self._train_spline_beam_models(data, initial_bps)

    def _train_spline_beam_models(
        self,
        data: GeneExpressionData,
        initial_bps: Sequence[float],
    ) -> Sequence[SplineModel]:
        models = []
        for i in range(data.num_genes):
            sliced_data = GeneExpressionData(
                t=data.t,
                Y=data.Y[:, i].reshape(-1, 1),
                state=data.state,
                gene_labels=[f"{i}"],
            )
            model = self._train_spline_beam(
                sliced_data, initial_branching_pt=initial_bps[i]
            )
            models.append(model)

        return models

    @staticmethod
    def _train_spline_beam(
        data: GeneExpressionData,
        initial_branching_pt: float,
        use_all_trunk: bool = True,
    ) -> SplineModel:
        t = data.t
        assert (
            data.num_genes == 1
        ), f"Expected to fit a spline to a single gene, instead got {data.num_genes}"
        y = data.Y.flatten()

        # create labels
        mtrunk = y[1:10].mean()
        labels = np.ones_like(y)

        # TODO: use ground truth state? Typically this would come from Monocle
        labels[np.logical_and(t >= initial_branching_pt, y >= mtrunk)] = 2
        labels[np.logical_and(t >= initial_branching_pt, y < mtrunk)] = 3

        # TODO: this isn't right
        #   Cell assignments should be random

        t_trunk = t[labels == 1]
        y_trunk = y[labels == 1]

        splines = []

        for i in range(3):
            tb = t[labels == i + 1]
            yb = y[labels == i + 1]

            if i == 0:
                tc = t_trunk
                yc = y_trunk
            else:
                if use_all_trunk:
                    t_trunk = t_trunk
                    y_trunk = y_trunk
                else:
                    # TODO: what's going on here?
                    t_trunk = t_trunk[(i - 1) :: 2]
                    y_trunk = y_trunk[(i - 1) :: 2]

                tc = np.hstack([t_trunk, tb])
                yc = np.hstack([y_trunk, yb])

            f = UnivariateSpline(tc, yc, s=20)
            splines.append(f)

        return splines  # type: ignore  # TODO: convince MyPy we're returning the right amount of splines

    @property
    def data(self) -> GeneExpressionData:
        return self._data

    @property
    def branching_times(self) -> np.ndarray:
        bps = []
        t = np.linspace(0, 1, 100)
        for model in self._models:
            spline_trunk, spline_2, spline_3 = model

            state_3_above_2 = spline_3(t) > spline_2(t)
            at_end_state_3_above_2 = state_3_above_2[-1]
            idx_bp = 99
            while (state_3_above_2[idx_bp] is at_end_state_3_above_2) and idx_bp >= 0:
                idx_bp -= 1

            bp = t[idx_bp]
            bps.append(bp)

        return np.array(bps)

    @property
    def state(self) -> np.ndarray:
        # if before bp, then trunk, otherwise pick the closest location
        state_labels = np.ones_like(self._data.Y)

        for gene_idx, model in enumerate(self._models):
            branching_time = self.branching_times[gene_idx]

            spline_trunk, spline_2, spline_3 = model
            distance_to_state_2 = np.abs(
                self._data.Y[:, gene_idx] - spline_2(self._data.t)
            )
            distance_to_state_3 = np.abs(
                self._data.Y[:, gene_idx] - spline_3(self._data.t)
            )
            closer_to_state_2 = distance_to_state_2 < distance_to_state_3

            # TODO: can probably rewrite in numpy
            for i in range(len(self._data.t)):
                # only assign to branches if we're past the learned branching time
                if self._data.t[i] >= branching_time:
                    state_labels[i, gene_idx] = 2 if closer_to_state_2[i] else 3

        return state_labels

    def predictions(self, t: ColumnVector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions_along_genes: Tuple[List[float], List[float], List[float]] = (
            [],
            [],
            [],
        )
        # state 1, 2, 3 respectively

        for gene_idx, model in enumerate(self._models):
            spline_trunk, spline_2, spline_3 = model
            for state_idx, spline in [(0, spline_trunk), (1, spline_2), (2, spline_3)]:
                predictions_along_genes[state_idx].append(spline(t).reshape(-1, 1))

        return (
            np.hstack(predictions_along_genes[0]),
            np.hstack(predictions_along_genes[1]),
            np.hstack(predictions_along_genes[2]),
        )
