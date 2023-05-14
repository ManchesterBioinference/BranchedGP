"""
This module contains utilities for counting label assignment correctness.
"""
import numpy as np

from .data_generation import BranchedData
from .gene_expression_api import GeneExpressionModel


def get_correct_label_mask(
    model: GeneExpressionModel, data: BranchedData
) -> np.ndarray:
    """
    Get a mask on data.Y where True value indicates that the learned state
    is the true state of the cell.

    WARNING: we assume data.t spanning [0, 1] and points being equidistant.
    """
    assert model.data.t.shape == data.t.shape

    # ground truth state - take global state and adjust based on branching times
    # TODO: we can probably vectorise
    ground_truth = np.ones_like(data.Y)  # initialise and mutate below
    for gene_idx in range(data.num_genes):
        gene_bp = data.branching_points[gene_idx]

        for cell_idx, pseudotime in enumerate(data.t):
            if pseudotime >= gene_bp:
                ground_truth[cell_idx, gene_idx] = data.state[cell_idx]

    correct_labels = ground_truth == model.state
    return correct_labels


def get_consistent_assignment_mask(model: GeneExpressionModel) -> np.ndarray:
    """
    Get a mask describing which cell assignments to branches g, h are consistent across genes.

    Trunk states are ignored.

    If model.data.Y has shape (N, K), then the returned bool array has shape (N,)
    """
    num_cells = model.data.t.shape[0]
    learned_states = model.state
    trunk_mask = np.ma.masked_equal(model.state, value=1).mask
    branch_mask = ~trunk_mask

    # TODO: we can probably vectorise the following
    assignments = []
    for i in range(num_cells):
        # Here we're assuming X is uniformly spaced across [0, 1].
        cell_states_per_gene = learned_states[i]
        # ignore cells that are in trunk state
        # we're only interested in branch inconsistencies
        branch_cell_states_per_gene = cell_states_per_gene[branch_mask[i, :]]

        # consistency means that all the entries are the same (trunk or branch!)
        consistent_assignment = (not len(branch_cell_states_per_gene)) or (  # all trunk
            len(np.unique(branch_cell_states_per_gene)) == 1
        )  # all the same

        assignments.append(consistent_assignment)

    return np.array(assignments, dtype=bool)


def get_incorrect_cell_label_pseudotimes(
    model: GeneExpressionModel, data: BranchedData
) -> np.ndarray:
    correct_cell_label_mask = get_correct_label_mask(model, data=data)
    # The above is across all genes. We now find which cells have the correct state for all genes.
    correct_cell_labels_across_all_genes = np.multiply.reduce(
        correct_cell_label_mask, axis=1
    )

    incorrect_cell_label_mask = np.logical_not(correct_cell_labels_across_all_genes)
    return data.t[incorrect_cell_label_mask]


def count_inconsistent_assignments(model: GeneExpressionModel) -> int:
    consistent_assignments_mask = get_consistent_assignment_mask(model)
    return len(consistent_assignments_mask) - consistent_assignments_mask.sum()


def mean_correct_labels_per_gene(model: GeneExpressionModel, data: BranchedData) -> int:
    correct_labels_mask = get_correct_label_mask(model, data=data)
    # count correct labels per gene and then average
    return correct_labels_mask.sum(axis=0).mean()
