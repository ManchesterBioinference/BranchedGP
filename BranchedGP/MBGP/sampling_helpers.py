import logging
from typing import List, Mapping, Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline

from .assigngp import AssignGP
from .data_generation import BranchedData
from .VBHelperFunctions import GetFunctionIndexListGeneral

LOG = logging.getLogger("sampling")


def sample_prior_as_branched_data(
    model: AssignGP, num_data_points: int = 100
) -> BranchedData:
    """
    Sample from the model's prior (that is, ignore the data).

    Return as BranchedData.
    In particular, this means that function samples are clipped according
    to where the branching points lie.
    """
    N = num_data_points
    D = model.Y.shape[1]

    x_new = np.linspace(0, 1, N)
    x_expanded, _, __ = GetFunctionIndexListGeneral(x_new)  # [3*N, 2]

    samples = model.sample_prior(x_expanded, num_samples=1)  # [1, 3*N, D]
    sample = samples.squeeze()  # [3*N, D]

    # The sample above is of f, g, h over the whole of x_new.
    # We now need to produce a sample from the branching process Y.
    # Therefore we need to throw away the trunk (f) bit after the branching point
    # as well as the branches (g and h) before the branching point.
    f_id = 1
    g_id = 2

    min_bp = 1.0
    ys = []
    for d in range(D):
        # Due to the construction of x_expanded, we know that
        # if x_expanded[i, 1] == 2, then x_expanded[i+1, 1] == 3 and they correspond to
        # the same time-step described in x_new.
        b = model.BranchingPoints[d]

        before_branching_mask = x_expanded[:, 0] < b

        trunk_mask = np.logical_and(x_expanded[:, 1] == f_id, before_branching_mask)
        trunk = sample[trunk_mask, d]  # we take all of the trunk

        # Now let's construct the branches.
        # Start with indices corresponding to g (after the branching point) and add 1 to every other index
        branch_mask = np.logical_and(x_expanded[:, 1] == g_id, x_expanded[:, 0] >= b)
        branch_idx = np.where(branch_mask)[0]
        branch_idx[::2] += 1
        sampled_g_and_h = sample[branch_idx, d]

        y_sample = np.row_stack(
            tuple(arr.reshape((-1, 1)) for arr in [trunk, sampled_g_and_h] if arr.size)
        )
        ys.append(y_sample)

        if b < min_bp:
            min_bp = b

            # Get the branch mask we actually used, including the offsets
            branch_mask = np.zeros(shape=branch_mask.shape, dtype=bool)
            branch_mask[branch_idx] = 1  # convert the indices into a mask

            x_after_bp = np.where(x_expanded[:, 0] >= b)[
                0
            ]  # assumes x_expanded[:, 0] is ascending
            if len(x_after_bp):
                bp_idx = x_after_bp[0]
            else:
                bp_idx = -1

            state_mask = np.concatenate((trunk_mask[:bp_idx], branch_mask[bp_idx:]))

    y = np.column_stack(ys)

    # Now add noise
    y += np.sqrt(model.likelihood.variance.numpy()) * np.random.standard_normal(y.shape)

    return BranchedData(
        t=x_new,
        Y=y,
        state=x_expanded[state_mask, 1],
        branching_points=model.BranchingPoints,  # type: ignore  # ndarray can be consumed as a sequence
        gene_labels=None,
    )


# TODO: the following can be used in above
def convert_latent_samples_to_branched_data(
    sample: np.ndarray,
    x_expanded: np.ndarray,
    branching_points: Sequence[float],
    noise: float,
) -> BranchedData:
    """
    Convert f, g, h samples to BranchedData.

    In particular, this means that function samples are clipped according
    to where the branching points lie.
    """
    N3, D = sample.shape
    assert len(branching_points) == D, (
        f"Expected a branching point per gene. "
        f"Instead got {D} genes, but {len(branching_points)} branching points"
    )
    assert x_expanded.shape == (
        N3,
        2,
    ), f"Expected x_expanded to have shape {(N3, 2)}. Instead got {x_expanded.shape}"

    # The sample above is of f, g, h over the whole of x_new.
    # We now need to produce a sample from the branching process Y.
    # Therefore we need to throw away the trunk (f) bit after the branching point
    # as well as the branches (g and h) before the branching point.
    f_id = 1
    g_id = 2

    min_bp = 1.0
    ys = []
    for d in range(D):
        # Due to the construction of x_expanded, we know that
        # if x_expanded[i, 1] == 2, then x_expanded[i+1, 1] == 3 and they correspond to
        # the same time-step described in x_new.
        b = branching_points[d]

        before_branching_mask = x_expanded[:, 0] < b

        trunk_mask = np.logical_and(x_expanded[:, 1] == f_id, before_branching_mask)
        trunk = sample[trunk_mask, d]  # we take all of the trunk

        # Now let's construct the branches.
        # Start with indices corresponding to g (after the branching point) and add 1 to every other index
        branch_mask = np.logical_and(x_expanded[:, 1] == g_id, x_expanded[:, 0] >= b)
        branch_idx = np.where(branch_mask)[0]
        branch_idx[::2] += 1
        sampled_g_and_h = sample[branch_idx, d]

        y_sample = np.row_stack(
            tuple(arr.reshape((-1, 1)) for arr in [trunk, sampled_g_and_h] if arr.size)
        )
        ys.append(y_sample)

        if b < min_bp:
            min_bp = b

            # Get the branch mask we actually used, including the offsets
            branch_mask = np.zeros(shape=branch_mask.shape, dtype=bool)
            branch_mask[branch_idx] = 1  # convert the indices into a mask

            bp_idx = np.where(x_expanded[:, 0] >= b)[0][
                0
            ]  # assumes x_expanded[:, 0] is ascending
            state_mask = np.concatenate((trunk_mask[:bp_idx], branch_mask[bp_idx:]))

    y = np.column_stack(ys)

    # Now add noise
    y += np.sqrt(noise) * np.random.standard_normal(y.shape)

    x_new = x_expanded[0::3, 0]

    return BranchedData(
        t=x_new,
        Y=y,
        state=x_expanded[state_mask, 1],
        branching_points=branching_points,
        gene_labels=None,
    )


def filter_single_crossing(samples: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    :param samples: [S, 3*N, D] where S is the number of samples,
                    3*N is the number of points in x_expanded and D is the number of genes.
    :param t: pseudotime, shape (N,)
    :return: filtered samples [S', 3*N, D] where 0 <= S' <= S. Samples are rejected if
             the branches meet more than once.
    """
    S, N3, D = samples.shape

    samples_to_keep = []
    for s in range(S):
        genes_with_good_samples = []

        for d in range(D):
            _, g, h = samples[s, 0::3, d], samples[s, 1::3, d], samples[s, 2::3, d]
            # We now want to look at the sign of (g - h).
            # If it changes more than once, then we know the branches cross more than once.
            # However, we want to be robust to noise, so we will fit a smooth spline to g and h
            # and check for crossings there
            g_spline = UnivariateSpline(t, g, s=20)
            h_spline = UnivariateSpline(t, h, s=20)
            g_smoothed = g_spline(t)
            h_smoothed = h_spline(t)

            g_above_h = (g_smoothed - h_smoothed) > 0
            # xor with a shifted version of itself to catch changes in sign
            crossings = np.logical_xor(g_above_h[1:], g_above_h[:-1]).sum()

            if crossings > 1:
                LOG.debug(
                    f"Discarding sample {s} due to multiple crossings in gene {d}"
                )
                continue
            elif crossings == 0:
                LOG.debug(
                    f"No crossings detected in sample {s}, gene {d}. "
                    f"Are you sure the sample is from a branching process?"
                )
            else:
                genes_with_good_samples.append(s)

        if len(genes_with_good_samples) == D:
            samples_to_keep.append(s)

    return samples[samples_to_keep, :, :]


Stack = List
GenesToSamples = Mapping[int, Stack[np.ndarray]]


def filter_single_crossing_per_dimension(
    samples: np.ndarray, t: np.ndarray
) -> GenesToSamples:
    """
    :param samples: [S, 3*N, D] where S is the number of samples,
                    3*N is the number of points in x_expanded and D is the number of genes.
    :param t: pseudotime, shape (N,)
    :return: A map from gene index d (0 <= d <= D-1) to a stack of gene samples, each of shape (3*N,)
        (here, a stack is just a collection that allows us to add elements to it
        via "append" and drop them via "pop")
    """
    S, N3, D = samples.shape

    ret: Mapping[int, Stack[np.ndarray]] = {i: [] for i in range(D)}

    for s in range(S):
        for d in range(D):
            _, g, h = samples[s, 0::3, d], samples[s, 1::3, d], samples[s, 2::3, d]
            # We now want to look at the sign of (g - h).
            # If it changes more than once, then we know the branches cross more than once.
            # However, we want to be robust to noise, so we will fit a smooth spline to g and h
            # and check for crossings there
            g_spline = UnivariateSpline(t, g, s=20)
            h_spline = UnivariateSpline(t, h, s=20)
            g_smoothed = g_spline(t)
            h_smoothed = h_spline(t)

            g_above_h = (g_smoothed - h_smoothed) > 0
            # xor with a shifted version of itself to catch changes in sign
            crossings = np.logical_xor(g_above_h[1:], g_above_h[:-1]).sum()

            if crossings > 1:
                LOG.debug(
                    f"Discarding sample {s} due to multiple crossings in gene {d}"
                )
                continue
            elif crossings == 0:
                LOG.debug(
                    f"No crossings detected in sample {s}, gene {d}. "
                    f"Are you sure the sample is from a branching process?"
                )
            else:
                ret[d].append(samples[s, :, d])

    return ret


def patch_dimension_samples(genes_to_samples: GenesToSamples) -> np.ndarray:
    """
    Given samples per gene, patch them together to create a sample pertaining to all genes.

    WARNING: this function mutates genes_to_samples by discarding the used samples.

    Return shape: [S, 3*N, D] where S is the number of samples, N is the length of pseudotime, and
        D is the number of genes.
    """
    D = len(genes_to_samples.keys())

    at_least_one_sample_per_gene = all(
        len(gene_samples) for gene_samples in genes_to_samples.values()
    )

    patched_samples: List[
        np.ndarray
    ] = []  # each element in the list is a valid sample from the full model
    while at_least_one_sample_per_gene:
        gene_samples = [genes_to_samples[i].pop() for i in range(D)]
        patched_samples.append(np.array(gene_samples).T)

        at_least_one_sample_per_gene = all(
            len(gene_samples) for gene_samples in genes_to_samples.values()
        )

    return np.array(patched_samples)


def get_synthetic_noisy_branched_data(
    model: AssignGP,
    num_samples: int,
    x_pts: int = 50,
    max_draws: int = 1_000,
) -> Sequence[BranchedData]:
    x_new = np.linspace(0, 1, x_pts)
    x_expanded, indices, _ = GetFunctionIndexListGeneral(x_new)

    num_samples_in_a_batch = 10

    branched_data_list: List[BranchedData] = []

    samples_drawn = 0
    continue_drawing_samples = (
        len(branched_data_list) < num_samples and samples_drawn < max_draws
    )

    num_genes = model.Y.shape[1]
    genes_to_samples: GenesToSamples = {i: [] for i in range(num_genes)}

    while continue_drawing_samples:
        samples = model.sample_prior(x_expanded, num_samples=num_samples_in_a_batch)
        samples_drawn += num_samples_in_a_batch

        # Get new samples
        new_genes_to_samples = filter_single_crossing_per_dimension(samples, x_new)
        for g, new_samples in new_genes_to_samples.items():
            genes_to_samples[g] += new_samples

        # Patch per-dimension samples into a sample from the model
        # We can do this since all draws are from the same model

        valid_samples = patch_dimension_samples(genes_to_samples)
        print(
            f"{samples_drawn} samples drawn, {len(branched_data_list)} valid samples constructed"
        )

        for sample in valid_samples:
            branched_data = convert_latent_samples_to_branched_data(
                sample,
                x_expanded=x_expanded,
                branching_points=model.BranchingPoints,  # type: ignore  # ndarray can be consumed as a sequence
                noise=model.likelihood.variance.numpy(),
            )
            branched_data_list.append(branched_data)

        continue_drawing_samples = (
            len(branched_data_list) < num_samples and samples_drawn < max_draws
        )

    return branched_data_list[:num_samples]
