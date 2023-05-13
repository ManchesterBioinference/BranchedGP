""" This module contains experiment utils for comparing the main model classes. """
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_generation import BranchedData
from .gene_expression_api import MBGP, ManyBGPs, SplineBEAM
from .label_utils import (
    count_inconsistent_assignments,
    get_incorrect_cell_label_pseudotimes,
    mean_correct_labels_per_gene,
)


@dataclass
class Result:
    """The synthetic dataset alongside the three models trained on it."""

    data: BranchedData  # the data we have attempted to fit models to. This contains the true BPs

    mbgp: MBGP
    many_bgps: ManyBGPs
    beam: SplineBEAM


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return np.sqrt(((a - b) ** 2).mean())


def convert_results_to_df(res: Sequence[Result]) -> pd.DataFrame:
    dictified_results = []

    for i, r in enumerate(res):
        dres = dict(
            sample_id=i,
            mbgp_bp_rmse=rmse(
                np.array(r.data.branching_points), r.mbgp.branching_times
            ),
            bgp_bp_rmse=rmse(
                np.array(r.data.branching_points), r.many_bgps.branching_times
            ),
            beam_bp_rmse=rmse(
                np.array(r.data.branching_points), r.beam.branching_times
            ),
            bgp_inconsistent_assignments=count_inconsistent_assignments(r.many_bgps),
            beam_inconsistent_assignments=count_inconsistent_assignments(r.beam),
            mean_mbgp_correct_labels=mean_correct_labels_per_gene(r.mbgp, data=r.data),
            mean_bgp_correct_labels=mean_correct_labels_per_gene(
                r.many_bgps, data=r.data
            ),
            mean_beam_correct_labels=mean_correct_labels_per_gene(r.beam, data=r.data),
        )
        dictified_results.append(dres)

    return pd.DataFrame(dictified_results)


def plot_incorrect_label_histograms(results: Sequence[Result]) -> None:
    all_incorrect_ptimes: Tuple[List[float], List[float], List[float]] = ([], [], [])
    # MBGP, BGP and BEAM respectively

    for i, res in enumerate(results):
        fig, axa = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for j, (model, name) in enumerate(
            [(res.mbgp, "MMBGP"), (res.many_bgps, "BGP"), (res.beam, "BEAM")]
        ):
            incorrect_ptimes = get_incorrect_cell_label_pseudotimes(model, res.data)

            axa[j].hist(incorrect_ptimes)
            axa[j].set_title(f"Sample {i}, {name} incorrect labels")

            all_incorrect_ptimes[j].extend(incorrect_ptimes)

        plt.show()

    fig, axa = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for j, name in enumerate(["MBGP", "BGP", "BEAM"]):
        ax = axa[j]
        sns.histplot(all_incorrect_ptimes[j], bins=20, ax=ax)
        ax.set_title(f"All {name} incorrect labels")

    plt.show()
