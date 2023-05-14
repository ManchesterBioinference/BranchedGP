from typing import List

import matplotlib.pyplot as plt

from BranchedGP.MBGP.assigngp import AssignGP
from BranchedGP.MBGP.data_generation import BranchedData
from BranchedGP.MBGP.experiment_utils import (
    Result,
    convert_results_to_df,
    plot_incorrect_label_histograms,
)
from BranchedGP.MBGP.gene_expression_api import MBGP, ManyBGPs, SplineBEAM
from BranchedGP.MBGP.plotting_helpers import plot_gene_expression_model
from BranchedGP.MBGP.sampling_helpers import get_synthetic_noisy_branched_data
from BranchedGP.MBGP.training_helpers import (
    AssignGPOptimiser,
    FunkyPrior,
    construct_assigngp_model,
    get_assigngp_with_target_bps,
)

if __name__ == "__main__":
    # Simple test script to check that stuff isn't going completely wrong
    NOISE = 0.1
    LENGTHSCALE = 0.5

    NUM_SAMPLES = 1
    TRUE_BPS = [i / 11 for i in range(1, 11, 1)]
    NUM_CELLS = 100

    HIGH_PRIOR_CONFIDENCE = 0.8

    dummy_model = get_assigngp_with_target_bps(
        TRUE_BPS,
        lengthscale=LENGTHSCALE,
        noise_variance=NOISE,
    )
    synthetic_noisy_data = get_synthetic_noisy_branched_data(
        dummy_model,
        num_samples=NUM_SAMPLES,
        x_pts=NUM_CELLS,
    )

    LOW_PRIOR_CONFIDENCE = 0.8
    UNINFORMATIVE_UNTIL = 0.8

    funky_prior_results: List[Result] = []

    class NullOptimiser(AssignGPOptimiser):
        def train(self, model: AssignGP) -> AssignGP:
            return model

    # ensure we use the same type of optimiser throughout
    def get_optimiser() -> AssignGPOptimiser:
        # return ElvijsAmazingOptimiser()
        return NullOptimiser()

    for i, data in enumerate(synthetic_noisy_data):
        print(f"Processing sample {i}")

        #
        # MMBGP
        #
        mmbgp = construct_assigngp_model(
            gene_expression=data,
            phi_constructor=FunkyPrior(
                data,
                informative_prior_confidence=HIGH_PRIOR_CONFIDENCE,
                uninformative_until=UNINFORMATIVE_UNTIL,
            ),
            initial_branching_points=[0.5]
            * data.num_genes,  # Don't start at the true locations
        )

        optimiser = get_optimiser()
        try:
            trained_mmbgp = optimiser.train(mmbgp)
        except Exception as ex:
            print(f"Error in training: {ex}")
            trained_mmbgp = mmbgp

        wrapped_mbgp = MBGP(model=trained_mmbgp, data=data)
        plot_gene_expression_model(data=data, model=wrapped_mbgp, alpha=0.3)

        #
        # BGPs
        #
        bgps = []

        for i in range(data.num_genes):
            print(f"Slicing for gene {i}")

            sliced_data = BranchedData(
                t=data.t,
                Y=data.Y[:, i].reshape(-1, 1),
                state=data.state,
                gene_labels=[f"{i}"],
                branching_points=data.branching_points[i : i + 1],
            )

            bgp = construct_assigngp_model(
                gene_expression=sliced_data,
                phi_constructor=FunkyPrior(
                    sliced_data,
                    informative_prior_confidence=HIGH_PRIOR_CONFIDENCE,
                    uninformative_until=UNINFORMATIVE_UNTIL,
                ),
                initial_branching_points=[0.5],  # Do not initialise to true locations
            )

            optimiser = get_optimiser()
            try:
                trained_bgp = optimiser.train(bgp)
            except Exception as ex:
                print(f"Error in training: {ex}")
                trained_bgp = bgp

            bgps.append(trained_bgp)

        wrapped_bgp = ManyBGPs(models=bgps, data=data)
        plot_gene_expression_model(data=data, model=wrapped_bgp, alpha=0.3)

        #
        # BEAM
        #
        splines = SplineBEAM(data=data, initial_bps=TRUE_BPS)
        plot_gene_expression_model(data=data, model=splines, alpha=0.3)

        result = Result(
            data=data,
            mbgp=wrapped_mbgp,
            many_bgps=wrapped_bgp,
            beam=splines,
        )
        funky_prior_results.append(result)

    #
    # Result analysis
    #

    funky_prior_df = convert_results_to_df(funky_prior_results)
    print(funky_prior_df.head(30))

    print(funky_prior_df.describe())

    plot_incorrect_label_histograms(funky_prior_results)
    plt.show()
