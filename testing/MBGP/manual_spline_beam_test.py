import matplotlib.pyplot as plt

from BranchedGP.MBGP.gene_expression_api import SplineBEAM
from BranchedGP.MBGP.plotting_helpers import plot_gene_expression_model
from BranchedGP.MBGP.sampling_helpers import get_synthetic_noisy_branched_data
from BranchedGP.MBGP.training_helpers import get_assigngp_with_target_bps

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

    for i, data in enumerate(synthetic_noisy_data):
        print(f"Processing sample {i}")

        splines = SplineBEAM(data=data, initial_bps=TRUE_BPS)
        plot_gene_expression_model(data=data, model=splines, alpha=0.3)
        plt.show()
