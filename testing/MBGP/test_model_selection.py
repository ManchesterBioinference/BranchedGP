# Test VB bound is valid objective function.
import time

import numpy as np
import pytest

from BranchedGP.MBGP.data_generation import ToyBranchedData
from BranchedGP.MBGP.FitBranchingModel import FitModel


@pytest.mark.parametrize("num_outputs", [1, 4])
def test_selection(num_outputs):
    # Control parameters
    if num_outputs == 1:
        data = ToyBranchedData(B=(0.5,), N=20)  # can also do data.plot()
    elif num_outputs == 2:
        data = ToyBranchedData(B=(0.8, 0.5), N=20)
    elif num_outputs == 4:
        data = ToyBranchedData(B=(0.2, 0.4, 0.5, 0.6), N=30)
    else:
        raise NotImplementedError
    BgridSearch = list(
        np.unique([0.1, 0.5])
    )  # make it a unique list, do not include true points
    t_start = time.time()
    fit_model_config = dict(
        M=0, maxiter=100, priorConfidence=0.65, kervar=1.0, kerlen=1.0, likvar=0.01
    )
    d = FitModel(BgridSearch, data.t, data.Y, data.state, **fit_model_config)
    print("Inference elapsed time %.1f" % (time.time() - t_start))
    # test results
    for ib, branching_pointsi in enumerate(data.branching_points):
        assert np.abs(d["Bmode"][ib] - branching_pointsi) < 0.1
