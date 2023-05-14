"""
Tests equivalence of multi to single assign code path under the special case
of a single branching value.
"""
import gpflow
import numpy as np
import pytest

# Branching files
from BranchedGP.MBGP import VBHelperFunctions
from BranchedGP.MBGP.assigngp import AssignGP


@pytest.mark.parametrize("num_dim", [1, 2])
def test_equivalence(num_dim):
    t = np.array([0.0, 0.25, 0.5, 0.5, 0.9, 0.9])
    Y = np.array(
        [[0.0, 0, 1, -1, 2.0, -3], [0, 0, 0, 0, 1, -1]]  # split at 0.5
    ).T  # split at 0.9
    # bv_list = [0.1, 0.9]  # list of branching kernel points

    XExpanded, indices, _ = VBHelperFunctions.GetFunctionIndexListGeneral(t)

    # Solve allocation problem
    phiInitial = np.array(
        [[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
    )
    phiPrior = np.repeat(np.array([0.2, 0.4, 0.4])[None, :], t.size, axis=0)
    assert np.allclose(phiInitial.sum(1), np.ones(t.size))
    assert np.allclose(phiPrior.sum(1), np.ones(t.size))
    # branching_pt = 0.5
    for D in range(1, 3):  # Dimension
        print("*" * 10, "Dimension", D)
        if D == 1:
            Yd = Y[:, 0][:, None]
        else:
            Yd = Y
        assert Yd.shape[1] == D
        common_settings = dict(phiPrior=phiPrior, phiInitial=phiInitial, fDebug=True)
        m_single = AssignGP(t, XExpanded, Yd, indices, **common_settings, multi=False)
        gpflow.set_trainable(m_single.likelihood.variance, False)
        gpflow.set_trainable(m_single.logPhi, True)

        ll_s = m_single.maximum_log_likelihood_objective()
        mu_s, var_s = m_single.predict_f(XExpanded)

        m_multi = AssignGP(t, XExpanded, Yd, indices, **common_settings, multi=True)
        gpflow.set_trainable(m_single.likelihood.variance, False)
        gpflow.set_trainable(m_single.logPhi, True)

        ll_m = m_multi.maximum_log_likelihood_objective()
        mu_m, var_m = m_multi.predict_f(XExpanded)

        assert np.allclose(ll_s, ll_m)
        assert np.allclose(mu_s, mu_m)
        assert np.allclose(var_s, var_m)
