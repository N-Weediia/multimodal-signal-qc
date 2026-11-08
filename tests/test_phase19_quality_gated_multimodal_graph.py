import numpy as np
import pandas as pd

from scripts.phase19_quality_gated_multimodal_graph import (
    correlation,
    exact_permutation_p,
    loo_robust_z,
)


def test_exact_permutation_is_bounded_and_detects_monotone_order():
    x = np.arange(1, 5, dtype=float)
    y = np.arange(1, 5, dtype=float)
    rho, p_nominal, n = correlation(x, y)
    p_exact = exact_permutation_p(x, y)
    assert rho == 1.0
    assert n == 4
    assert 0.0 < p_exact <= 1.0
    assert p_exact <= 0.1
    assert 0.0 <= p_nominal <= 1.0


def test_correlation_rejects_constant_edge():
    rho, p_value, n = correlation(np.ones(5), np.arange(5, dtype=float))
    assert np.isnan(rho)
    assert np.isnan(p_value)
    assert n == 5


def test_loo_robust_z_does_not_use_target_value_as_reference():
    values = pd.Series([0.0, 1.0, 2.0, 3.0])
    z_target = loo_robust_z(values, 0)
    z_other = loo_robust_z(values, 3)
    assert z_target < 0
    assert z_other > 0
    assert z_target != z_other
