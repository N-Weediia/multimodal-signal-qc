import numpy as np
import pandas as pd

from scripts.phase11_nof1_response_fingerprint import bootstrap_ci, marker_summary, signflip_p


def test_exact_signflip_all_positive_is_two_sided_minimum():
    assert signflip_p(np.array([1.0, 1.0, 1.0, 1.0, 1.0])) == 2 / 32


def test_bootstrap_is_deterministic_and_ordered():
    lo, hi = bootstrap_ci(np.array([-1.0, 0.0, 1.0]), B=1000, seed=7)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert (lo, hi) == bootstrap_ci(np.array([-1.0, 0.0, 1.0]), B=1000, seed=7)


def test_marker_summary_keeps_subject_n_and_missingness():
    d = pd.DataFrame({"subject": ["a", "b", "c", "d", "e"], "x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    out = marker_summary(d, ["x"])
    row = out.iloc[0]
    assert int(row.n_subjects) == 4
    assert int(row.positive_count) == 4
    assert float(row.direction_concordance) == 1.0


def test_magnitude_only_marker_does_not_report_direction():
    d = pd.DataFrame({"subject": ["a", "b", "c"], "spatial_channel_change": [0.1, 0.2, 0.3]})
    out = marker_summary(d, ["spatial_channel_change"])
    row = out.iloc[0]
    assert row.directionality == "magnitude_only"
    assert pd.isna(row.direction_concordance)
    assert pd.isna(row.exact_signflip_p_mean)
