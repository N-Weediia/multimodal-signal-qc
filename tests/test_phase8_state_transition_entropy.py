import numpy as np

from scripts.phase8_state_transition_entropy import sequence_metrics


def test_label_invariant_state_descriptors_survive_state_swap() -> None:
    original, _ = sequence_metrics(np.array([1, 1, 2, 2, 1]))
    swapped, _ = sequence_metrics(np.array([2, 2, 1, 1, 2]))
    for key in (
        "occupancy_entropy_norm",
        "occupancy_imbalance_abs",
        "switch_rate",
        "persistence_index",
        "mean_dwell_events",
        "max_dwell_events",
        "dwell_cv",
        "transition_entropy_norm",
        "conditional_transition_entropy_norm",
        "stay_fraction",
        "switch_fraction",
    ):
        assert np.isclose(original[key], swapped[key], equal_nan=True)


def test_empty_and_singleton_sequences_are_graceful() -> None:
    empty, transition = sequence_metrics(np.array([], dtype=int))
    singleton, _ = sequence_metrics(np.array([1], dtype=int))
    assert transition.shape == (2, 2)
    assert empty["n_events"] == 0
    assert np.isnan(singleton["switch_rate"])
