from __future__ import annotations

import quality_gated_phenotype as phenotype


def test_quality_gated_outputs_drop_sec_and_exclude_eye_total_score() -> None:
    motor = [
        {"subject": "2026000001", "name": "A", "session": "base_line", "fnirs_event_proxy_delta": "1", "fnirs_event_count_used": "99", "emg_global_coactivation": "2", "eeg_frontal_alpha": "3", "bridge_event_spearman_rho": "0.5", "bridge_event_pairs": "99"},
        {"subject": "2026000001", "name": "A", "session": "sec", "fnirs_event_proxy_delta": "2", "fnirs_event_count_used": "99", "emg_global_coactivation": "3", "eeg_frontal_alpha": "4", "bridge_event_spearman_rho": "0.6", "bridge_event_pairs": "99"},
        {"subject": "2026000001", "name": "A", "session": "sec_test", "fnirs_event_proxy_delta": "3", "fnirs_event_count_used": "99", "emg_global_coactivation": "4", "eeg_frontal_alpha": "5", "bridge_event_spearman_rho": "0.7", "bridge_event_pairs": "99"},
    ]
    availability = [
        {"subject": "2026000001", "session": "base_line", "modality": "FNIRS", "available": "1", "quality_score": "80"},
        {"subject": "2026000001", "session": "base_line", "modality": "EMG", "available": "1", "quality_score": "60"},
        {"subject": "2026000001", "session": "base_line", "modality": "EEG", "available": "1", "quality_score": "40"},
    ]
    tcm = [{"subject": "2026000001", "session": "base_line", "report_score_archive_only": "20", "eye_fixation_saccades": "4", "four_crack_score": "0.7", "four_tongue_color": "淡红"}]

    rows, deltas = phenotype.build_outputs(motor, availability, tcm)

    assert [(row["subject"], row["session"]) for row in rows] == [
        ("2026000001", "base_line"), ("2026000001", "sec_test")
    ]
    assert len(deltas) == 1
    assert rows[0]["fnirs_quality_weight"] == 0.8
    assert rows[0]["emg_quality_weight"] == 0.6
    assert rows[0]["eeg_quality_weight"] == 0.4
    assert rows[0]["eye_report_present"] == 1
    assert rows[0]["four_report_present"] == 1
    missing_row = rows[1]
    assert missing_row["report_row_present"] == 0
    assert missing_row["eye_report_present"] == 0
    assert missing_row["eye_process_feature_count"] == 0
    assert "report_score_archive_only" not in rows[0]
    assert rows[0]["eye_process_feature_count"] == 1
    assert rows[0]["tcm_raw_feature_count"] == 1
    assert rows[0]["eye_process_contribution"] <= 5
    assert rows[0]["tcm_raw_contribution"] <= 5
    assert rows[0]["fnirs_feature_count"] == 1
    assert rows[0]["emg_bridge_feature_count"] == 1
