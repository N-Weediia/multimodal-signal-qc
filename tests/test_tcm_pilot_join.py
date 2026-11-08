from __future__ import annotations

import csv
from pathlib import Path

import integrate_tcm_reports


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pilot_join_excludes_scores_and_canonicalizes_sec(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    output = tmp_path / "tcm"
    _write(analysis / "motor_multimodal_features.csv", [
        {"subject": "2026000001", "session": "base_line", "motor": "1"},
        {"subject": "2026000001", "session": "sec", "motor": "2"},
        {"subject": "2026000001", "session": "sec_test", "motor": "3"},
        {"subject": "2026000002", "session": "base_line", "motor": "4"},
    ])
    _write(output / "eye_report_pilot_features.csv", [{
        "subject": "2026000001", "session": "base_line", "report_score_archive_only": "20", "prosaccade_latency_ms": "400", "metric_use": "quantitative_process_metrics_only"
    }])
    _write(output / "four_diagnosis_pilot_features.csv", [{
        "subject": "2026000001", "session": "base_line", "crack_score": "0.7", "tongue_color": "淡红"
    }])

    result_path = integrate_tcm_reports.build_pilot_join(analysis, output)

    assert result_path == output / "tcm_multimodal_pilot.csv"
    with result_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["subject"], row["session"]) for row in rows} == {
        ("2026000001", "base_line"), ("2026000001", "sec_test"), ("2026000002", "base_line")
    }
    baseline = next(row for row in rows if row["subject"] == "2026000001" and row["session"] == "base_line")
    assert baseline["eye_prosaccade_latency_ms"] == "400.0"
    assert baseline["four_crack_score"] == "0.7"
    assert all("report_score_archive_only" not in key for key in baseline)
    assert baseline["eye_total_score_model_feature"] == "no"
    missing = next(row for row in rows if row["subject"] == "2026000002")
    assert missing.get("eye_prosaccade_latency_ms", "") == ""
    assert "不推断证型" in (output / "tcm_multimodal_pilot_method.md").read_text(encoding="utf-8")
