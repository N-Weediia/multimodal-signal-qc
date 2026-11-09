"""Create lightweight summaries from the completed all-subject analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_subtasks(out: Path, paired: set[str]) -> None:
    rows = [row for row in read_rows(out / "movement_subtask_summary.csv") if row.get("modality") == "fNIRS" and row.get("subject") in paired and row.get("task_name") in {"摸头", "摸膝", "摸腰"}]
    output: list[dict[str, object]] = []
    for task in ("摸头", "摸膝", "摸腰"):
        for metric, label in (("mean_response_delta", "fNIRS event response"), ("mean_left_right_asymmetry", "fNIRS L-R asymmetry")):
            changes: list[float] = []
            for subject in sorted(paired):
                baseline = next((row for row in rows if row["subject"] == subject and row["session"] == "base_line" and row["task_name"] == task), None)
                followup = next((row for row in rows if row["subject"] == subject and row["session"] == "sec_test" and row["task_name"] == task), None)
                if baseline and followup and baseline.get(metric) not in (None, "") and followup.get(metric) not in (None, ""):
                    changes.append(float(followup[metric]) - float(baseline[metric]))
            if changes:
                values = np.asarray(changes, dtype=float)
                output.append({
                    "task_name": task,
                    "metric": label,
                    "paired_subjects": int(values.size),
                    "mean_change": float(np.mean(values)),
                    "median_change": float(np.median(values)),
                    "sd_change": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "positive_count": int(np.count_nonzero(values > 0)),
                    "negative_count": int(np.count_nonzero(values < 0)),
                    "change_values": ",".join(f"{value:.6g}" for value in values),
                })
    write_rows(out / "expanded_subtask_effects.csv", output)


def summarize_states(out: Path, paired: set[str]) -> None:
    rows = read_rows(out / "fnirs_event_state_summary.csv")
    output: list[dict[str, object]] = []
    rates: list[float] = []
    for subject in sorted(paired):
        baseline = next((row for row in rows if row["subject"] == subject and row["session"] == "base_line"), None)
        followup = next((row for row in rows if row["subject"] == subject and row["session"] == "sec_test"), None)
        if not baseline or not followup:
            continue
        baseline_rate = int(baseline["state_switches"]) / max(int(baseline["events"]) - 1, 1)
        followup_rate = int(followup["state_switches"]) / max(int(followup["events"]) - 1, 1)
        change = followup_rate - baseline_rate
        rates.append(change)
        output.append({
            "subject": subject,
            "baseline_events": baseline["events"],
            "followup_events": followup["events"],
            "baseline_switch_rate": baseline_rate,
            "followup_switch_rate": followup_rate,
            "switch_rate_change": change,
            "baseline_silhouette": baseline["silhouette"],
            "followup_silhouette": followup["silhouette"],
        })
    values = np.asarray(rates, dtype=float)
    if values.size:
        output.append({
            "subject": "GROUP_SUMMARY",
            "paired_subjects": int(values.size),
            "median_switch_rate_change": float(np.median(values)),
            "positive_change_count": int(np.count_nonzero(values > 0)),
            "negative_change_count": int(np.count_nonzero(values < 0)),
        })
    write_rows(out / "expanded_state_effects.csv", output)


def summarize_quality(out: Path) -> None:
    rows = read_rows(out / "expanded_availability_inventory.csv")
    output: list[dict[str, object]] = []
    for task, modality in (("运动任务", "FNIRS"), ("运动任务", "EEG"), ("运动相关肌电", "EMG"), ("静息态", "FNIRS"), ("眼动任务", "FNIRS"), ("眼动任务", "EEG")):
        selected = [row for row in rows if row["task"] == task and row["modality"] == modality]
        scores = []
        for row in selected:
            try:
                scores.append(float(row["quality_score"]))
            except (TypeError, ValueError):
                pass
        output.append({
            "task": task,
            "modality": modality,
            "recordings": len(selected),
            "quality_ge_80": int(np.count_nonzero(np.asarray(scores) >= 80.0)) if scores else 0,
            "quality_lt_80": int(np.count_nonzero(np.asarray(scores) < 80.0)) if scores else 0,
            "missing_quality": int(len(selected) - len(scores)),
            "mean_quality_score": float(np.mean(scores)) if scores else "",
            "low_quality_subjects": ",".join(f"{row['subject']}({row['quality_score']})" for row in selected if row.get("quality_score") and float(row["quality_score"]) < 80.0),
        })
    write_rows(out / "expanded_quality_summary.csv", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/analysis/20260901_all_cases"))
    args = parser.parse_args()
    metadata = json.loads((args.out / "expanded_analysis_metadata.json").read_text(encoding="utf-8"))
    paired = set(metadata["paired_baseline_sec_test_subjects"])
    summarize_subtasks(args.out, paired)
    summarize_states(args.out, paired)
    summarize_quality(args.out)
    print(f"summaries written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
