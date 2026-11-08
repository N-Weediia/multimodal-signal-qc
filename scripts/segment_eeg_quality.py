#!/usr/bin/env python
"""Event-aware EEG quality screening for movement-task recordings.

The output is a quality mask and a descriptive comparison of pre-action,
action, and post-action windows.  It never overwrites source BDF files and it
does not interpolate or "repair" a segment silently.  A segment can be kept
for downstream analysis only when its provenance and quality columns are
carried along.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Running a file under scripts/ does not automatically expose the project root
# for imports of the existing readers and event-label helpers.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import explore_202608_cases as events_module
import main as eeg_main


ACTION_TASKS = {"摸头", "摸膝", "摸腰", "握拳"}
PRIMARY_ACTION_TASKS = {"摸头", "摸膝", "摸腰"}
EEG_BAND_HZ = (0.5, 45.0)
LINE_HZ = 50.0
EPOCH_SECONDS = 2.0
ARTIFACT_PTP_UV = 200.0
FLAT_PTP_UV = 2.0


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def replace_subject_rows(path: Path, rows: list[dict], subjects: set[str], append: bool) -> list[dict]:
    """Replace only this run's subjects when completing a long paired cohort."""
    if not append or not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        prior = list(csv.DictReader(handle))
    retained = [row for row in prior if row.get("subject") not in subjects]
    return retained + rows


def finite_mean(values: Iterable[float]) -> float | str:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return round(float(finite.mean()), 6) if finite.size else ""


def _task_directories(root: Path, token: str) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_dir() and token in path.name]


def movement_recording(data_root: Path, subject: str, session: str) -> Path | None:
    eeg_root = data_root / subject / session / "EEG"
    folders = _task_directories(eeg_root, "运动")
    candidates = [
        path
        for folder in folders
        for path in folder.glob("*.bdf")
        if path.name.lower() != "evt.bdf"
    ]
    # Exports such as data.1.bdf can be a tiny aborted recording beside the
    # formal data.bdf.  Prefer the largest recording while retaining the path
    # in the output for auditability.
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def primary_action_count(events: np.ndarray) -> int:
    labels = events_module.infer_movement_event_labels(events, "")
    return sum(label["task_name"] in PRIMARY_ACTION_TASKS for label in labels)


def get_events(recording: Path) -> tuple[np.ndarray | None, str, str]:
    eeg_event_path = recording.parent / "evt.bdf"
    eeg_events = events_module.read_bdf_annotation(eeg_event_path)
    eeg_primary_count = primary_action_count(eeg_events.times) if eeg_events is not None else 0
    if eeg_events is not None and eeg_primary_count >= 15:
        return eeg_events.times, "EEG evt.bdf description=1", "observed"
    # A missing or partial EEG companion event file is common in reorganized
    # exports. A lone/partial EEG trigger must not prevent a complete fNIRS
    # marker series from serving as the documented timing fallback.
    session_root = recording.parents[2]
    fnirs_candidates = sorted(
        path
        for folder in _task_directories(session_root / "FNIRS", "运动")
        for path in folder.glob("*.nirs")
    )
    if fnirs_candidates:
        fnirs_events = events_module.read_fnirs_events(fnirs_candidates[0])
        fnirs_primary_count = primary_action_count(fnirs_events.times) if fnirs_events is not None else 0
        if fnirs_events is not None and fnirs_primary_count >= 15:
            if eeg_events is not None:
                source = f"fNIRS complete marker fallback after incomplete EEG evt ({eeg_primary_count}/15 primary actions)"
            else:
                source = "fNIRS complete marker fallback (inferred EEG timing)"
            return fnirs_events.times, source, "inferred_from_fnirs"
    if eeg_events is not None and eeg_events.times.size:
        return eeg_events.times, f"EEG evt.bdf incomplete ({eeg_primary_count}/15 primary actions)", "observed_incomplete"
    return None, "none", "missing"


def filter_eeg(raw: np.ndarray, sample_rate: float) -> np.ndarray:
    robust_ref = raw - np.median(raw, axis=0, keepdims=True)
    detrended = signal.detrend(robust_ref, axis=-1, type="linear")
    notch_b, notch_a = signal.iirnotch(LINE_HZ, Q=35, fs=sample_rate)
    without_line = signal.filtfilt(notch_b, notch_a, detrended, axis=-1)
    upper = min(EEG_BAND_HZ[1], sample_rate * 0.45)
    sos = signal.butter(4, (EEG_BAND_HZ[0], upper), btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, without_line, axis=-1).astype(np.float32)


def quality_metrics(segment: np.ndarray, sample_rate: float) -> dict[str, float]:
    epoch_samples = max(1, int(round(sample_rate * EPOCH_SECONDS)))
    count = segment.shape[1] // epoch_samples
    if count < 1:
        return {
            "epoch_count": 0,
            "bad_epoch_fraction": float("nan"),
            "broad_bad_epoch_fraction": float("nan"),
            "flat_epoch_fraction": float("nan"),
            "bad_channel_fraction": float("nan"),
            "median_ptp_uv": float("nan"),
            "median_robust_std_uv": float("nan"),
            "median_abs_derivative_uv": float("nan"),
            "quality_status": "TOO_SHORT",
        }
    usable = segment[:, :count * epoch_samples].reshape(segment.shape[0], count, epoch_samples)
    ptp = np.ptp(usable, axis=2)
    bad_epochs = ptp > ARTIFACT_PTP_UV
    flat_epochs = ptp < FLAT_PTP_UV
    bad_epoch_fraction = float(np.mean(np.any(bad_epochs, axis=0)))
    # Distinguish a local electrode failure from a cap-wide movement artifact.
    # A 20% channel quorum is deliberately conservative for a 59-channel cap.
    broad_bad_epoch_fraction = float(np.mean(np.mean(bad_epochs, axis=0) >= 0.20))
    bad_channel_fraction = float(np.mean(np.mean(bad_epochs, axis=1) >= 0.20))
    flat_epoch_fraction = float(np.mean(np.any(flat_epochs, axis=0)))
    robust_std = np.median(np.abs(segment - np.median(segment, axis=1, keepdims=True)), axis=1) * 1.4826
    # A cap shift tends to increase the common abrupt derivative across many electrodes.
    diff = np.diff(segment, axis=1)
    median_abs_derivative = float(np.median(np.abs(diff))) if diff.size else float("nan")
    if broad_bad_epoch_fraction <= 0.10 and bad_channel_fraction <= 0.10:
        status = "GOOD"
    elif broad_bad_epoch_fraction <= 0.35 and bad_channel_fraction <= 0.35:
        status = "CAUTION"
    else:
        status = "POOR"
    return {
        "epoch_count": int(count),
        "bad_epoch_fraction": bad_epoch_fraction,
        "broad_bad_epoch_fraction": broad_bad_epoch_fraction,
        "flat_epoch_fraction": flat_epoch_fraction,
        "bad_channel_fraction": bad_channel_fraction,
        "median_ptp_uv": float(np.median(ptp)),
        "median_robust_std_uv": float(np.median(robust_std)),
        "median_abs_derivative_uv": median_abs_derivative,
        "quality_status": status,
    }


def segment_rows(subject: str, session: str, recording: Path, events: np.ndarray, event_source: str, alignment_status: str,
                 filtered: np.ndarray, sample_rate: float, visit_protocol_complete: bool, visit_status: str) -> list[dict]:
    labels = events_module.infer_movement_event_labels(events, subject)
    rows: list[dict] = []
    duration = filtered.shape[1] / sample_rate
    for label in labels:
        task = str(label["task_name"])
        if task not in ACTION_TASKS:
            continue
        onset = float(label["timestamp_seconds"])
        block = int(label["block_index"])
        windows = (
            ("pre_action", onset - 5.0, onset),
            ("execution", onset, onset + 5.0),
            ("post_action_rest", onset + 5.0, onset + 20.0),
        )
        for role, start, end in windows:
            clipped_start = max(0.0, start)
            clipped_end = min(duration, end)
            if clipped_end <= clipped_start:
                continue
            left = int(round(clipped_start * sample_rate))
            right = int(round(clipped_end * sample_rate))
            metrics = quality_metrics(filtered[:, left:right], sample_rate)
            rows.append({
                "subject": subject,
                "session": session,
                "recording": str(recording),
                "task": task,
                "block": block,
                "segment_role": role,
                "marker_index": int(label["marker_index"]),
                "onset_seconds": round(onset, 4),
                "start_seconds": round(clipped_start, 4),
                "end_seconds": round(clipped_end, 4),
                "duration_seconds": round(clipped_end - clipped_start, 4),
                "event_source": event_source,
                "event_alignment": alignment_status,
                "recording_duration_seconds": round(duration, 4),
                "primary_action_marker_count": primary_action_count(events),
                "visit_protocol_complete": visit_protocol_complete,
                "visit_status": visit_status,
                **metrics,
            })
    return rows


def plot_results(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    tasks = ["摸头", "摸膝", "摸腰", "握拳"]
    roles = ["pre_action", "execution", "post_action_rest"]
    recordings = sorted({(r["subject"], r["session"]) for r in rows})
    fig, axes = plt.subplots(len(recordings), 2, figsize=(12, 4 * len(recordings)), squeeze=False, constrained_layout=True)
    for row_index, (subject, session) in enumerate(recordings):
        subset = [r for r in rows if r["subject"] == subject and r["session"] == session]
        ax = axes[row_index, 0]
        xlabels = []
        values = []
        colors = []
        for r in subset:
            xlabels.append(f"{r['task']}{r['block']}\n{r['segment_role'].replace('_', ' ')}")
            value = float(r["broad_bad_epoch_fraction"])
            values.append(value if np.isfinite(value) else np.nan)
            colors.append({"GOOD": "#2a9d8f", "CAUTION": "#e9c46a", "POOR": "#e76f51", "TOO_SHORT": "#888888"}.get(r["quality_status"], "#888888"))
        ax.bar(np.arange(len(values)), values, color=colors, edgecolor="#333333", linewidth=0.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("broad-artifact 2 s epoch fraction")
        ax.set_title(f"{subject} {session}  segmented EEG quality")
        ax.set_xticks(np.arange(len(values)), xlabels, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.2)
        ax.axhline(0.10, color="#2a9d8f", lw=0.8, ls="--")
        ax.axhline(0.35, color="#e76f51", lw=0.8, ls="--")

        ax = axes[row_index, 1]
        matrix = np.full((len(tasks), len(roles)), np.nan)
        for r in subset:
            if r["task"] in tasks and r["segment_role"] in roles:
                matrix[tasks.index(r["task"]), roles.index(r["segment_role"])] = float(r["bad_channel_fraction"])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(np.arange(len(roles)), ["pre", "execution", "post-rest"])
        ax.set_yticks(np.arange(len(tasks)), tasks)
        ax.set_title("bad-channel fraction")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--subjects", nargs="+", default=["2026081403", "2026081501"])
    parser.add_argument("--sessions", nargs="+", default=["base_line", "sec_test"])
    parser.add_argument("--output", type=Path, default=Path("outputs/segment_eeg_quality_20260902"))
    parser.add_argument("--append", action="store_true", help="replace the supplied subjects within an existing output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    subject_summary: list[dict] = []
    for subject in args.subjects:
        for session in args.sessions:
            recording = movement_recording(args.data_root, subject, session)
            if recording is None:
                subject_summary.append({"subject": subject, "session": session, "status": "missing_movement_eeg"})
                continue
            event_times, source, alignment = get_events(recording)
            if event_times is None:
                subject_summary.append({"subject": subject, "session": session, "recording": str(recording), "status": "missing_events"})
                continue
            header, physical, _ = eeg_main.read_bdf(recording, None)
            configured, _ = eeg_main.eeg_configuration_for_recording(recording)
            lookup = {eeg_main.normalize_label(label): i for i, label in enumerate(header.labels)}
            indices = [lookup[eeg_main.normalize_label(label)] for label in configured if eeg_main.normalize_label(label) in lookup]
            filtered = filter_eeg(physical[indices].astype(np.float32), header.sample_rate)
            labels = events_module.infer_movement_event_labels(event_times, subject)
            main_action_onsets = [
                float(label["timestamp_seconds"])
                for label in labels
                if label["task_name"] in PRIMARY_ACTION_TASKS
            ]
            duration = filtered.shape[1] / header.sample_rate
            marker_complete = len(main_action_onsets) == 15
            recording_complete = bool(main_action_onsets) and duration >= max(main_action_onsets) + 5.0
            visit_protocol_complete = marker_complete and recording_complete
            if not marker_complete:
                visit_status = "incomplete_primary_marker_series"
            elif not recording_complete:
                visit_status = "incomplete_eeg_recording_for_primary_protocol"
            elif alignment == "observed":
                visit_status = "complete_observed_markers"
            else:
                visit_status = "complete_inferred_from_fnirs"
            rows = segment_rows(
                subject, session, recording, event_times, source, alignment,
                filtered, header.sample_rate, visit_protocol_complete, visit_status,
            )
            all_rows.extend(rows)
            for task in sorted({r["task"] for r in rows}):
                execution = [r for r in rows if r["task"] == task and r["segment_role"] == "execution"]
                rest = [r for r in rows if r["task"] == task and r["segment_role"] == "post_action_rest"]
                subject_summary.append({
                    "subject": subject,
                    "session": session,
                    "recording": str(recording),
                    "task": task,
                    "event_source": source,
                    "event_alignment": alignment,
                    "recording_duration_seconds": round(duration, 4),
                    "primary_action_marker_count": len(main_action_onsets),
                    "visit_protocol_complete": visit_protocol_complete,
                    "visit_status": visit_status,
                    "blocks": len(execution),
                    "execution_broad_bad_epoch_fraction_mean": finite_mean(r["broad_bad_epoch_fraction"] for r in execution),
                    "post_rest_broad_bad_epoch_fraction_mean": finite_mean(r["broad_bad_epoch_fraction"] for r in rest),
                    "execution_bad_channel_fraction_mean": finite_mean(r["bad_channel_fraction"] for r in execution),
                    "post_rest_bad_channel_fraction_mean": finite_mean(r["bad_channel_fraction"] for r in rest),
                    "status": "ok",
                })
    all_rows = replace_subject_rows(args.output / "segment_quality.csv", all_rows, set(args.subjects), args.append)
    subject_summary = replace_subject_rows(args.output / "task_segment_summary.csv", subject_summary, set(args.subjects), args.append)
    write_csv(args.output / "segment_quality.csv", all_rows)
    write_csv(args.output / "task_segment_summary.csv", subject_summary)
    plot_results(all_rows, args.output / "segment_quality_overview.png")
    metadata = {
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(args.data_root.resolve()),
        "subjects": sorted({str(row["subject"]) for row in all_rows}),
        "sessions": args.sessions,
        "window_definition": {
            "pre_action": "marker-5s to marker",
            "execution": "marker to marker+5s",
            "post_action_rest": "marker+5s to marker+20s",
            "quality_epoch_seconds": EPOCH_SECONDS,
        },
        "caution": "when EEG evt.bdf is absent or incomplete for the 15 primary action blocks, a complete fNIRS marker series is used as inferred EEG timing and remains labeled by source; a truncated EEG recording remains an incomplete visit.",
        "rows": len(all_rows),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    visits = {
        (str(row["subject"]), str(row["session"])): row
        for row in subject_summary
        if row.get("status") == "ok"
    }
    status_counts = Counter(str(row.get("visit_status", row.get("status", "unknown"))) for row in visits.values())
    alignment_counts = Counter(str(row.get("event_alignment", "missing")) for row in visits.values())
    cohort_subjects = sorted({subject for subject, _ in visits})
    incomplete = [
        f"{subject}/{session}: {row.get('visit_status', row.get('status', 'unknown'))}"
        for (subject, session), row in sorted(visits.items())
        if str(row.get("visit_protocol_complete", "")).lower() != "true"
    ]
    report = [
        "# 分段 EEG 质控初步报告",
        "",
        "本报告把运动任务按自动 marker 划为每个 block 的 `pre_action`（前 5 s）、`execution`（动作开始后 5 s）和 `post_action_rest`（随后 15 s）。",
        "`bad_epoch_fraction` 表示任意导联异常的 2 s 窗比例；`broad_bad_epoch_fraction` 要求至少 20% 导联同时异常，更接近脑电帽整体位移。",
        "",
        "## 解释规则",
        "",
        "- `GOOD`/`CAUTION`/`POOR` 主要依据广泛伪迹比例和坏导联比例，不因单个坏导联直接删除全段。",
        "- 局部坏导联：后续可在保留段内插值或降权；脑电帽整体异常：整段剔除或单独建模。",
        "- EEG marker 缺失或不足以覆盖 15 个主动作 block 时，按当前分析约定优先用完整 fNIRS marker 反推，并保留 `inferred_from_fnirs` 来源标签；它与 EEG 原始 marker 的证据等级不同。",
        "",
        "## Paired7 cohort summary",
        "",
        f"- Requested cohort: {', '.join(cohort_subjects)} (n={len(cohort_subjects)}); visit-status counts: {dict(status_counts)}.",
        f"- Marker alignment across visits: {dict(alignment_counts)}. `握拳` is retained only as QC evidence where present; downstream Phase 17 metrics are restricted to 摸头/摸膝/摸腰.",
        "- `2026081502/base_line` has a complete fNIRS marker series but a truncated movement EEG recording. It is explicitly marked `incomplete_eeg_recording_for_primary_protocol`; its earliest windows remain QC evidence only and must not constitute a valid Phase 17 visit.",
        f"- Incomplete visits in this output: {'; '.join(incomplete) if incomplete else 'none'}.",
        "",
        "## 输出",
        "",
        "- `segment_quality.csv`：每个 block、每个时间段的完整质量证据。",
        "- `task_segment_summary.csv`：按受试者、会话和小任务汇总。",
        "- `segment_quality_overview.png`：广泛伪迹柱状图和坏导联比例热图。",
        "",
        "Phase 17 may use only visits marked `visit_protocol_complete=true`, and must continue to report observed versus inferred marker provenance.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(all_rows), "summary_rows": len(subject_summary), "append": args.append}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
