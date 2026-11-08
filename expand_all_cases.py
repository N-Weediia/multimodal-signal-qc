"""Expand the movement analysis to every outer-ID subject in the raw disk.

Identity is taken from ``编号姓名数据核对表.xlsx`` and never from filenames.
The existing exploratory feature functions are reused, while this wrapper
adds recursive path handling, availability/QC inventory, and all-subject
paired summaries.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from openpyxl import load_workbook

import explore_202608_cases as base
from emg_qc import (
    EXPECTED_SENSOR_IDS,
    MAPPING_PROFILE_SHIFTED_2026080901,
    MAPPING_PROFILE_STANDARD,
    SENSOR_MUSCLES,
    SENSOR_TO_MUSCLE_INDEX,
    SHIFTED_SENSOR_IDS,
)


LOCAL_REPRO_ROOT = Path(__file__).resolve().parent / "data_repro_20260903"
RAW_ROOT = LOCAL_REPRO_ROOT if LOCAL_REPRO_ROOT.is_dir() else (
    Path(r"J:\2026二康") if Path(r"J:\2026二康").is_dir() else Path(r"J:\2026二康\数据")
)
IDENTITY_XLSX = RAW_ROOT / "编号姓名数据核对表.xlsx"
QC_INVENTORY = Path("outputs/qc/20260830/recording_inventory.csv")
DEFAULT_OUT = Path("outputs/analysis/20260901_all_cases")
KNOWN_SESSIONS = {"base_line", "sec", "sec_test", "thi_test"}
TASKS = {"运动": "运动任务", "静息态": "静息态", "眼动": "眼动任务"}


def set_data_root(data_root: Path) -> None:
    """Set the raw-data root while retaining the historical default."""
    global RAW_ROOT, IDENTITY_XLSX
    RAW_ROOT = data_root.expanduser().resolve()
    IDENTITY_XLSX = RAW_ROOT / "编号姓名数据核对表.xlsx"


def load_identity(path: Path) -> dict[str, str]:
    workbook = load_workbook(io.BytesIO(path.read_bytes()), read_only=True, data_only=True)
    try:
        sheet = workbook[workbook.sheetnames[0]]
        identity: dict[str, str] = {}
        for row in sheet.iter_rows(min_row=4, values_only=True):
            if row[0] is None or row[1] is None:
                continue
            subject = str(row[0]).strip()
            name = str(row[1]).strip()
            if subject and name:
                identity[subject] = name
        return identity
    finally:
        workbook.close()


def discover_sessions(identity: dict[str, str]) -> dict[str, list[str]]:
    sessions: dict[str, list[str]] = {}
    for subject in identity:
        root = RAW_ROOT / subject
        values = []
        if root.exists():
            values = sorted(item.name for item in root.iterdir() if item.is_dir() and item.name in KNOWN_SESSIONS)
        sessions[subject] = values
    return sessions


def canonicalize_sessions(sessions: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Drop duplicate ``sec`` follow-ups when canonical ``sec_test`` exists."""
    canonical: dict[str, list[str]] = {}
    excluded: list[dict[str, str]] = []
    for subject, values in sessions.items():
        values = list(values)
        if "sec_test" in values and "sec" in values:
            values = [value for value in values if value != "sec"]
            excluded.append({"subject": subject, "session": "sec", "canonical_session": "sec_test", "reason": "duplicate follow-up; sec_test retained"})
        canonical[subject] = values
    return canonical, excluded


def _largest(paths: Iterable[Path]) -> Path | None:
    values = [path for path in paths if path.is_file()]
    if not values:
        return None
    return max(values, key=lambda path: (path.stat().st_size, str(path)))


def robust_find_recording(subject: str, session: str, modality: str, task: str) -> Path | None:
    """Find the largest valid recording under the outer-ID/session folder."""
    root = RAW_ROOT / subject / session
    if not root.exists():
        return None
    if modality == "EEG":
        modality_root = root / "EEG"
        if not modality_root.exists():
            return None
        directories = [item for item in modality_root.rglob("*") if item.is_dir() and task in item.name]
        candidates = [path for directory in directories for path in directory.glob("*.bdf") if path.name.lower() != "evt.bdf"]
        return _largest(candidates)
    if modality == "FNIRS":
        # Restrict to the FNIRS tree.  A few exports contain misfiled .nirs
        # files under EEG; those are not silently reclassified as fNIRS.
        modality_root = root / "FNIRS"
        if not modality_root.exists():
            return None
        directories = [item for item in modality_root.rglob("*") if item.is_dir() and task in item.name]
        candidates = [path for directory in directories for path in directory.glob("*.nirs")]
        return _largest(candidates)
    if modality == "EMG":
        modality_root = root / "EMG"
        if not modality_root.exists():
            return None
        return _largest(modality_root.rglob("*.csv"))
    raise ValueError(modality)


def _load_qc_inventory(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    values: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = str(row.get("source_relative_path", "")).replace("/", "\\").lower()
            if source:
                values[source] = row
    return values


def _relative(path: Path) -> str:
    return str(path.relative_to(RAW_ROOT)).replace("/", "\\")


def build_availability(identity: dict[str, str], sessions: dict[str, list[str]], qc: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject, name in identity.items():
        for session in sessions.get(subject, []):
            for modality, task_name in (("FNIRS", "运动任务"), ("EEG", "运动任务"), ("EMG", "运动相关肌电"), ("FNIRS", "静息态"), ("FNIRS", "眼动任务"), ("EEG", "眼动任务")):
                task = "运动" if task_name in {"运动任务", "运动相关肌电"} else ("静息态" if task_name == "静息态" else "眼动")
                path = robust_find_recording(subject, session, modality, task) if not (modality == "EMG") else robust_find_recording(subject, session, modality, "")
                relative = _relative(path) if path else ""
                qc_row = qc.get(relative.lower(), {})
                score_text = qc_row.get("mean_quality_score", "")
                try:
                    score = float(score_text) if score_text else float("nan")
                except ValueError:
                    score = float("nan")
                status = qc_row.get("session_status", "")
                rows.append({
                    "subject": subject,
                    "name": name,
                    "session": session,
                    "task": task_name,
                    "modality": modality,
                    "available": bool(path),
                    "path": relative,
                    "quality_score": score if np.isfinite(score) else "",
                    "quality_status": status,
                    "known_ir_interference": qc_row.get("known_ir_interference", ""),
                    "eye_tracker_occlusion_status": qc_row.get("eye_tracker_occlusion_status", ""),
                    "analysis_branch": qc_row.get("analysis_branch", ""),
                    "quality_degradation": qc_row.get("quality_degradation", ""),
                    "primary_role": "主分析" if modality == "FNIRS" and task_name == "运动任务" else ("辅助分析" if modality == "EMG" and task_name == "运动相关肌电" else "低权重/机制探索"),
                    "qc_note": qc_row.get("message", ""),
                })
    return rows


def _to_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    base._write_rows(path, rows)


def write_paired_summaries(out_root: Path, identity: dict[str, str], sessions: dict[str, list[str]]) -> None:
    source = out_root / "motor_feature_comparison.csv"
    if not source.exists():
        return
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_subject: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_subject.setdefault(str(row.get("subject", "")), {})[str(row.get("session", ""))] = row
    selected = [
        "fnirs_event_proxy_delta",
        "fnirs_event_left_right_asymmetry",
        "fnirs_network_mean_abs_fisher_z",
        "rest_fnirs_network_mean_abs_fisher_z",
        "eye_fnirs_proxy_global_abs_mean",
        "eye_fnirs_analysis_sample_coverage",
        "eye_fnirs_mean_quality_score",
        "emg_global_coactivation",
        "eeg_theta_envcorr_left_sensorimotor_right_sensorimotor",
    ]
    delta_rows: list[dict[str, object]] = []
    for subject, sessions_by_name in by_subject.items():
        base_row = sessions_by_name.get("base_line")
        follow_row = sessions_by_name.get("sec_test")
        if not base_row or not follow_row:
            continue
        for feature in selected:
            baseline = _to_float(base_row.get(feature))
            followup = _to_float(follow_row.get(feature))
            if baseline is None or followup is None:
                continue
            delta_rows.append({
                "subject": subject,
                "name": identity.get(subject, base_row.get("name", "")),
                "feature": feature,
                "baseline": baseline,
                "followup": followup,
                "absolute_change": followup - baseline,
                "percent_change": (followup - baseline) / max(abs(baseline), 1e-6) * 100.0,
                "baseline_session": "base_line",
                "followup_session": "sec_test",
            })
    _write_rows(out_root / "expanded_case_delta_summary.csv", delta_rows)
    effects: list[dict[str, object]] = []
    for feature in selected:
        values = [float(row["absolute_change"]) for row in delta_rows if row["feature"] == feature]
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        effects.append({
            "feature": feature,
            "paired_subjects": int(array.size),
            "mean_change": float(np.mean(array)),
            "median_change": float(np.median(array)),
            "sd_change": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
            "iqr_change": float(np.percentile(array, 75) - np.percentile(array, 25)) if array.size > 1 else 0.0,
            "positive_count": int(np.count_nonzero(array > 0)),
            "negative_count": int(np.count_nonzero(array < 0)),
            "zero_count": int(np.count_nonzero(array == 0)),
            "interpretation": "descriptive paired effect; subjects are independent units",
        })
    _write_rows(out_root / "expanded_group_effects.csv", effects)

    baseline_rows: list[dict[str, object]] = []
    for subject, sessions_by_name in by_subject.items():
        row = sessions_by_name.get("base_line")
        if not row:
            continue
        output: dict[str, object] = {"subject": subject, "name": identity.get(subject, row.get("name", ""))}
        for key in selected:
            value = _to_float(row.get(key))
            if value is not None:
                output[key] = value
        baseline_rows.append(output)
    _write_rows(out_root / "expanded_baseline_features.csv", baseline_rows)


def write_expanded_heatmap(out_root: Path) -> None:
    import matplotlib.pyplot as plt

    path = out_root / "expanded_case_delta_summary.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    subjects = sorted({str(row["subject"]) for row in rows})
    features = [
        ("fnirs_event_proxy_delta", "任务 fNIRS 响应"),
        ("fnirs_event_left_right_asymmetry", "fNIRS 左右差异"),
        ("fnirs_network_mean_abs_fisher_z", "任务网络耦合"),
        ("rest_fnirs_network_mean_abs_fisher_z", "静息网络耦合"),
        ("eye_fnirs_proxy_global_abs_mean", "眼动 fNIRS proxy"),
        ("eye_fnirs_analysis_sample_coverage", "眼动 fNIRS 有效覆盖"),
        ("emg_global_coactivation", "EMG 协同"),
    ]
    matrix = np.full((len(subjects), len(features)), np.nan)
    for i, subject in enumerate(subjects):
        for j, (feature, _) in enumerate(features):
            match = next((row for row in rows if row["subject"] == subject and row["feature"] == feature), None)
            if match:
                matrix[i, j] = float(match["percent_change"])
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.55 * len(subjects) + 2.0)), dpi=160)
    finite = np.abs(matrix[np.isfinite(matrix)])
    vmax = max(50.0, float(np.percentile(finite, 90))) if finite.size else 50.0
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(subjects)), subjects)
    ax.set_xticks(np.arange(len(features)), [label for _, label in features], rotation=25, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("全部配对受试者：基线到二次运动记录的变化")
    ax.set_xlabel("受试者内描述性百分比变化")
    fig.colorbar(image, ax=ax, label="变化 (%)")
    fig.tight_layout()
    fig.savefig(out_root / "expanded_case_delta_heatmap.png")
    plt.close(fig)


def run(
    out_root: Path,
    data_root: Path | None = None,
    qc_inventory: Path | None = None,
) -> None:
    if data_root is not None:
        set_data_root(data_root)
    identity = load_identity(IDENTITY_XLSX)
    discovered_sessions = discover_sessions(identity)
    sessions, excluded_sessions = canonicalize_sessions(discovered_sessions)
    qc = _load_qc_inventory(qc_inventory or QC_INVENTORY)
    availability = build_availability(identity, sessions, qc)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_rows(out_root / "expanded_availability_inventory.csv", availability)
    (out_root / "subject_identity_from_xlsx.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")

    # Reuse the tested feature pipeline, replacing only identity/session/path
    # discovery.  All derived labels inherit the Excel identity mapping.
    base.RAW_ROOT = RAW_ROOT
    base.SUBJECTS = identity
    base.SESSIONS = sessions
    base.find_recording = robust_find_recording
    base.run(out_root)
    write_paired_summaries(out_root, identity, sessions)
    write_expanded_heatmap(out_root)
    metadata = {
        "identity_source": str(IDENTITY_XLSX),
        "raw_root": str(RAW_ROOT),
        "subject_count": len(identity),
        "sessions": sessions,
        "excluded_duplicate_sessions": excluded_sessions,
        "paired_baseline_sec_test_subjects": [subject for subject in identity if "base_line" in sessions.get(subject, []) and "sec_test" in sessions.get(subject, [])],
        "primary_modality": "movement fNIRS; EEG is low-weight exploratory where QC permits; EMG is protocol-aligned auxiliary output",
        "emg_expected_sensor_ids": list(EXPECTED_SENSOR_IDS),
        "emg_sensor_to_muscle_index": {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
        "emg_sensor_to_muscle_label": {str(k): v for k, v in SENSOR_MUSCLES.items()},
        "emg_mapping_profiles": {
            MAPPING_PROFILE_STANDARD: {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
            MAPPING_PROFILE_SHIFTED_2026080901: {str(k): index for index, k in enumerate(SHIFTED_SENSOR_IDS, start=1)},
        },
        "emg_primary_alignment": "logical_muscle_index; physical sensor IDs retained for audit only",
        "identity_rule": "outer folder ID is authoritative; name is taken only from the Excel cross-check table",
        "subtask_rule": "automatic movement marker times are treated as accurate; task labels follow the confirmed fixed protocol order; marker value 1 is not a task code",
        "caution": "paired summaries are descriptive; independent sample is the subject, not event/block count",
    }
    (out_root / "expanded_analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"subjects={len(identity)} paired={len(metadata['paired_baseline_sec_test_subjects'])} output={out_root.resolve()}")


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=None, help="raw-data root (default: historical J: path)")
    parser.add_argument(
        "--qc-inventory",
        type=Path,
        default=None,
        help="recording_inventory.csv generated by main.py; optional",
    )
    args = parser.parse_args()
    run(args.out, data_root=args.data_root, qc_inventory=args.qc_inventory)


if __name__ == "__main__":
    main_cli()
