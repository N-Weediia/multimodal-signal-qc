"""Quality-gated relative MBLL review for the fixed paired-seven cohort.

This is deliberately a new analysis branch.  It leaves the 20260908 prototype
and all previous outputs untouched.  The branch uses raw intensity to form
natural-log optical density (OD), applies the already accepted stage channel
gate and sample masks, and never treats repaired/filter-padding samples as
measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal
from scipy.io import loadmat

from fnirs_qc import read_fnirs

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_repro_20260903"
STAGE = ROOT / "outputs" / "preprocessed" / "stage_20260904_eye_fnirs_imu_verified"
OUT = ROOT / "outputs" / "mbll_reviewed_20260908"
PAIRED7 = ("2026080901", "2026081401", "2026081403", "2026081501", "2026081502", "2026081503", "2026081701")
# These three conditions are the only completed, protocol-defined movement
# blocks.  The later 2026081401/2026081403 grip markers are retained in the
# audit, but are intentionally never made into event estimates: their start/
# end boundaries and protocol completeness are not established.
CORE_MOVEMENT_TASKS = ("摸头", "摸膝", "摸腰")

# Scott Prahl's OMLC tabulation, copied at the exact device wavelengths.  The
# table's e values are decadic absorbance coefficients (cm^-1 M^-1): OMLC says
# to multiply by 2.303 to obtain the natural-log absorption coefficient mu_a.
# This script forms OD=-ln(I/I0), therefore it uses E_NATURAL=ln(10)*E_DEC.
# The equivalent decadic implementation is OD_dec=OD_natural/ln(10) with E_DEC.
EXTINCTION_E_DEC = {
    730.0: {"hbo": 390.0, "hbr": 1102.2},
    850.0: {"hbo": 1058.0, "hbr": 691.32},
}
EXTINCTION_E_NATURAL = {
    wavelength: {species: float(np.log(10.0) * coefficient) for species, coefficient in values.items()}
    for wavelength, values in EXTINCTION_E_DEC.items()
}
FILTER_GRID = ((0.005, 0.2), (0.01, 0.2), (0.01, 0.3))
DISTANCE_MODES = ("stage_geometry", "fixed_5cm")
DPF_PROFILES = (
    "age_formula",
    "age_formula_common_0p8",
    "age_formula_common_1p2",
    "age_formula_ratio730_0p8",
    "age_formula_ratio730_1p2",
    "fixed_6_6",
)
DEFAULT_VARIANT = "age_formula__stage_geometry__f0.01-0.2"
MIN_WINDOW_COVERAGE = 0.80
MIN_SEGMENT_SECONDS = 3.0
RUN_CONTEXT: dict[str, str] = {"pytest_result": "not_recorded", "pytest_version": "not_recorded", "smoke_result": "not_recorded"}


@dataclass
class EventSeries:
    times: np.ndarray
    source: str


def read_fnirs_events(path: Path | None) -> EventSeries | None:
    """Dependency-light copy of the accepted fNIRS marker reader.

    This is intentionally byte-for-byte equivalent in semantics to the
    validated ``explore_202608_cases.read_fnirs_events`` path, while avoiding
    that analysis module's optional NetworkX dependency. It reads the raw
    ``s`` array and deduplicates only repeated status-column onsets <=1 s.
    """
    if path is None or not path.exists():
        return None
    document = loadmat(path, variable_names=["t", "s"], simplify_cells=True)
    if "t" not in document or "s" not in document:
        return None
    time = np.ravel(np.asarray(document["t"], dtype=float))
    marker = np.asarray(document["s"])
    indices = np.flatnonzero(marker != 0) if marker.ndim == 1 else np.flatnonzero(np.any(marker != 0, axis=1))
    if not indices.size:
        return None
    raw_times = time[indices]
    kept = [float(raw_times[0])]
    for value in raw_times[1:]:
        if float(value) - kept[-1] > 1.0:
            kept.append(float(value))
        elif float(value) < kept[-1]:
            kept[-1] = float(value)
    return EventSeries(np.asarray(kept, dtype=float), "fNIRS s marker (deduplicated)")


def infer_movement_event_labels(times: np.ndarray, subject: str) -> list[dict[str, Any]]:
    """Mirror the shared verified fixed-protocol sequence, never marker values."""
    labels: list[dict[str, Any]] = []
    action_map = {
        3: ("摸头", 1), 4: ("摸头", 2), 5: ("摸头", 3), 6: ("摸头", 4), 7: ("摸头", 5),
        9: ("摸膝", 1), 10: ("摸膝", 2), 11: ("摸膝", 3), 12: ("摸膝", 4), 13: ("摸膝", 5),
        15: ("摸腰", 1), 16: ("摸腰", 2), 17: ("摸腰", 3), 18: ("摸腰", 4), 19: ("摸腰", 5),
    }
    for index, onset in enumerate(np.asarray(times, dtype=float), start=1):
        if index == 1:
            task_name, event_role, status, confidence, block = "运动任务同步起点", "sync_start", "observed", "high", ""
        elif index == 2:
            task_name, event_role, status, confidence, block = "首个动作前准备/边界", "pre_first_action_boundary", "protocol_labeled", "high", ""
        elif index in action_map:
            task_name, block = action_map[index]
            event_role, status, confidence = "action_block_start", "protocol_labeled", "high"
        elif index == 8:
            task_name, event_role, status, confidence, block = "摸头-摸膝转换边界", "task_transition", "protocol_labeled", "high", ""
        elif index == 14:
            task_name, event_role, status, confidence, block = "摸膝-摸腰转换边界", "task_transition", "protocol_labeled", "high", ""
        elif index == 20:
            task_name, event_role, status, confidence, block = "前三动作结束边界", "task_end_boundary", "protocol_labeled", "high", ""
        elif subject in {"2026081401", "2026081403"} and 20 <= index <= 25:
            task_name, event_role, status, confidence, block = "握拳", "action_block_start", "protocol_labeled", "low", index - 19
        else:
            task_name, event_role, status, confidence, block = "未分类/异常标记", "unclassified", "unclassified", "low", ""
        previous_interval = f"previous_marker_interval_seconds={float(times[index - 1] - times[index - 2]):.2f}" if index > 1 else ""
        labels.append({"marker_index": index, "task_name": task_name, "event_role": event_role,
                       "inference_status": status, "confidence": confidence, "block_index": block,
                       "evidence": previous_interval or "first_valid_marker", "timestamp_seconds": float(onset)})
    return labels


def dpf_formula(wavelength_nm: float, age_years: float) -> float:
    """Scholkmann & Wolf (2013), evaluated without clipping age or wavelength."""
    w = float(wavelength_nm)
    a = float(age_years)
    return 223.3 + 0.05624 * a ** 0.8493 - 5.723e-7 * w**3 + 0.001245 * w**2 - 0.9025 * w


def parse_filename_dates(path: Path) -> dict[str, Any]:
    """Use filename dates only as unverified metadata, never directory dates."""
    found = re.findall(r"(?:19|20)\d{2}-\d{2}-\d{2}", path.name)
    result: dict[str, Any] = {"acquisition_date": "", "birth_date": "", "age_years": np.nan, "age_status": "unavailable"}
    if len(found) >= 2:
        acquisition, birth = date.fromisoformat(found[0]), date.fromisoformat(found[1])
        result.update({
            "acquisition_date": acquisition.isoformat(),
            "birth_date": birth.isoformat(),
            "age_years": (acquisition - birth).days / 365.2425,
            "age_status": "derived_from_filename_dates_unverified",
        })
    return result


def dpf_values(profile: str, age_years: float) -> np.ndarray:
    base = np.asarray([dpf_formula(730.0, age_years), dpf_formula(850.0, age_years)], dtype=float)
    if profile == "age_formula":
        return base
    if profile == "age_formula_common_0p8":
        return base * 0.8
    if profile == "age_formula_common_1p2":
        return base * 1.2
    if profile == "age_formula_ratio730_0p8":
        return base * np.asarray([0.8, 1.0])
    if profile == "age_formula_ratio730_1p2":
        return base * np.asarray([1.2, 1.0])
    if profile == "fixed_6_6":
        return np.asarray([6.0, 6.0])
    raise ValueError(f"Unknown DPF profile {profile}")


def dpf_scope(age_years: float) -> str:
    flags = ["850_nm_outside_690_832_validation_display_range", "frontal_site_not_verified_from_raw_metadata"]
    if not np.isfinite(age_years):
        flags.append("age_unavailable")
    elif age_years > 50:
        flags.append("age_above_0_50_validation_display_range")
    elif age_years < 0:
        flags.append("invalid_negative_age")
    return ";".join(flags)


def validate_wavelength_order(wavelength_pairs: Iterable[tuple[float, float]]) -> None:
    """Reject reversed or mixed raw wavelength columns before any MBLL solve."""
    observed = set(wavelength_pairs)
    if observed != {(730.0, 850.0)}:
        raise ValueError(f"Expected raw column order (730, 850) nm only; observed {observed}")


def read_stage_records(task_contains: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load stage-approved recordings; duplicate candidates are recorded, not merged."""
    candidates: list[dict[str, Any]] = []
    for summary_path in STAGE.rglob("preprocessing_summary.json"):
        item = json.loads(summary_path.read_text(encoding="utf-8"))
        if item.get("modality") != "FNIRS" or task_contains not in str(item.get("task", "")):
            continue
        source = str(item.get("source_relative_path", ""))
        raw_path = RAW / source
        if not raw_path.exists():
            raise FileNotFoundError(f"Stage source is not in reproducible input: {raw_path}")
        candidates.append({"summary_path": summary_path, "stage": item, "raw_path": raw_path, "source_relative_path": source})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        parts = Path(item["source_relative_path"]).parts
        grouped[(parts[0], parts[1])].append(item)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for (subject, session), items in sorted(grouped.items()):
        # This preserves the stage's original deterministic reader choice: the
        # lexicographically first record per subject/session/task.  It is an
        # explicit repeated-record strategy, not a pooled observation.
        items.sort(key=lambda x: x["source_relative_path"])
        chosen = items[0]
        chosen["subject"], chosen["session"] = subject, session
        selected.append(chosen)
        for discarded in items[1:]:
            excluded.append({"subject": subject, "session": session, "task_branch": task_contains,
                             "analysis_branch": discarded["stage"].get("analysis_branch", "unrecorded"),
                             "source_relative_path": discarded["source_relative_path"],
                             "reason": "duplicate_stage_candidate; lexicographically_first_stage_source_retained"})
    return selected, excluded


def load_stage_mask(record: dict[str, Any], channel_labels: list[str], n_samples: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map stage kept-channel masks back to the raw-channel order."""
    stage_dir = record["summary_path"].parent
    mask_path = stage_dir / "fnirs_cleaned_od.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing accepted stage mask: {mask_path}")
    archive = np.load(mask_path, allow_pickle=False)
    kept_labels = [str(x) for x in archive["labels"]]
    stage_mask = np.asarray(archive["analysis_sample_mask"], dtype=bool)
    if stage_mask.shape[1] != n_samples:
        raise ValueError(f"Stage mask length {stage_mask.shape[1]} != raw length {n_samples}: {mask_path}")
    index = {label: i for i, label in enumerate(channel_labels)}
    selected_indices = []
    for label in kept_labels:
        if label not in index:
            raise ValueError(f"Stage label not found in raw reader: {label}")
        selected_indices.append(index[label])
    full_mask = np.zeros((n_samples, len(channel_labels)), dtype=bool)
    full_mask[:, selected_indices] = stage_mask.T
    channel_gate = np.zeros(len(channel_labels), dtype=bool)
    channel_gate[selected_indices] = True
    metadata = {"mask_path": str(mask_path.relative_to(ROOT)), "kept_channel_count": int(channel_gate.sum()),
                "stage_analysis_sample_percent": float(stage_mask.mean() * 100.0)}
    return full_mask, channel_gate, metadata


def raw_natural_od(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for channel in range(values.shape[1]):
        valid = valid_mask[:, channel] & np.isfinite(values[:, channel]) & (values[:, channel] > 0)
        if valid.sum() < 2:
            continue
        baseline = float(np.median(values[valid, channel]))
        result[valid, channel] = -np.log(values[valid, channel] / baseline)
    return result


def robust_z_masked(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    for channel in range(values.shape[1]):
        valid = valid_mask[:, channel] & np.isfinite(values[:, channel])
        if valid.sum() < 2:
            continue
        median = float(np.median(values[valid, channel]))
        scale = float(np.median(np.abs(values[valid, channel] - median)) * 1.4826)
        output[valid, channel] = (values[valid, channel] - median) / max(scale, 1e-12)
    return output


def contiguous_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def filter_masked(values: np.ndarray, valid_mask: np.ndarray, fs: float, low: float, high: float) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Filter actual valid runs only; gap fills and filtfilt padding stay invalid."""
    if high >= fs * 0.5:
        raise ValueError(f"High cutoff {high} invalid for fs={fs}")
    sos = signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
    # scipy's default pad is short, but a 3 s core exclusion is intentional:
    # it makes filter padding distinguishable from usable observations.
    edge = max(int(np.ceil(3.0 * max(len(section) for section in sos) / fs)), int(np.ceil(fs * 1.0)))
    filtered = np.full(values.shape, np.nan, dtype=float)
    usable = np.zeros(values.shape, dtype=bool)
    audit = {"short_invalid_or_motion_gap_count": 0, "long_invalid_or_motion_gap_count": 0,
             "short_valid_segment_count": 0, "filter_padding_excluded_samples": 0}
    for channel in range(values.shape[1]):
        channel_mask = valid_mask[:, channel] & np.isfinite(values[:, channel])
        invalid = ~channel_mask
        for start, end in contiguous_runs(invalid):
            if end - start >= int(round(fs)):
                audit["long_invalid_or_motion_gap_count"] += 1
            else:
                audit["short_invalid_or_motion_gap_count"] += 1
        for start, end in contiguous_runs(channel_mask):
            length = end - start
            if length < max(int(np.ceil(MIN_SEGMENT_SECONDS * fs)), 2 * edge + 3):
                audit["short_valid_segment_count"] += 1
                continue
            filtered[start:end, channel] = signal.sosfiltfilt(sos, values[start:end, channel])
            core_start, core_end = start + edge, end - edge
            if core_end > core_start:
                usable[core_start:core_end, channel] = True
                audit["filter_padding_excluded_samples"] += 2 * edge
    filtered[~usable] = np.nan
    return filtered, usable, audit


def unmix_natural_od(od_wavelength_channel: np.ndarray, dpf_pair: np.ndarray, distance_cm: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve natural-log OD with E_NATURAL=ln(10)*E_DEC exactly once."""
    ext = np.asarray([[EXTINCTION_E_NATURAL[730.0]["hbo"], EXTINCTION_E_NATURAL[730.0]["hbr"]],
                      [EXTINCTION_E_NATURAL[850.0]["hbo"], EXTINCTION_E_NATURAL[850.0]["hbr"]]], dtype=float)
    design = ext * dpf_pair[:, None]
    concentrations = np.einsum("sw,twc->tsc", np.linalg.pinv(design), od_wavelength_channel)
    concentrations /= distance_cm[None, None, :]
    return concentrations * 1e6, float(np.linalg.cond(design))


def is_completed_movement_action(label: dict[str, Any]) -> bool:
    """True only for the fifteen prespecified markers in completed tasks."""
    return bool(label.get("event_role") == "action_block_start" and label.get("task_name") in CORE_MOVEMENT_TASKS)


def event_rows_for_signal(record: dict[str, Any], time: np.ndarray, signal_values: np.ndarray, usable: np.ndarray,
                          channel_gate: np.ndarray, event_labels: list[dict[str, Any]], variant: str,
                          signal_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in event_labels:
        if not is_completed_movement_action(label):
            continue
        onset = float(label["timestamp_seconds"])
        baseline = (time >= onset - 5.0) & (time < onset)
        response = (time >= onset + 2.0) & (time < onset + 12.0)
        if not baseline.any() or not response.any():
            rows.append({"subject": record["subject"], "session": record["session"], "task_name": label["task_name"],
                         "marker_index": label["marker_index"], "onset_seconds": onset, "variant": variant,
                         "signal": signal_name, "status": "outside_recording", "channel_count_gated": int(channel_gate.sum()),
                         "channel_count_used": 0, "baseline_coverage_mean": np.nan, "response_coverage_mean": np.nan,
                         "delta": np.nan})
            continue
        base_coverage = usable[baseline].mean(axis=0)
        response_coverage = usable[response].mean(axis=0)
        eligible = channel_gate & (base_coverage >= MIN_WINDOW_COVERAGE) & (response_coverage >= MIN_WINDOW_COVERAGE)
        # Reduce only channels which already meet coverage. This avoids both
        # fabricated gap values and meaningless all-NaN reductions for a
        # rejected channel.
        delta = np.full(signal_values.shape[1], np.nan, dtype=float)
        if eligible.any():
            delta[eligible] = (np.nanmean(signal_values[response][:, eligible], axis=0) -
                               np.nanmean(signal_values[baseline][:, eligible], axis=0))
        valid_delta = eligible & np.isfinite(delta)
        rows.append({"subject": record["subject"], "session": record["session"], "task_name": label["task_name"],
                     "marker_index": label["marker_index"], "onset_seconds": onset, "variant": variant, "signal": signal_name,
                     "status": "usable" if valid_delta.any() else "coverage_or_channel_gate_failed",
                     "channel_count_gated": int(channel_gate.sum()), "channel_count_used": int(valid_delta.sum()),
                     "baseline_coverage_mean": float(np.mean(base_coverage[channel_gate])) if channel_gate.any() else np.nan,
                     "response_coverage_mean": float(np.mean(response_coverage[channel_gate])) if channel_gate.any() else np.nan,
                     "delta": float(np.mean(delta[valid_delta])) if valid_delta.any() else np.nan})
    return rows


def movement_marker_audit(record: dict[str, Any], event_labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve raw marker order while separating protocol labels from marker values.

    The hardware stream contains value 1 only.  It contributes timestamp/
    boundary evidence, never a task identity.  Identity is supplied only by
    the verified outer ``运动`` branch and fixed sequence in the shared helper.
    """
    rows: list[dict[str, Any]] = []
    for label in event_labels:
        core = is_completed_movement_action(label)
        if core:
            status = "included_core_protocol_action"
            reason = "outer-directory movement branch plus fixed confirmed sequence"
        elif label["task_name"] == "握拳":
            status = "excluded_incomplete_grip_extension"
            reason = "later extension has incomplete/uncertain boundaries; no event window imputed"
        else:
            status = "boundary_or_unclassified_not_analyzed"
            reason = "synchronization/transition/boundary marker is not an action estimate"
        rows.append({
            "subject": record["subject"], "session": record["session"],
            "source_relative_path": record["source_relative_path"],
            "marker_index": label["marker_index"], "onset_seconds": label["timestamp_seconds"],
            "hardware_marker_value": "1", "hardware_value_semantics": "synchronization_or_boundary_only_not_task_identity",
            "protocol_task_label": label["task_name"], "protocol_event_role": label["event_role"],
            "protocol_inference_status": label["inference_status"], "analysis_status": status,
            "reason": reason, "label_evidence": label["evidence"],
        })
    return rows


def eeg_eog_ecg_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record the user-confirmed absence of usable EOG/ECG without reading it as physiology."""
    rows = []
    for record in records:
        parts = Path(record["source_relative_path"]).parts
        eeg_relative = str(Path(parts[0]) / parts[1] / "EEG" / "运动" / "data.bdf")
        eeg_stage_summary = STAGE / parts[0] / parts[1] / "EEG" / "运动" / "preprocessing_summary.json"
        rows.append({
            "subject": record["subject"], "session": record["session"],
            "eeg_source_relative_path": eeg_relative,
            "stage_eeg_summary_exists": eeg_stage_summary.exists(),
            "eog_ecg_channel_names": "EOG/ECG-named acquisition channels",
            "acquisition_fact": "user-confirmed not connected during acquisition",
            "qc_status": "missing_unconnected_reference_not_usable",
            "correction_attempted": False,
            "correction_policy": "no EOG/ECG regression, ICA component selection, reference subtraction, or physiological interpretation",
        })
    return rows


def aggregate_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["session"], row["task_name"], row["variant"], row["signal"])].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        usable = [row for row in group if row["status"] == "usable" and np.isfinite(row["delta"])]
        result.append({"subject": key[0], "session": key[1], "task_name": key[2], "variant": key[3], "signal": key[4],
                       "n_protocol_events": len(group), "n_events_used": len(usable),
                       "event_delta_mean": float(np.mean([x["delta"] for x in usable])) if usable else np.nan,
                       "mean_channel_count_used": float(np.mean([x["channel_count_used"] for x in usable])) if usable else np.nan,
                       "mean_baseline_coverage": float(np.mean([x["baseline_coverage_mean"] for x in usable])) if usable else np.nan,
                       "mean_response_coverage": float(np.mean([x["response_coverage_mean"] for x in usable])) if usable else np.nan})
    return result


def require_unique(rows: list[dict[str, Any]], fields: tuple[str, ...], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in output:
            raise ValueError(f"Non-unique {label} key {key}; refusing a many-to-many merge")
        output[key] = row
    return output


def paired_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("subject", "session", "task_name", "variant", "signal")
    index = require_unique(summary_rows, key_fields, "subject-session-task-variant-signal summary")
    result = []
    for subject in PAIRED7:
        combinations = {(row["task_name"], row["variant"], row["signal"]) for row in summary_rows if row["subject"] == subject}
        for task_name, variant, signal_name in sorted(combinations):
            base = index.get((subject, "base_line", task_name, variant, signal_name))
            follow = index.get((subject, "sec_test", task_name, variant, signal_name))
            if base is None or follow is None:
                continue
            b, f = base["event_delta_mean"], follow["event_delta_mean"]
            result.append({"subject": subject, "task_name": task_name, "variant": variant, "signal": signal_name,
                           "baseline_value": b, "followup_value": f,
                           "followup_minus_baseline": float(f - b) if np.isfinite(b) and np.isfinite(f) else np.nan,
                           "baseline_events_used": base["n_events_used"], "followup_events_used": follow["n_events_used"]})
    return result


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if np.isfinite(row["followup_minus_baseline"]):
            grouped[(row["task_name"], row["variant"], row["signal"])].append(float(row["followup_minus_baseline"]))
    out = []
    for (task, variant, signal_name), values in sorted(grouped.items()):
        x = np.asarray(values)
        out.append({"task_name": task, "variant": variant, "signal": signal_name, "n_subjects": int(x.size),
                    "mean_followup_minus_baseline": float(x.mean()), "median_followup_minus_baseline": float(np.median(x)),
                    "positive_count": int((x > 0).sum()), "negative_count": int((x < 0).sum()), "zero_count": int((x == 0).sum())})
    return out


def longitudinal_proxy_mbll_comparison(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare participant-level baseline-to-follow-up changes, never session signs."""
    default_mbll = [r for r in summary_rows if r["variant"] == DEFAULT_VARIANT and r["signal"] == "mbll_hbo_uM"]
    default_raw = [r for r in summary_rows if r["variant"] == "od_proxy__f0.01-0.2" and r["signal"] == "raw_od_proxy"]
    default_hist = [r for r in summary_rows if r["variant"] == "od_proxy__f0.01-0.2" and r["signal"] == "historical_z_od_proxy"]
    key = ("subject", "session", "task_name")
    m, raw, hist = (require_unique(x, key, label) for x, label in ((default_mbll, "default MBLL longitudinal"), (default_raw, "raw OD longitudinal"), (default_hist, "historical OD longitudinal")))
    rows = []
    for subject in PAIRED7:
        for task_name in CORE_MOVEMENT_TASKS:
            bkey, fkey = (subject, "base_line", task_name), (subject, "sec_test", task_name)
            if not all(k in d for k in (bkey, fkey) for d in (m, raw, hist)):
                continue
            m_delta = m[fkey]["event_delta_mean"] - m[bkey]["event_delta_mean"]
            raw_delta = raw[fkey]["event_delta_mean"] - raw[bkey]["event_delta_mean"]
            hist_delta = hist[fkey]["event_delta_mean"] - hist[bkey]["event_delta_mean"]
            def same(a: float, b: float) -> bool | str:
                return bool(np.sign(a) == np.sign(b)) if np.isfinite(a) and np.isfinite(b) and a != 0 and b != 0 else "undefined"
            rows.append({
                "subject": subject, "task_name": task_name,
                "mbll_hbo_followup_minus_baseline": m_delta,
                "raw_od_proxy_followup_minus_baseline": raw_delta,
                "historical_z_od_proxy_followup_minus_baseline": hist_delta,
                "same_change_direction_mbll_vs_raw_od": same(m_delta, raw_delta),
                "same_change_direction_mbll_vs_historical_proxy": same(m_delta, hist_delta),
                "comparison_interpretation": "participant-level baseline-to-sec_test change; same task, event windows, stage channel gate, and restored sample mask; descriptive only",
            })
    return rows


def task_contrast_rows(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Requested head-minus-knee contrast, kept distinct from within-task responses."""
    relevant = [r for r in summary_rows if r["signal"] in {"mbll_hbo_uM", "mbll_hbr_uM", "raw_od_proxy", "historical_z_od_proxy"} and (
        (r["signal"] in {"mbll_hbo_uM", "mbll_hbr_uM"} and r["variant"] == DEFAULT_VARIANT) or
        (r["signal"] not in {"mbll_hbo_uM", "mbll_hbr_uM"} and r["variant"] == "od_proxy__f0.01-0.2")
    )]
    index = require_unique(relevant, ("subject", "session", "task_name", "signal"), "head-minus-knee source")
    subject_session = []
    paired = []
    for subject in PAIRED7:
        values: dict[tuple[str, str], float] = {}
        for session in ("base_line", "sec_test"):
            for signal_name in ("mbll_hbo_uM", "mbll_hbr_uM", "raw_od_proxy", "historical_z_od_proxy"):
                head, knee = index.get((subject, session, "摸头", signal_name)), index.get((subject, session, "摸膝", signal_name))
                if head is None or knee is None:
                    continue
                contrast = head["event_delta_mean"] - knee["event_delta_mean"]
                values[(session, signal_name)] = contrast
                subject_session.append({"subject": subject, "session": session, "signal": signal_name,
                                        "contrast": "head_minus_contralateral_knee", "value": contrast,
                                        "interpretation": "requested task contrast, descriptive; separate from within-task and longitudinal effects"})
        for signal_name in ("mbll_hbo_uM", "mbll_hbr_uM", "raw_od_proxy", "historical_z_od_proxy"):
            if ("base_line", signal_name) in values and ("sec_test", signal_name) in values:
                paired.append({"subject": subject, "signal": signal_name, "contrast": "head_minus_contralateral_knee",
                               "baseline_value": values[("base_line", signal_name)], "followup_value": values[("sec_test", signal_name)],
                               "followup_minus_baseline": values[("sec_test", signal_name)] - values[("base_line", signal_name)],
                               "interpretation": "requested task-contrast change, descriptive participant-level comparison"})
    return subject_session, paired


def sensitivity_stability(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize whether subject-level longitudinal signs survive each choice."""
    selected = [r for r in paired if r["signal"] == "mbll_hbo_uM"]
    default = require_unique([r for r in selected if r["variant"] == DEFAULT_VARIANT], ("subject", "task_name"), "default sensitivity changes")
    rows = []
    for task_name in CORE_MOVEMENT_TASKS:
        for variant in sorted({r["variant"] for r in selected if r["task_name"] == task_name}):
            values = [r for r in selected if r["task_name"] == task_name and r["variant"] == variant and np.isfinite(r["followup_minus_baseline"])]
            comparable = []
            for row in values:
                ref = default.get((row["subject"], task_name))
                if ref is not None and np.isfinite(ref["followup_minus_baseline"]):
                    comparable.append((row["followup_minus_baseline"], ref["followup_minus_baseline"]))
            x = np.asarray([v[0] for v in comparable], dtype=float)
            ref = np.asarray([v[1] for v in comparable], dtype=float)
            nonzero = (x != 0) & (ref != 0)
            ratios = np.abs(x[nonzero] / ref[nonzero]) if nonzero.any() else np.asarray([])
            common_scale = "age_formula_common_" in variant
            distance_note = "fixed_distance" if "fixed_5cm" in variant else "coordinate_geometry_distance"
            rows.append({
                "task_name": task_name, "variant": variant, "n_subjects_with_change": len(values),
                "n_sign_comparable": int(nonzero.sum()), "sign_match_count": int(np.sum(np.sign(x[nonzero]) == np.sign(ref[nonzero]))),
                "sign_match_rate_vs_default": float(np.mean(np.sign(x[nonzero]) == np.sign(ref[nonzero]))) if nonzero.any() else np.nan,
                "mean_followup_minus_baseline_uM": float(np.mean([r["followup_minus_baseline"] for r in values])) if values else np.nan,
                "median_followup_minus_baseline_uM": float(np.median([r["followup_minus_baseline"] for r in values])) if values else np.nan,
                "min_followup_minus_baseline_uM": float(np.min([r["followup_minus_baseline"] for r in values])) if values else np.nan,
                "max_followup_minus_baseline_uM": float(np.max([r["followup_minus_baseline"] for r in values])) if values else np.nan,
                "median_abs_amplitude_ratio_vs_default": float(np.median(ratios)) if ratios.size else np.nan,
                "min_abs_amplitude_ratio_vs_default": float(np.min(ratios)) if ratios.size else np.nan,
                "max_abs_amplitude_ratio_vs_default": float(np.max(ratios)) if ratios.size else np.nan,
                "sensitivity_note": ("common DPF scaling is algebraically expected to change amplitude more than sign; " if common_scale else "wavelength-ratio DPF change can alter HbO/HbR separation; ") +
                                    ("fixed/common distance primarily rescales amplitude, though heterogeneous channel geometry can reweight a channel average; " if distance_note == "fixed_distance" else "coordinate-derived distance can reweight channel averages; ") +
                                    ("filter band can change a window contrast"),
            })
    return rows


def exact_sign_flip_p(values: list[float]) -> float:
    """Two-sided exact sign-flip p for the mean; suitable only for tiny n."""
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not x.size:
        return np.nan
    observed = abs(float(np.mean(x)))
    null = np.asarray([abs(float(np.mean(np.asarray(signs) * x))) for signs in product((-1.0, 1.0), repeat=x.size)])
    return float(np.mean(null >= observed - np.finfo(float).eps))


def exploratory_exact_tests(paired: list[dict[str, Any]], head_knee_paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unadjusted participant-level sign-flip tests, retained as exploratory only."""
    rows = []
    for signal_name in ("mbll_hbo_uM", "mbll_hbr_uM"):
        for task_name in CORE_MOVEMENT_TASKS:
            values = [r["followup_minus_baseline"] for r in paired if r["variant"] == DEFAULT_VARIANT and r["signal"] == signal_name and r["task_name"] == task_name]
            values = [float(v) for v in values if np.isfinite(v)]
            rows.append({"analysis": "within_task_baseline_to_sec_test", "signal": signal_name, "task_or_contrast": task_name,
                         "n_subjects": len(values), "mean_change": float(np.mean(values)) if values else np.nan,
                         "exact_two_sided_sign_flip_p": exact_sign_flip_p(values),
                         "interpretation": "exploratory exact participant-level sign-flip test; no multiplicity correction; not confirmatory"})
    for signal_name in ("mbll_hbo_uM", "mbll_hbr_uM"):
        values = [float(r["followup_minus_baseline"]) for r in head_knee_paired if r["signal"] == signal_name and np.isfinite(r["followup_minus_baseline"])]
        rows.append({"analysis": "task_contrast_baseline_to_sec_test", "signal": signal_name, "task_or_contrast": "head_minus_contralateral_knee",
                     "n_subjects": len(values), "mean_change": float(np.mean(values)) if values else np.nan,
                     "exact_two_sided_sign_flip_p": exact_sign_flip_p(values),
                     "interpretation": "exploratory requested task-contrast exact sign-flip test; no multiplicity correction; not confirmatory"})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_figures(paired: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    default = [r for r in paired if r["variant"] == DEFAULT_VARIANT and r["signal"] == "mbll_hbo_uM"]
    tasks = ["摸头", "摸膝", "摸腰"]
    task_display = {"摸头": "Head touch", "摸膝": "Contralateral knee", "摸腰": "Back-waist touch"}
    # A shared ordinate avoids visual overstatement from task-specific scaling.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, tasks):
        vals = [r for r in default if r["task_name"] == task and np.isfinite(r["baseline_value"]) and np.isfinite(r["followup_value"])]
        for row in vals:
            ax.plot([0, 1], [row["baseline_value"], row["followup_value"]], color="#5470c6", alpha=.55, lw=1)
        ax.scatter(np.zeros(len(vals)), [r["baseline_value"] for r in vals], color="#5470c6", s=18, label="baseline")
        ax.scatter(np.ones(len(vals)), [r["followup_value"] for r in vals], color="#ee6666", s=18, label="follow-up")
        ax.set_title(f"{task_display[task]} (n={len(vals)})")
        ax.set_xticks([0, 1], ["baseline", "follow-up"])
        ax.set_ylabel("event ΔHbO (µM)")
        ax.axhline(0, color="0.75", lw=.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired subject-level MBLL event response (default sensitivity branch)", fontsize=11)
    fig.savefig(OUT / "figure_paired_hbo_default.png", dpi=220)
    plt.close(fig)

    vals = [r for r in comparison if np.isfinite(r["mbll_hbo_uM"]) and np.isfinite(r["raw_od_proxy"]) and r["task_name"] in tasks]
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    colors = {"摸头": "#5470c6", "摸膝": "#91cc75", "摸腰": "#fac858"}
    for task in tasks:
        subset = [r for r in vals if r["task_name"] == task]
        ax.scatter([r["raw_od_proxy"] for r in subset], [r["mbll_hbo_uM"] for r in subset], s=28, alpha=.8, label=task_display[task], color=colors[task])
    ax.axhline(0, color="0.75", lw=.8); ax.axvline(0, color="0.75", lw=.8)
    ax.set(xlabel="raw-OD wavelength contrast Δ", ylabel="MBLL ΔHbO (µM)", title="Same window/channel/mask comparison")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "figure_od_mbll_same_window.png", dpi=220)
    plt.close(fig)


def generate_teacher_assets(paired: list[dict[str, Any]], longitudinal: list[dict[str, Any]], sensitivity: list[dict[str, Any]],
                            paired_stats: list[dict[str, Any]], source_audit: list[dict[str, Any]], marker_audit: list[dict[str, Any]],
                            eye_rows: list[dict[str, Any]], eeg_rows: list[dict[str, Any]], exact_tests: list[dict[str, Any]]) -> None:
    """Create aggregate-only teaching assets with no names, IDs, DOBs, or paths."""
    import matplotlib.pyplot as plt

    teacher = OUT / "teacher_assets"
    teacher.mkdir(exist_ok=True)
    # The source figure already contains no subject identifiers; copy it into
    # the explicitly anonymized teaching bundle rather than rebuilding data.
    (teacher / "default_mbll_hbo_paired.png").write_bytes((OUT / "figure_paired_hbo_default.png").read_bytes())
    raw = sum(r["same_change_direction_mbll_vs_raw_od"] == True for r in longitudinal)
    historical = sum(r["same_change_direction_mbll_vs_historical_proxy"] == True for r in longitudinal)
    total = len(longitudinal)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    bars = ax.bar(["Raw OD proxy", "Historical z-OD proxy"], [raw / total if total else 0, historical / total if total else 0], color=["#5470c6", "#91cc75"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Same longitudinal-change direction fraction")
    ax.set_title("MBLL direction agreement (aggregate only)")
    for bar, count in zip(bars, (raw, historical)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .025, f"{count}/{total}", ha="center", va="bottom")
    fig.savefig(teacher / "od_mbll_longitudinal_direction_agreement.png", dpi=220)
    plt.close(fig)
    task_display = {"摸头": "Head touch", "摸膝": "Contralateral knee", "摸腰": "Back-waist touch"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), constrained_layout=True)
    ax, ratio_ax = axes
    task_order = list(CORE_MOVEMENT_TASKS)
    rates = []
    for task in task_order:
        values = [float(r["sign_match_rate_vs_default"]) for r in sensitivity if r["task_name"] == task and r["sign_match_rate_vs_default"] != ""]
        rates.append(float(np.mean(values)) if values else np.nan)
    bars = ax.bar([task_display[t] for t in task_order], rates, color=["#5470c6", "#91cc75", "#fac858"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Mean sign agreement vs default")
    ax.set_title("Sensitivity stability across 36 settings per task")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + .025, f"{rate:.0%}", ha="center", va="bottom")
    # One point per sensitivity setting: its median participant-level |Δ|
    # ratio to the default. Vertical lines show the setting-level range.
    rng = np.random.default_rng(20260908)
    for index, task in enumerate(task_order):
        medians = np.asarray([float(r["median_abs_amplitude_ratio_vs_default"]) for r in sensitivity if r["task_name"] == task and r["median_abs_amplitude_ratio_vs_default"] != ""], dtype=float)
        if not medians.size:
            continue
        jitter = rng.uniform(-.14, .14, size=medians.size)
        ratio_ax.scatter(np.full(medians.size, index) + jitter, medians, s=14, alpha=.55, color=["#5470c6", "#91cc75", "#fac858"][index])
        ratio_ax.vlines(index, np.min(medians), np.max(medians), color="0.25", lw=1.2)
        ratio_ax.scatter([index], [np.median(medians)], marker="D", s=35, color="black", zorder=4)
    ratio_ax.axhline(1, color="0.4", ls="--", lw=.9)
    ratio_ax.set_xticks(range(len(task_order)), [task_display[t] for t in task_order])
    ratio_ax.set_ylabel("|longitudinal Δ| / default\n(per-setting median; points and min-max range)")
    ratio_ax.set_title("Amplitude sensitivity across settings")
    fig.suptitle("Sensitivity summary: direction stability does not imply amplitude stability", fontsize=11)
    fig.text(.5, -.02, "All panels are descriptive and do not establish neural specificity.", ha="center", fontsize=8)
    fig.savefig(teacher / "sensitivity_sign_stability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    default_hbo = [r for r in paired_stats if r["variant"] == DEFAULT_VARIANT and r["signal"] == "mbll_hbo_uM"]
    distance = [float(r["distance_min_cm"]) for r in source_audit] + [float(r["distance_max_cm"]) for r in source_audit]
    core_count = sum(r["analysis_status"] == "included_core_protocol_action" for r in marker_audit)
    grip_count = sum(r["analysis_status"] == "excluded_incomplete_grip_extension" for r in marker_audit)
    rows = [
        {"metric": "movement_records", "value": len(source_audit), "note": "paired-seven baseline/sec_test movement recordings"},
        {"metric": "core_protocol_windows", "value": core_count, "note": "only completed head/knee/back-waist actions"},
        {"metric": "grip_extension_excluded", "value": grip_count, "note": "incomplete/uncertain extension boundaries"},
        {"metric": "raw_OD_direction_matches_MBLL", "value": f"{raw}/{total}", "note": "participant-level longitudinal change"},
        {"metric": "historical_z_OD_direction_matches_MBLL", "value": f"{historical}/{total}", "note": "participant-level longitudinal change"},
        {"metric": "sensitivity_sign_stable", "value": f"{sum(float(r['sign_match_rate_vs_default']) == 1.0 for r in sensitivity)}/{len(sensitivity)}", "note": "task x sensitivity setting rows"},
        {"metric": "eye_records_degraded_inventory_only", "value": len(eye_rows), "note": "EYE_IR_DEGRADED_MASKED; no movement MBLL or IR-removal claim"},
        {"metric": "EEG_EOG_ECG_unconnected_no_correction", "value": len(eeg_rows), "note": "user-confirmed unavailable reference"},
        {"metric": "coordinate_distance_range_cm", "value": f"{min(distance):.2f}-{max(distance):.2f}", "note": "coordinates declare cm; physical cap placement not verified; no short-separation channels"},
        {"metric": "S07_followup_knee_waist_missing", "value": "not analyzed", "note": "deterministically retained follow-up recording has 7 deduplicated markers / 5 action markers (only head markers 3-7); no knee/waist markers, rather than a statistical row drop"},
    ]
    for r in default_hbo:
        rows.append({"metric": f"default_HbO_{task_display[r['task_name']]}", "value": f"n={r['n_subjects']}; mean={float(r['mean_followup_minus_baseline']):.5f} uM; median={float(r['median_followup_minus_baseline']):.5f} uM; +/{r['positive_count']} -/{r['negative_count']}", "note": "follow-up minus baseline; exploratory participant-level summary"})
    for r in exact_tests:
        rows.append({"metric": f"exact_sign_flip_{r['signal']}_{r['task_or_contrast']}", "value": f"n={r['n_subjects']}; p={r['exact_two_sided_sign_flip_p']:.6g}", "note": r["interpretation"]})
    write_csv(teacher / "summary_for_teacher.csv", rows)
    summary_json = {"summary_rows": rows, "anonymization": "No names, dates of birth, source participant identifiers, raw paths, or absolute paths are included in teacher_assets."}
    (teacher / "summary_for_teacher.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")


def eye_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected, excluded = read_stage_records("眼动")
    rows = []
    for record in selected:
        stage = record["stage"]
        rows.append({"subject": record["subject"], "session": record["session"], "source_relative_path": record["source_relative_path"],
                     "analysis_branch": stage.get("analysis_branch"), "known_ir_interference": stage.get("known_ir_interference"),
                     "eog_ecg_reference": "unconnected_confirmed; no correction attempted", "event_policy": "no movement labels or MBLL event estimates applied"})
    return rows, excluded


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, duplicate_exclusions = read_stage_records("运动")
    expected = {(s, session) for s in PAIRED7 for session in ("base_line", "sec_test")}
    found = {(r["subject"], r["session"]) for r in records}
    if found != expected:
        raise ValueError(f"Stage paired-seven movement records do not match expected cohort. Missing={expected-found}, extra={found-expected}")

    event_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    marker_audit_rows: list[dict[str, Any]] = []
    source_audit_rows: list[dict[str, Any]] = []
    for record in records:
        raw_path = record["raw_path"]
        fnirs = read_fnirs(raw_path, None)
        labels = [channel.label for channel in fnirs.channels]
        validate_wavelength_order((channel.wavelength_a_nm, channel.wavelength_b_nm) for channel in fnirs.channels)
        stage_mask, channel_gate, mask_meta = load_stage_mask(record, labels, len(fnirs.time))
        intensity_730 = np.column_stack([channel.intensity_a for channel in fnirs.channels])
        intensity_850 = np.column_stack([channel.intensity_b for channel in fnirs.channels])
        raw_valid = np.isfinite(intensity_730) & np.isfinite(intensity_850) & (intensity_730 > 0) & (intensity_850 > 0)
        analysis_mask = stage_mask & raw_valid
        od730 = raw_natural_od(intensity_730, analysis_mask)
        od850 = raw_natural_od(intensity_850, analysis_mask)
        dates = parse_filename_dates(raw_path)
        if not np.isfinite(dates["age_years"]):
            raise ValueError(f"Age unavailable; refusing a silent default: {raw_path}")
        events = read_fnirs_events(raw_path)
        if events is None:
            raise ValueError(f"No fNIRS marker stream: {raw_path}")
        movement_labels = infer_movement_event_labels(events.times, record["subject"])
        distance = np.asarray([np.linalg.norm(fnirs.source_positions[c.source - 1] - fnirs.detector_positions[c.detector - 1]) for c in fnirs.channels])
        if not np.all(np.isfinite(distance)) or np.any(distance <= 0):
            raise ValueError(f"Invalid source-detector geometry: {raw_path}")
        marker_audit_rows.extend(movement_marker_audit(record, movement_labels))
        source_audit_rows.append({
            "subject": record["subject"], "session": record["session"], "source_relative_path": record["source_relative_path"],
            "stage_summary_path": str(record["summary_path"].relative_to(ROOT)),
            "source_sha256": record["stage"].get("source_sha256", ""),
            "task_assignment": "outer-directory FNIRS/运动 plus fixed protocol helper; never hardware marker value",
            "age_years": dates["age_years"], "age_status": dates["age_status"],
            "acquisition_date": dates["acquisition_date"], "birth_date": dates["birth_date"],
            "dpf_scope": dpf_scope(float(dates["age_years"])),
            "coordinate_unit_declared": "cm", "physical_cap_distance_verified": False,
            "geometry_interpretation": "coordinate-derived source-detector distance; physical cap placement remains to be confirmed",
            "distance_min_cm": float(np.min(distance)), "distance_median_cm": float(np.median(distance)), "distance_max_cm": float(np.max(distance)),
            "short_separation_channels_present": False,
            "short_separation_policy": "none available; no short-channel regression performed",
            "channel_gate_count": int(channel_gate.sum()),
            "analysis_sample_percent_after_stage_and_raw_gate": float(analysis_mask[:, channel_gate].mean() * 100),
        })
        for low, high in FILTER_GRID:
            variant_filter = f"f{low:g}-{high:g}"
            od_stack = np.stack([od730, od850], axis=1)
            filtered_od, usable, filter_audit = filter_masked(od_stack.reshape(len(fnirs.time), -1), np.repeat(analysis_mask, 2, axis=1), fnirs.sample_rate_hz, low, high)
            filtered_od = filtered_od.reshape(len(fnirs.time), 2, len(fnirs.channels))
            usable_w = usable.reshape(len(fnirs.time), 2, len(fnirs.channels))
            # Both wavelengths must survive the same mask for an OD/MBLL fair comparison.
            usable_pair = np.all(usable_w, axis=1)
            raw_proxy = (filtered_od[:, 1, :] - filtered_od[:, 0, :]) / np.sqrt(2.0)
            historical_stack = np.stack([robust_z_masked(od730, analysis_mask), robust_z_masked(od850, analysis_mask)], axis=1)
            filtered_hist, usable_hist, _ = filter_masked(historical_stack.reshape(len(fnirs.time), -1), np.repeat(analysis_mask, 2, axis=1), fnirs.sample_rate_hz, low, high)
            historical_proxy = (filtered_hist.reshape(len(fnirs.time), 2, len(fnirs.channels))[:, 1, :] - filtered_hist.reshape(len(fnirs.time), 2, len(fnirs.channels))[:, 0, :]) / np.sqrt(2.0)
            for signal_name, values, valid in (("raw_od_proxy", raw_proxy, usable_pair), ("historical_z_od_proxy", historical_proxy, np.all(usable_hist.reshape(len(fnirs.time), 2, len(fnirs.channels)), axis=1))):
                event_rows.extend(event_rows_for_signal(record, fnirs.time, values, valid, channel_gate, movement_labels, f"od_proxy__{variant_filter}", signal_name))
            for profile in DPF_PROFILES:
                pair = dpf_values(profile, float(dates["age_years"]))
                for distance_mode in DISTANCE_MODES:
                    run_distance = distance if distance_mode == "stage_geometry" else np.full_like(distance, 5.0)
                    concentrations, condition = unmix_natural_od(filtered_od, pair, run_distance)
                    variant = f"{profile}__{distance_mode}__{variant_filter}"
                    event_rows.extend(event_rows_for_signal(record, fnirs.time, concentrations[:, 0, :], usable_pair, channel_gate, movement_labels, variant, "mbll_hbo_uM"))
                    event_rows.extend(event_rows_for_signal(record, fnirs.time, concentrations[:, 1, :], usable_pair, channel_gate, movement_labels, variant, "mbll_hbr_uM"))
                    audit_rows.append({"subject": record["subject"], "session": record["session"], "source_relative_path": record["source_relative_path"],
                                       "variant": variant, "sample_rate_hz": fnirs.sample_rate_hz, "time_origin_seconds": float(fnirs.time[0]),
                                       "time_end_seconds": float(fnirs.time[-1]), "marker_count_deduplicated": len(movement_labels),
                                       "action_marker_count": sum(x["event_role"] == "action_block_start" for x in movement_labels),
                                       "age_years": dates["age_years"], "age_status": dates["age_status"], "acquisition_date": dates["acquisition_date"], "birth_date": dates["birth_date"],
                                       "dpf_730": pair[0], "dpf_850": pair[1], "dpf_scope": dpf_scope(float(dates["age_years"])),
                                       "distance_mode": distance_mode, "distance_median_cm": float(np.median(run_distance)), "distance_min_cm": float(np.min(run_distance)), "distance_max_cm": float(np.max(run_distance)),
                                       "mbll_design_condition": condition, "filter_low_hz": low, "filter_high_hz": high,
                                       "channel_gate_count": int(channel_gate.sum()), "analysis_sample_percent_after_stage_and_raw_gate": float(analysis_mask[:, channel_gate].mean() * 100),
                                       **mask_meta, **filter_audit})

    summary = aggregate_events(event_rows)
    paired = paired_rows(summary)
    paired_stats = paired_summary(paired)
    default_mbll = [r for r in summary if r["variant"] == DEFAULT_VARIANT and r["signal"] == "mbll_hbo_uM"]
    default_od = [r for r in summary if r["variant"] == "od_proxy__f0.01-0.2" and r["signal"] == "raw_od_proxy"]
    default_historical = [r for r in summary if r["variant"] == "od_proxy__f0.01-0.2" and r["signal"] == "historical_z_od_proxy"]
    mbll_index = require_unique(default_mbll, ("subject", "session", "task_name"), "default MBLL")
    od_index = require_unique(default_od, ("subject", "session", "task_name"), "default raw OD")
    historical_index = require_unique(default_historical, ("subject", "session", "task_name"), "default historical OD")
    comparison = []
    for key, mbll in sorted(mbll_index.items()):
        od_row, historic = od_index.get(key), historical_index.get(key)
        if od_row is None or historic is None:
            continue
        hbo, raw_od, historical = mbll["event_delta_mean"], od_row["event_delta_mean"], historic["event_delta_mean"]
        comparison.append({"subject": key[0], "session": key[1], "task_name": key[2], "mbll_hbo_uM": hbo, "raw_od_proxy": raw_od,
                           "historical_z_od_proxy": historical,
                           "same_direction_mbll_vs_raw_od": bool(np.sign(hbo) == np.sign(raw_od)) if np.isfinite(hbo) and np.isfinite(raw_od) and hbo != 0 and raw_od != 0 else "undefined",
                           "same_direction_mbll_vs_historical_proxy": bool(np.sign(hbo) == np.sign(historical)) if np.isfinite(hbo) and np.isfinite(historical) and hbo != 0 and historical != 0 else "undefined",
                           "comparison_interpretation": "descriptive same-window/channel/mask contrast; not evidence of neural specificity"})
    longitudinal_comparison = longitudinal_proxy_mbll_comparison(summary)
    head_knee_by_session, head_knee_paired = task_contrast_rows(summary)
    sensitivity_rows = sensitivity_stability(paired)
    exact_test_rows = exploratory_exact_tests(paired, head_knee_paired)
    eye_rows, eye_exclusions = eye_inventory()
    eeg_rows = eeg_eog_ecg_audit(records)
    write_csv(OUT / "movement_event_responses.csv", event_rows)
    write_csv(OUT / "movement_subject_task_summary.csv", summary)
    write_csv(OUT / "movement_run_audit.csv", audit_rows)
    write_csv(OUT / "movement_marker_audit.csv", marker_audit_rows)
    write_csv(OUT / "source_input_geometry_audit.csv", source_audit_rows)
    write_csv(OUT / "eeg_eog_ecg_missing_reference_qc.csv", eeg_rows)
    write_csv(OUT / "paired_subject_task.csv", paired)
    write_csv(OUT / "paired_summary.csv", paired_stats)
    write_csv(OUT / "sensitivity_stability_summary.csv", sensitivity_rows)
    write_csv(OUT / "od_mbll_same_window_comparison.csv", comparison)
    write_csv(OUT / "od_mbll_longitudinal_change_comparison.csv", longitudinal_comparison)
    write_csv(OUT / "head_minus_knee_subject_session.csv", head_knee_by_session)
    write_csv(OUT / "head_minus_knee_longitudinal.csv", head_knee_paired)
    write_csv(OUT / "exploratory_exact_signflip_tests.csv", exact_test_rows)
    write_csv(OUT / "repeated_record_exclusions.csv", duplicate_exclusions + eye_exclusions)
    write_csv(OUT / "eye_degraded_branch_inventory.csv", eye_rows)
    metadata = {
        "cohort": {"subjects": list(PAIRED7), "n": 7, "sessions": ["base_line", "sec_test"], "unit_of_analysis": "participant; event repetitions aggregated within participant/session/task"},
        "input": {"raw_root": str(RAW.relative_to(ROOT)), "stage_root": str(STAGE.relative_to(ROOT)), "stage_selection": "lexicographically first stage-approved source per subject/session/task; duplicates retained only in exclusion audit", "task_semantics": "hardware marker value 1 is synchronization/boundary only; task names require outer movement branch plus fixed protocol order", "eeg_eog_ecg": "EOG/ECG-named channels were user-confirmed unconnected. They are recorded as unavailable references, and no correction was attempted."},
        "mbll": {"od": "natural-log -ln(I/median valid raw intensity); no z-scoring before MBLL", "baseline": "per-channel median of actual stage-mask-valid raw intensity", "event_contrast": "mean(2 to 12 s after marker) minus mean(-5 to 0 s before marker)", "output_unit": "uM relative concentration change, conditional on extinction/DPF/distance assumptions", "extinction": {"source": "https://omlc.org/spectra/hemoglobin/summary.html", "values_decadic_cm_inverse_M_inverse": EXTINCTION_E_DEC, "values_natural_cm_inverse_M_inverse": EXTINCTION_E_NATURAL, "log_convention": "natural OD solved with ln(10)*E_decadic exactly once"}, "dpf": {"source": "Scholkmann F, Wolf M. J Biomed Opt. 2013;18(10):105004. doi:10.1117/1.JBO.18.10.105004", "formula": "223.3 + 0.05624*age^0.8493 - 5.723e-7*lambda^3 + 0.001245*lambda^2 - 0.9025*lambda", "validation_warning": "source figure displays 0-50 years and 690-832 nm; 850 nm and out-of-range ages are labeled extrapolations; formula is for frontal head and raw metadata does not verify site"}},
        "quality": {"channel_gate": "stage kept_channels", "sample_gate": "stage analysis_sample_mask intersected with raw positive/finite intensity", "filter": "each valid contiguous run independently; invalid/motion gaps not interpolated; 1 s core excluded at each valid-run edge", "event_coverage": f"both baseline and response must have >= {MIN_WINDOW_COVERAGE:.0%} usable samples per channel", "movement_scope": "only the three completed fixed actions (head, contralateral knee, rear waist); anomalous grip extension audited but excluded", "eye": "separate known-IR-degraded inventory; no movement labels, MBLL, or IR-removal claim; EOG/ECG unconnected, no correction"},
        "grid": {"dpf_profiles": list(DPF_PROFILES), "distance_modes": list(DISTANCE_MODES), "filter_hz": [list(x) for x in FILTER_GRID], "default_variant": DEFAULT_VARIANT},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Reviewed relative MBLL branch\n\n"
        "This output is a quality-gated sensitivity analysis, not an absolute hemoglobin measurement or a neural-specificity test. "
        "It uses exactly the stage_20260904_eye_fnirs_imu_verified paired-seven movement cohort; repeated files are not pooled. "
        "`movement_subject_task_summary.csv` is the analysis table: five within-task events are aggregated before any between-session calculation. Only the three completed fixed actions are analyzed; later grip-extension markers are retained only in `movement_marker_audit.csv`. "
        "`od_mbll_same_window_comparison.csv` is a cross-sectional response-direction table; its direction columns are not longitudinal claims. `od_mbll_longitudinal_change_comparison.csv` is the separate participant-level baseline-to-sec_test direction comparison. `sensitivity_stability_summary.csv` reports each MBLL choice against the default branch. `head_minus_knee_*.csv` contains the requested task contrast and is explicitly separate from within-task responses. "
        "The eye branch is intentionally inventory-only because its marker semantics are not movement labels and infrared interference remains unresolved.\n",
        encoding="utf-8")
    test_path = ROOT / "tests" / "test_mbll_relative_sensitivity_reviewed.py"
    smoke_path = ROOT / "tests" / "run_mbll_relative_sensitivity_reviewed_smoke.py"
    code_snapshot = OUT / "reproducible_sources"
    code_snapshot.mkdir(exist_ok=True)
    # Snapshots are records of the exact code executed. The manifest identifies
    # the canonical runnable root paths, so a reader is not misled into running
    # an isolated copy without its data/preprocessing dependencies.
    (code_snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if test_path.exists():
        (code_snapshot / test_path.name).write_bytes(test_path.read_bytes())
    if smoke_path.exists():
        (code_snapshot / smoke_path.name).write_bytes(smoke_path.read_bytes())
    manifest = {
        "canonical_analysis_script": str(Path(__file__).resolve()),
        "canonical_pytest_file": str(test_path.resolve()),
        "canonical_smoke_runner": str(smoke_path.resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "test_file_sha256": sha256_file(test_path) if test_path.exists() else "missing",
        "smoke_runner_sha256": sha256_file(smoke_path) if smoke_path.exists() else "missing",
        "reproduce_analysis_command": "python -B mbll_relative_sensitivity_reviewed_20260908.py",
        "reproduce_smoke_tests_command": "python -B tests\\run_mbll_relative_sensitivity_reviewed_smoke.py",
        "optional_pytest_command": "python -B -m pytest -q tests\\test_mbll_relative_sensitivity_reviewed.py",
        "snapshot_note": "Snapshots preserve the executed source; run from the project root so raw/stage paths resolve.",
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_figures(paired, comparison)
    generate_teacher_assets(paired, longitudinal_comparison, sensitivity_rows, paired_stats, source_audit_rows, marker_audit_rows, eye_rows, eeg_rows, exact_test_rows)
    import scipy
    import matplotlib
    unique_checks = {
        "movement_subject_task_summary": len(require_unique(summary, ("subject", "session", "task_name", "variant", "signal"), "manifest summary")) == len(summary),
        "paired_subject_task": len(require_unique(paired, ("subject", "task_name", "variant", "signal"), "manifest paired")) == len(paired),
        "od_mbll_longitudinal": len(require_unique(longitudinal_comparison, ("subject", "task_name"), "manifest longitudinal")) == len(longitudinal_comparison),
        "sensitivity_stability": len(require_unique(sensitivity_rows, ("task_name", "variant"), "manifest sensitivity")) == len(sensitivity_rows),
    }
    run_log = {
        "analysis_status": "completed",
        "executed_at_local": datetime.now().astimezone().isoformat(),
        "argv": [sys.executable, *sys.argv],
        "analysis_command_copyable": "python -B mbll_relative_sensitivity_reviewed_20260908.py",
        "smoke_test_argv": [sys.executable, "-B", "tests\\run_mbll_relative_sensitivity_reviewed_smoke.py"],
        "smoke_test_command_copyable": "python -B tests\\run_mbll_relative_sensitivity_reviewed_smoke.py",
        "optional_pytest_command_copyable": "python -B -m pytest -q tests\\test_mbll_relative_sensitivity_reviewed.py",
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pytest": RUN_CONTEXT["pytest_version"]},
        "pytest_evidence": RUN_CONTEXT["pytest_result"],
        "smoke_test_evidence": RUN_CONTEXT["smoke_result"],
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "output": str(OUT), "movement_records": len(records), "event_rows": len(event_rows),
        "subject_task_rows": len(summary), "paired_rows": len(paired),
        "longitudinal_proxy_comparison_rows": len(longitudinal_comparison),
        "sensitivity_rows": len(sensitivity_rows), "eye_inventory_rows": len(eye_rows),
        "main_csv_rows": {"movement_subject_task_summary": len(summary), "paired_subject_task": len(paired), "longitudinal_proxy_comparison": len(longitudinal_comparison), "sensitivity_stability": len(sensitivity_rows), "exact_signflip": len(exact_test_rows)},
        "unique_key_checks": unique_checks,
        "implementation_notes": [
            "No EOG/ECG correction: channels were user-confirmed unconnected.",
            "Eye files are inventory-only EYE_IR_DEGRADED_MASKED records and are not movement MBLL data.",
            "No invalid or motion-marked sample is restored to event statistics after filtering.",
        ],
    }
    (OUT / "run_log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "run_manifest.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "run_log.txt").write_text("\n".join([
        f"status={run_log['analysis_status']}", f"executed_at_local={run_log['executed_at_local']}", "argv=" + json.dumps(run_log["argv"], ensure_ascii=False),
        f"analysis_command_copyable={run_log['analysis_command_copyable']}", f"smoke_test_command_copyable={run_log['smoke_test_command_copyable']}",
        f"python={run_log['versions']['python'].splitlines()[0]}", f"numpy={run_log['versions']['numpy']}", f"scipy={run_log['versions']['scipy']}",
        f"matplotlib={run_log['versions']['matplotlib']}", f"pytest={run_log['versions']['pytest']}", f"pytest_evidence={run_log['pytest_evidence']}", f"smoke_test_evidence={run_log['smoke_test_evidence']}",
        f"script_sha256={run_log['analysis_script_sha256']}", f"movement_records={len(records)}", f"event_rows={len(event_rows)}", f"subject_task_rows={len(summary)}", f"paired_rows={len(paired)}",
        "unique_key_checks=" + json.dumps(unique_checks, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "records": len(records), "event_rows": len(event_rows), "subject_task_rows": len(summary), "paired_rows": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--pytest-result", default="not_recorded", help="external synthetic-test result recorded in the run manifest")
    parser.add_argument("--pytest-version", default="not_recorded", help="pytest version used for the external synthetic tests")
    parser.add_argument("--smoke-result", default="not_recorded", help="dependency-free same-interpreter smoke-test result recorded in the run manifest")
    args = parser.parse_args()
    if args.output.resolve() != OUT.resolve():
        OUT = args.output.resolve()
    RUN_CONTEXT = {"pytest_result": str(args.pytest_result), "pytest_version": str(args.pytest_version), "smoke_result": str(args.smoke_result)}
    run()
