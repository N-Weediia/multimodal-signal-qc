"""Quality-gated motor EEG sensitivity branch for the paired-seven cohort.

The existing EEG feature pipeline is left untouched.  This branch reads the
segment-level QC evidence, reopens the largest motor-task BDF for each paired
subject/session, drops segments with broad artefact, and excludes local bad
channels only within the affected segment.  It reports block/task summaries
and compares subject-level baseline -> sec_test deltas with the legacy
continuous-record EEG features.
"""
from __future__ import annotations

import json
import hashlib
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import main as eeg_main
from scripts.segment_eeg_quality import filter_eeg, movement_recording
try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, RAW_DATA_ROOT, SEGMENT_EEG_ROOT, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, RAW_DATA_ROOT, SEGMENT_EEG_ROOT, phase_root

QUALITY_PATH = SEGMENT_EEG_ROOT / "segment_quality.csv"
LEGACY_PATH = CANONICAL_ROOT / "motor_multimodal_features.csv"
OUT = phase_root("17_eeg_quality")
DATA_ROOT = RAW_DATA_ROOT
SUBJECTS = list(PAIRED_SUBJECTS)
SESSIONS = ["base_line", "sec_test"]
TASKS = ["摸头", "摸膝", "摸腰"]
ROIS = {
    "frontal": ["Fz", "F3", "F4", "FCz", "FC1", "FC2"],
    "left_sensorimotor": ["C1", "C3", "CP1", "CP3"],
    "right_sensorimotor": ["C2", "C4", "CP2", "CP4"],
    "parietal": ["Pz", "P3", "P4", "POz"],
}
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
BROAD_EXCLUDE = 0.35
LOCAL_BAD_EPOCH_FRACTION = 0.20
ARTIFACT_PTP_UV = 200.0
EPOCH_SECONDS = 2.0

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def load_quality() -> pd.DataFrame:
    if not QUALITY_PATH.exists():
        raise FileNotFoundError(QUALITY_PATH)
    q = pd.read_csv(QUALITY_PATH)
    q["subject"] = q["subject"].astype(str)
    q = q[q.subject.isin(SUBJECTS) & q.session.isin(SESSIONS) & q.task.isin(TASKS)].copy()
    for col in ["block", "start_seconds", "end_seconds", "broad_bad_epoch_fraction", "bad_channel_fraction"]:
        q[col] = pd.to_numeric(q[col], errors="coerce")
    if "visit_protocol_complete" not in q.columns:
        raise ValueError("segment-quality input lacks visit_protocol_complete; rerun the paired7 segment QC branch")
    q["visit_protocol_complete"] = q["visit_protocol_complete"].astype(str).str.lower().eq("true")
    return q


def scalp_data(path: Path) -> tuple[np.ndarray, list[str], float]:
    header, physical, _ = eeg_main.read_bdf(path, None)
    wanted = {eeg_main.normalize_label(x) for x in eeg_main.FULL_EEG_CHANNELS}
    indices = [i for i, label in enumerate(header.labels) if eeg_main.normalize_label(str(label)) in wanted]
    if not indices:
        raise ValueError(f"no configured scalp channels in {path}")
    labels = [str(header.labels[i]) for i in indices]
    return physical[indices].astype(np.float32), labels, float(header.sample_rate)


def segment_quality_mask(segment: np.ndarray, fs: float) -> tuple[np.ndarray, float, float]:
    epoch_samples = max(1, int(round(fs * EPOCH_SECONDS)))
    count = segment.shape[1] // epoch_samples
    if count < 1:
        return np.ones(segment.shape[0], dtype=bool), np.nan, np.nan
    usable = segment[:, :count * epoch_samples].reshape(segment.shape[0], count, epoch_samples)
    ptp = np.ptp(usable, axis=2)
    bad_epochs = ptp > ARTIFACT_PTP_UV
    local_bad = np.mean(bad_epochs, axis=1) >= LOCAL_BAD_EPOCH_FRACTION
    broad = float(np.mean(np.mean(bad_epochs, axis=0) >= 0.20))
    return ~local_bad, broad, float(np.mean(local_bad))


def robust_z(data: np.ndarray) -> np.ndarray:
    med = np.nanmedian(data, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(data - med), axis=1, keepdims=True) * 1.4826
    return np.clip((data - med) / np.maximum(mad, 1e-9), -20.0, 20.0).astype(np.float32)


def band_power(values: np.ndarray, fs: float, lo: float, hi: float) -> float:
    if values.size < 8:
        return np.nan
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(int(fs * 4), values.size), detrend="linear")
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else np.nan


def segment_features(clean: np.ndarray, labels: list[str], fs: float, keep: np.ndarray) -> dict[str, float]:
    label_map = {eeg_main.normalize_label(label): i for i, label in enumerate(labels)}
    kept = np.flatnonzero(keep)
    if kept.size < 2:
        return {"n_channels_used": int(kept.size), "eeg_global_alpha_beta_ratio": np.nan, "eeg_theta_frontal": np.nan, "eeg_frontal_alpha": np.nan, "eeg_frontal_beta": np.nan}
    series = clean[kept]
    result: dict[str, float] = {"n_channels_used": int(kept.size)}
    all_alpha = np.nanmean([band_power(row, fs, *BANDS["alpha"]) for row in series])
    all_beta = np.nanmean([band_power(row, fs, *BANDS["beta"]) for row in series])
    result["eeg_global_alpha_beta_ratio"] = float(all_alpha / max(all_beta, 1e-12)) if np.isfinite(all_alpha) and np.isfinite(all_beta) else np.nan
    for roi, wanted in ROIS.items():
        idx = [label_map[eeg_main.normalize_label(x)] for x in wanted if eeg_main.normalize_label(x) in label_map and keep[label_map[eeg_main.normalize_label(x)]]]
        if not idx:
            result[f"eeg_{roi}_theta"] = np.nan
            result[f"eeg_{roi}_alpha"] = np.nan
            result[f"eeg_{roi}_beta"] = np.nan
            continue
        roi_series = np.nanmean(clean[idx], axis=0)
        for band, (lo, hi) in BANDS.items():
            result[f"eeg_{roi}_{band}"] = band_power(roi_series, fs, lo, hi)
    frontal_idx = [label_map[eeg_main.normalize_label(x)] for x in ROIS["frontal"] if eeg_main.normalize_label(x) in label_map and keep[label_map[eeg_main.normalize_label(x)]]]
    if frontal_idx:
        frontal = np.nanmean(clean[frontal_idx], axis=0)
        theta = signal.sosfiltfilt(signal.butter(4, (4.0, 8.0), btype="bandpass", fs=fs, output="sos"), frontal)
        result["eeg_theta_env_frontal_mean"] = float(np.mean(np.abs(signal.hilbert(theta))))
    else:
        result["eeg_theta_env_frontal_mean"] = np.nan
    return result


def extract(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    block_rows: list[dict[str, object]] = []
    coverage = (
        q.groupby(["subject", "session"], sort=True)
        .agg(
            visit_protocol_complete=("visit_protocol_complete", "all"),
            visit_status=("visit_status", lambda x: ";".join(sorted(set(map(str, x))))),
            event_alignment=("event_alignment", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .set_index(["subject", "session"])
    )
    for (subject, session), qg in q.groupby(["subject", "session"], sort=True):
        if not bool(coverage.loc[(subject, session), "visit_protocol_complete"]):
            # A partial EEG recording cannot become a complete visit merely
            # because its earliest event windows passed quality screening.
            continue
        path = movement_recording(DATA_ROOT, subject, session)
        if path is None:
            raise FileNotFoundError(f"motor BDF missing for {subject} {session}")
        raw, labels, fs = scalp_data(path)
        filtered = filter_eeg(raw, fs)
        clean = robust_z(filtered)
        quality_filtered = filtered
        del raw
        for (task, block), group in qg.groupby(["task", "block"], sort=True):
            roles: dict[str, dict[str, object]] = {}
            for _, row in group.iterrows():
                role = str(row.segment_role)
                left = max(0, int(round(float(row.start_seconds) * fs)))
                right = min(clean.shape[1], int(round(float(row.end_seconds) * fs)))
                quality_segment = quality_filtered[:, left:right]
                segment = clean[:, left:right]
                keep, broad, local_fraction = segment_quality_mask(quality_segment, fs)
                broad_source = float(row.broad_bad_epoch_fraction)
                broad_gate = bool(np.isfinite(broad_source) and broad_source > BROAD_EXCLUDE)
                # The source table is authoritative for broad gating; the
                # recomputed value is retained as an audit check.
                roles[role] = {"source_broad": broad_source, "recomputed_broad": broad, "source_local": float(row.bad_channel_fraction), "recomputed_local": local_fraction, "broad_excluded": broad_gate, "keep": keep, "event_source": str(row.event_source), "event_alignment": str(row.event_alignment), "start": float(row.start_seconds), "end": float(row.end_seconds)}
                # Extract local-channel-gated features for every segment. The
                # strict qg estimate below masks broad segments, while this
                # retained raw estimate enables explicit threshold sensitivity.
                roles[role]["features"] = segment_features(segment, labels, fs, keep)
            pre, execution = roles.get("pre_action"), roles.get("execution")
            event_delta_raw = np.nan
            if pre and execution:
                a = pre.get("features", {}).get("eeg_theta_env_frontal_mean", np.nan)
                b = execution.get("features", {}).get("eeg_theta_env_frontal_mean", np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    event_delta_raw = float(b - a)
            event_delta = event_delta_raw if pre and execution and not pre["broad_excluded"] and not execution["broad_excluded"] else np.nan
            execution_features = execution.get("features", {}) if execution else {}
            execution_qg_features = execution_features if execution and not execution["broad_excluded"] else {}
            block_rows.append({"subject": subject, "session": session, "task": task, "block": int(block), "recording": str(path), "recording_bytes": int(path.stat().st_size), "event_source": execution.get("event_source", "") if execution else "", "event_alignment": execution.get("event_alignment", "") if execution else "", "pre_source_broad_bad_epoch_fraction": pre.get("source_broad", np.nan) if pre else np.nan, "execution_source_broad_bad_epoch_fraction": execution.get("source_broad", np.nan) if execution else np.nan, "pre_broad_excluded": bool(pre["broad_excluded"]) if pre else True, "execution_broad_excluded": bool(execution["broad_excluded"]) if execution else True, "pre_local_bad_channel_fraction": pre.get("source_local", np.nan) if pre else np.nan, "execution_local_bad_channel_fraction": execution.get("source_local", np.nan) if execution else np.nan, "pre_recomputed_local_bad_channel_fraction": pre.get("recomputed_local", np.nan) if pre else np.nan, "execution_recomputed_local_bad_channel_fraction": execution.get("recomputed_local", np.nan) if execution else np.nan, "pre_recomputed_broad_bad_epoch_fraction": pre.get("recomputed_broad", np.nan) if pre else np.nan, "execution_recomputed_broad_bad_epoch_fraction": execution.get("recomputed_broad", np.nan) if execution else np.nan, "pre_channels_used": pre.get("features", {}).get("n_channels_used", 0) if pre else 0, "execution_channels_used": execution_features.get("n_channels_used", 0), "eeg_event_theta_frontal_delta_raw": event_delta_raw, "eeg_global_alpha_beta_ratio_execution_raw": execution_features.get("eeg_global_alpha_beta_ratio", np.nan), "eeg_event_theta_frontal_delta_qg": event_delta, "eeg_global_alpha_beta_ratio_execution_qg": execution_qg_features.get("eeg_global_alpha_beta_ratio", np.nan), "eeg_frontal_theta_execution_qg": execution_qg_features.get("eeg_frontal_theta", np.nan), "eeg_frontal_alpha_execution_qg": execution_qg_features.get("eeg_frontal_alpha", np.nan), "eeg_frontal_beta_execution_qg": execution_qg_features.get("eeg_frontal_beta", np.nan)})
        del quality_filtered, clean
    blocks = pd.DataFrame(block_rows)
    task_rows = []
    for (subject, session, task), g in blocks.groupby(["subject", "session", "task"], sort=True):
        task_rows.append({"subject": subject, "session": session, "task": task, "recording": g.recording.iloc[0], "recording_bytes": int(g.recording_bytes.iloc[0]), "event_source": ";".join(sorted(set(g.event_source))), "event_alignment": ";".join(sorted(set(g.event_alignment))), "blocks_total": int(len(g)), "blocks_execution_broad_excluded": int(g.execution_broad_excluded.sum()), "blocks_execution_valid": int((~g.execution_broad_excluded).sum()), "blocks_event_metric_valid": int(g.eeg_event_theta_frontal_delta_qg.notna().sum()), "blocks_pre_broad_excluded": int(g.pre_broad_excluded.sum()), "mean_execution_local_bad_channel_fraction": float(g.execution_local_bad_channel_fraction.mean()), "mean_execution_local_bad_channel_fraction_recomputed": float(g.execution_recomputed_local_bad_channel_fraction.mean()), "mean_execution_channels_used": float(g.execution_channels_used.mean()), "eeg_event_theta_frontal_delta_qg": float(g.eeg_event_theta_frontal_delta_qg.mean()), "eeg_global_alpha_beta_ratio_execution_qg": float(g.eeg_global_alpha_beta_ratio_execution_qg.mean())})
    tasks = pd.DataFrame(task_rows)
    subject_rows = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            status = coverage.loc[(subject, session)]
            g = tasks[(tasks.subject == subject) & (tasks.session == session)]
            if not bool(status.visit_protocol_complete):
                subject_rows.append({"subject": subject, "session": session, "event_alignment": status.event_alignment, "visit_protocol_complete": False, "visit_status": status.visit_status, "eeg_event_theta_frontal_delta_qg": np.nan, "eeg_global_alpha_beta_ratio_execution_qg": np.nan, "blocks_total": 0, "blocks_execution_broad_excluded": 0, "blocks_execution_valid": 0, "blocks_event_metric_valid": 0, "mean_execution_local_bad_channel_fraction": np.nan, "mean_execution_channels_used": np.nan})
                continue
            subject_rows.append({"subject": subject, "session": session, "event_alignment": ";".join(sorted(set(g.event_alignment))), "visit_protocol_complete": True, "visit_status": status.visit_status, "eeg_event_theta_frontal_delta_qg": float(g.eeg_event_theta_frontal_delta_qg.mean()), "eeg_global_alpha_beta_ratio_execution_qg": float(g.eeg_global_alpha_beta_ratio_execution_qg.mean()), "blocks_total": int(g.blocks_total.sum()), "blocks_execution_broad_excluded": int(g.blocks_execution_broad_excluded.sum()), "blocks_execution_valid": int(g.blocks_execution_valid.sum()), "blocks_event_metric_valid": int(g.blocks_event_metric_valid.sum()), "mean_execution_local_bad_channel_fraction": float(np.average(g.mean_execution_local_bad_channel_fraction, weights=g.blocks_total)), "mean_execution_channels_used": float(np.average(g.mean_execution_channels_used, weights=g.blocks_total))})
    return blocks, tasks, pd.DataFrame(subject_rows)


def compare(subject: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_PATH)
    legacy["subject"] = legacy["subject"].astype(str)
    legacy = legacy[legacy.subject.isin(SUBJECTS) & legacy.session.isin(SESSIONS)]
    rows = []
    for sid in SUBJECTS:
        old_pair = legacy[legacy.subject.eq(sid)].set_index("session")
        new_pair = subject[subject.subject.eq(sid)].set_index("session")
        old_b = old_pair.loc["base_line"] if "base_line" in old_pair.index else None
        old_f = old_pair.loc["sec_test"] if "sec_test" in old_pair.index else None
        new_b = new_pair.loc["base_line"] if "base_line" in new_pair.index else None
        new_f = new_pair.loc["sec_test"] if "sec_test" in new_pair.index else None
        old_event = float(old_f.eeg_event_theta_frontal_delta) - float(old_b.eeg_event_theta_frontal_delta) if old_b is not None and old_f is not None else np.nan
        old_ratio = float(old_f.eeg_global_alpha_beta_ratio) - float(old_b.eeg_global_alpha_beta_ratio) if old_b is not None and old_f is not None else np.nan
        new_event = float(new_f.eeg_event_theta_frontal_delta_qg) - float(new_b.eeg_event_theta_frontal_delta_qg) if new_b is not None and new_f is not None else np.nan
        new_ratio = float(new_f.eeg_global_alpha_beta_ratio_execution_qg) - float(new_b.eeg_global_alpha_beta_ratio_execution_qg) if new_b is not None and new_f is not None else np.nan
        baseline_complete = bool(new_b.visit_protocol_complete) if new_b is not None else False
        followup_complete = bool(new_f.visit_protocol_complete) if new_f is not None else False
        rows.append({"subject": sid, "old_eeg_event_theta_frontal_delta_change": old_event, "new_qg_eeg_event_theta_frontal_delta_change": new_event, "new_minus_old_event_change": new_event - old_event, "old_eeg_global_alpha_beta_ratio_change": old_ratio, "new_qg_eeg_global_alpha_beta_ratio_execution_change": new_ratio, "new_minus_old_ratio_change": new_ratio - old_ratio, "baseline_event_alignment": new_b.event_alignment if new_b is not None else "missing", "followup_event_alignment": new_f.event_alignment if new_f is not None else "missing", "baseline_visit_status": new_b.visit_status if new_b is not None else "missing", "followup_visit_status": new_f.visit_status if new_f is not None else "missing", "baseline_visit_protocol_complete": baseline_complete, "followup_visit_protocol_complete": followup_complete, "baseline_qg_broad_excluded_execution_blocks": int(new_b.blocks_execution_broad_excluded) if new_b is not None else 0, "followup_qg_broad_excluded_execution_blocks": int(new_f.blocks_execution_broad_excluded) if new_f is not None else 0, "baseline_qg_execution_valid_blocks": int(new_b.blocks_execution_valid) if new_b is not None else 0, "followup_qg_execution_valid_blocks": int(new_f.blocks_execution_valid) if new_f is not None else 0, "event_delta_comparison_valid": bool(baseline_complete and followup_complete and np.isfinite(old_event) and np.isfinite(new_event)), "ratio_comparison_valid": bool(baseline_complete and followup_complete and np.isfinite(old_ratio) and np.isfinite(new_ratio))})
    return pd.DataFrame(rows)


def threshold_sensitivity(blocks: pd.DataFrame) -> pd.DataFrame:
    """Re-summarize retained local-gated blocks at broad thresholds."""
    rows: list[dict[str, object]] = []
    for threshold in (0.35, 0.50, 0.75):
        for (subject, session), g in blocks.groupby(["subject", "session"], sort=True):
            valid = g[(g.pre_source_broad_bad_epoch_fraction <= threshold) & (g.execution_source_broad_bad_epoch_fraction <= threshold)]
            rows.append({"threshold": threshold, "subject": subject, "session": session, "blocks_total": int(len(g)), "blocks_valid_pre_and_execution": int(len(valid)), "blocks_execution_broad_excluded": int((g.execution_source_broad_bad_epoch_fraction > threshold).sum()), "mean_execution_local_bad_channel_fraction": float(valid.execution_recomputed_local_bad_channel_fraction.mean()) if len(valid) else np.nan, "mean_execution_channels_used": float(valid.execution_channels_used.mean()) if len(valid) else np.nan, "eeg_event_theta_frontal_delta": float(valid.eeg_event_theta_frontal_delta_raw.mean()) if len(valid) else np.nan, "eeg_global_alpha_beta_ratio_execution": float(valid.eeg_global_alpha_beta_ratio_execution_raw.mean()) if len(valid) else np.nan, "event_alignment": ";".join(sorted(set(g.event_alignment))), "broad_gate_source": "segment_quality source broad_bad_epoch_fraction"})
    out = pd.DataFrame(rows)
    deltas: list[dict[str, object]] = []
    for threshold, g in out.groupby("threshold"):
        for subject in SUBJECTS:
            pair = g[g.subject.eq(subject)].set_index("session")
            if not {"base_line", "sec_test"}.issubset(pair.index):
                continue
            deltas.append({"threshold": threshold, "subject": subject, "baseline_valid_blocks": int(pair.loc["base_line", "blocks_valid_pre_and_execution"]), "followup_valid_blocks": int(pair.loc["sec_test", "blocks_valid_pre_and_execution"]), "event_delta_change": float(pair.loc["sec_test", "eeg_event_theta_frontal_delta"] - pair.loc["base_line", "eeg_event_theta_frontal_delta"]) if pd.notna(pair.loc["base_line", "eeg_event_theta_frontal_delta"]) and pd.notna(pair.loc["sec_test", "eeg_event_theta_frontal_delta"]) else np.nan, "ratio_delta_change": float(pair.loc["sec_test", "eeg_global_alpha_beta_ratio_execution"] - pair.loc["base_line", "eeg_global_alpha_beta_ratio_execution"]) if pd.notna(pair.loc["base_line", "eeg_global_alpha_beta_ratio_execution"]) and pd.notna(pair.loc["sec_test", "eeg_global_alpha_beta_ratio_execution"]) else np.nan, "baseline_event_alignment": pair.loc["base_line", "event_alignment"], "followup_event_alignment": pair.loc["sec_test", "event_alignment"]})
    return pd.DataFrame(deltas)


def plot(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(len(comparison))
    w = 0.36
    event_n = int(comparison.event_delta_comparison_valid.sum()) if "event_delta_comparison_valid" in comparison else int((np.isfinite(comparison.old_eeg_event_theta_frontal_delta_change) & np.isfinite(comparison.new_qg_eeg_event_theta_frontal_delta_change)).sum())
    ratio_n = int(comparison.ratio_comparison_valid.sum()) if "ratio_comparison_valid" in comparison else int((np.isfinite(comparison.old_eeg_global_alpha_beta_ratio_change) & np.isfinite(comparison.new_qg_eeg_global_alpha_beta_ratio_execution_change)).sum())
    ax = axes[0]
    ax.bar(x - w / 2, comparison.old_eeg_event_theta_frontal_delta_change, w, color="#8c8c8c", label="legacy continuous")
    ax.bar(x + w / 2, comparison.new_qg_eeg_event_theta_frontal_delta_change, w, color="#2166ac", label="segment-gated")
    ax.axhline(0, color="0.3", lw=0.7); ax.set_xticks(x, [s[-4:] for s in comparison.subject]); ax.set_ylabel("theta frontal event delta change"); ax.set_title(f"EEG event response sensitivity | cohort n={len(SUBJECTS)}, paired n={event_n}"); ax.legend(fontsize=7)
    ax = axes[1]
    ax.scatter(comparison.old_eeg_global_alpha_beta_ratio_change, comparison.new_qg_eeg_global_alpha_beta_ratio_execution_change, s=58, color="#d95f02", edgecolor="white", linewidth=0.7)
    for _, row in comparison.iterrows():
        ax.annotate(row.subject[-4:], (row.old_eeg_global_alpha_beta_ratio_change, row.new_qg_eeg_global_alpha_beta_ratio_execution_change), xytext=(4, 3), textcoords="offset points", fontsize=7)
    lim = np.nanmax(np.abs(np.r_[comparison.old_eeg_global_alpha_beta_ratio_change, comparison.new_qg_eeg_global_alpha_beta_ratio_execution_change])) or 1.0
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=0.8); ax.set_xlabel("legacy ratio delta"); ax.set_ylabel("gated execution ratio delta"); ax.set_title(f"Band-ratio delta sensitivity | cohort n={len(SUBJECTS)}, paired n={ratio_n}"); fig.suptitle("Motor EEG quality-gated sensitivity | base_line -> sec_test", fontsize=12)
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"eeg_quality_sensitivity.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_threshold_sensitivity(threshold: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    summary = threshold.groupby("threshold")["followup_valid_blocks"].mean().reindex([0.35, 0.50, 0.75])
    baseline = threshold.groupby("threshold")["baseline_valid_blocks"].mean().reindex([0.35, 0.50, 0.75])
    x = np.arange(3); w = 0.35
    ax.bar(x - w / 2, baseline, w, color="#8c8c8c", label="baseline valid blocks")
    ax.bar(x + w / 2, summary, w, color="#2166ac", label="sec_test valid blocks")
    ax.set_xticks(x, ["0.35 strict", "0.50", "0.75 permissive"])
    ax.set_ylim(0, 15.8); ax.set_ylabel("mean valid pre + execution blocks (of 15)")
    ax.set_title(f"Broad-artifact threshold sensitivity | n={len(SUBJECTS)}")
    ax.legend(fontsize=7)
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"eeg_broad_threshold_sensitivity.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(blocks: pd.DataFrame, tasks: pd.DataFrame, comparison: pd.DataFrame, threshold: pd.DataFrame) -> None:
    align_counts = blocks.groupby("event_alignment").size().to_dict()
    visit_status_counts = pd.concat(
        [comparison["baseline_visit_status"], comparison["followup_visit_status"]]
    ).value_counts().to_dict()
    complete_visits = int(
        comparison[["baseline_visit_protocol_complete", "followup_visit_protocol_complete"]].to_numpy(dtype=bool).sum()
    )
    finite_event = int(comparison.event_delta_comparison_valid.sum())
    finite_ratio = int(comparison.ratio_comparison_valid.sum())
    event_pairs = comparison[comparison.event_delta_comparison_valid]
    ratio_pairs = comparison[comparison.ratio_comparison_valid]
    event_same_sign = int((np.sign(event_pairs.old_eeg_event_theta_frontal_delta_change) == np.sign(event_pairs.new_qg_eeg_event_theta_frontal_delta_change)).sum())
    ratio_same_sign = int((np.sign(ratio_pairs.old_eeg_global_alpha_beta_ratio_change) == np.sign(ratio_pairs.new_qg_eeg_global_alpha_beta_ratio_execution_change)).sum())
    strict = threshold[threshold.threshold.eq(0.35)]
    permissive = threshold[threshold.threshold.eq(0.50)]
    cohort = ", ".join(SUBJECTS)
    total = len(SUBJECTS)
    lines = ["# Phase 17 motor EEG quality-gated sensitivity", "", f"Fixed cohort: {cohort} (n={total}). Independent unit is the participant; baseline -> sec_test is approximately two weeks under the fixed combination acupuncture/electroacupuncture protocol.", "", "## Processing", "", "The largest non-evt BDF in each subject/session motor folder is the formal entry. Each task block uses pre_action, execution and post_action_rest windows from the paired-seven segment-quality table. A segment is broad-gated when source `broad_bad_epoch_fraction > 0.35`; local bad channels are identified at the same 2-s PTP > 200 uV and >=20% bad-epoch rule, then excluded only within that segment. Broad gating and local-channel exclusion are reported separately.", "", f"Complete visits retained for segment-derived metrics: {complete_visits}/{2 * total}; visit-status counts: {visit_status_counts}. Event alignment counts across retained block rows: {align_counts}. `observed` denotes a complete EEG companion evt.bdf series; `inferred_from_fnirs` denotes the instructed complete fNIRS timing fallback when EEG markers are absent or incomplete. A truncated EEG recording remains an incomplete visit even if its early windows pass QC.", "", "## Comparability", "", "The legacy columns are continuous-record features (`eeg_event_theta_frontal_delta` and `eeg_global_alpha_beta_ratio`) from the current canonical motor feature table. The gated event metric uses execution minus pre-action frontal theta-envelope means; the gated ratio is computed in retained execution segments and averaged by task then subject/session. They are sensitivity counterparts, not numerically identical estimators.", "", f"Finite old/new paired subject comparisons: event-theta {finite_event}/{total}; alpha-beta ratio {finite_ratio}/{total}. Direction agreement among finite pairs: event-theta {event_same_sign}/{finite_event}; alpha-beta ratio {ratio_same_sign}/{finite_ratio}. Missing pairs reflect incomplete visits, quality-gated or legacy missingness, not imputation.", "", "## Broad threshold sensitivity", "", f"At strict 0.35, mean valid pre+execution blocks per complete session are baseline={strict.baseline_valid_blocks.mean():.1f}, sec_test={strict.followup_valid_blocks.mean():.1f}; at 0.50 they are baseline={permissive.baseline_valid_blocks.mean():.1f}, sec_test={permissive.followup_valid_blocks.mean():.1f}. Thresholds are applied to the source segment-quality broad-artifact fraction; threshold-specific rows expose the coverage/estimate trade-off and remain sensitivity diagnostics, not the formal estimate.", "", "## Limitations", "", f"No ICA, interpolation, rereferencing optimization or artifact repair is introduced. Broad artifact is gated using the precomputed source fraction, while local channel identities are reconstructed from the BDF with the same thresholds. Event timing remains heterogeneous across subjects (observed versus inferred); one incomplete visit is not upgraded by fallback timing. n={total} supports descriptive robustness only; no inferential p-values or treatment efficacy claims are made.", "", "Outputs: `eeg_quality_gated_block_features.csv`, `eeg_quality_gated_task_summary.csv`, `eeg_quality_gated_subject_summary.csv`, `eeg_quality_sensitivity_comparison.csv`, `threshold_sensitivity.csv`, and `eeg_quality_sensitivity.{png,svg,pdf}` plus `eeg_broad_threshold_sensitivity.{png,svg,pdf}`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_subject_rows(path: Path, frame: pd.DataFrame, subjects: list[str], append: bool) -> pd.DataFrame:
    if not append or not path.exists():
        return frame
    prior = pd.read_csv(path)
    if "subject" not in prior:
        return frame
    prior["subject"] = prior["subject"].astype(str)
    return pd.concat([prior[~prior.subject.isin(subjects)], frame], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(PAIRED_SUBJECTS))
    parser.add_argument("--append", action="store_true", help="replace supplied subjects within the paired7 sensitivity outputs")
    args = parser.parse_args()
    global SUBJECTS
    SUBJECTS = list(args.subjects)
    OUT.mkdir(parents=True, exist_ok=True)
    q = load_quality()
    blocks, tasks, subjects = extract(q)
    blocks = merge_subject_rows(OUT / "eeg_quality_gated_block_features.csv", blocks, SUBJECTS, args.append)
    tasks = merge_subject_rows(OUT / "eeg_quality_gated_task_summary.csv", tasks, SUBJECTS, args.append)
    subjects = merge_subject_rows(OUT / "eeg_quality_gated_subject_summary.csv", subjects, SUBJECTS, args.append)
    expected_subject_sessions = {(subject, session) for subject in PAIRED_SUBJECTS for session in SESSIONS}
    observed_subject_sessions = set(zip(subjects.subject.astype(str), subjects.session.astype(str)))
    if observed_subject_sessions != expected_subject_sessions:
        missing = sorted(expected_subject_sessions - observed_subject_sessions)
        unexpected = sorted(observed_subject_sessions - expected_subject_sessions)
        raise ValueError(f"paired7 subject-session completeness failed; missing={missing}; unexpected={unexpected}")
    # Rebuild all downstream comparisons from the accumulated paired-seven
    # subject table, including explicit NaNs for incomplete visits.
    SUBJECTS = list(PAIRED_SUBJECTS)
    comparison = compare(subjects)
    threshold = threshold_sensitivity(blocks)
    blocks.to_csv(OUT / "eeg_quality_gated_block_features.csv", index=False, encoding="utf-8-sig")
    tasks.to_csv(OUT / "eeg_quality_gated_task_summary.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(OUT / "eeg_quality_gated_subject_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "eeg_quality_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    threshold.to_csv(OUT / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    plot(comparison)
    plot_threshold_sensitivity(threshold)
    full_quality = pd.read_csv(QUALITY_PATH)
    metadata = {"subjects": SUBJECTS, "processed_subjects_this_run": list(PAIRED_SUBJECTS), "append": args.append, "sessions": SESSIONS, "tasks": TASKS, "quality_source": str(QUALITY_PATH), "quality_source_sha256": sha256_file(QUALITY_PATH), "legacy_source": str(LEGACY_PATH), "legacy_source_sha256": sha256_file(LEGACY_PATH), "script_sha256": sha256_file(Path(__file__)), "broad_exclude_threshold": BROAD_EXCLUDE, "local_bad_epoch_fraction_threshold": LOCAL_BAD_EPOCH_FRACTION, "artifact_ptp_uv": ARTIFACT_PTP_UV, "event_source_values": sorted(set(full_quality.event_alignment.astype(str))), "subject_session_completeness_assertion": {"expected": len(expected_subject_sessions), "observed": len(observed_subject_sessions), "passed": True}, "note": "old outputs and raw data are read-only"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(blocks, tasks, comparison, threshold)
    print(json.dumps({"output": str(OUT), "block_rows": len(blocks), "task_rows": len(tasks), "subject_session_rows": len(subjects), "comparison_rows": len(comparison), "append": args.append}, ensure_ascii=False))


if __name__ == "__main__":
    main()
