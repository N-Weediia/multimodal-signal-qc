"""Exploratory multimodal analysis for the two highlighted Hegu cases.

The raw recordings stay on the mounted J: drive.  This script reads them and
writes only derived tables/figures to ``outputs/analysis/20260901_case_ideas``.  It is
intentionally an exploratory bridge analysis, not a confirmatory clinical
model: the independent sample is two people, while windows are used only to
characterise within-session dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import signal, stats
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score

import main
from emg_qc import EXPECTED_SENSOR_IDS, assess_emg_channel, read_emg_csv, resolve_emg_mapping
from fnirs_qc import (
    assess_fnirs_channel,
    build_fnirs_sample_masks,
    is_infrared_eye_tracking_task,
    read_fnirs,
)


LOCAL_REPRO_ROOT = Path(__file__).resolve().parent / "data_repro_20260903"
RAW_ROOT = LOCAL_REPRO_ROOT if LOCAL_REPRO_ROOT.is_dir() else Path(r"J:\2026二康\数据")
DEFAULT_OUT = Path("outputs/analysis/20260901_case_ideas")
SUBJECTS = {
    "2026081403": "S1403",
    "2026081501": "S1501",
}
SESSIONS = {
    "2026081403": ["base_line", "sec", "sec_test"],
    "2026081501": ["base_line", "sec_test"],
}


def set_data_root(data_root: Path) -> None:
    """Override the historical raw-data root for local/subset analyses."""
    global RAW_ROOT
    RAW_ROOT = data_root.expanduser().resolve()


def canonicalize_sessions(sessions: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    canonical: dict[str, list[str]] = {}
    excluded: list[dict[str, str]] = []
    for subject, values in sessions.items():
        values = list(values)
        if "sec_test" in values and "sec" in values:
            values = [value for value in values if value != "sec"]
            excluded.append({"subject": subject, "session": "sec", "canonical_session": "sec_test", "reason": "duplicate follow-up; sec_test retained"})
        canonical[subject] = values
    return canonical, excluded


def infer_movement_event_labels(times: np.ndarray, subject: str) -> list[dict[str, object]]:
    """Map movement markers to protocol subtasks using the supplied protocol.

    Marker value ``1`` does not encode a task.  Marker timestamps are automatic
    and treated as accurate; the mapping below assigns task names from the
    confirmed fixed protocol order.  It is used for stratification, not as a
    replacement for the raw marker table.
    """
    labels: list[dict[str, object]] = []
    action_map = {
        3: ("摸头", 1), 4: ("摸头", 2), 5: ("摸头", 3),
        6: ("摸头", 4), 7: ("摸头", 5),
        9: ("摸膝", 1), 10: ("摸膝", 2), 11: ("摸膝", 3),
        12: ("摸膝", 4), 13: ("摸膝", 5),
        15: ("摸腰", 1), 16: ("摸腰", 2), 17: ("摸腰", 3),
        18: ("摸腰", 4), 19: ("摸腰", 5),
    }
    for index, onset in enumerate(np.asarray(times, dtype=float), start=1):
        if index == 1:
            task_name, event_role, status, confidence, block = "运动任务同步起点", "sync_start", "observed", "高", ""
        elif index == 2:
            task_name, event_role, status, confidence, block = "首个动作前准备/边界", "pre_first_action_boundary", "protocol_labeled", "高", ""
        elif index in action_map:
            task_name, block = action_map[index]
            event_role, status, confidence = "action_block_start", "protocol_labeled", "高"
        elif index == 8:
            task_name, event_role, status, confidence, block = "摸头-摸膝转换边界", "task_transition", "protocol_labeled", "高", ""
        elif index == 14:
            task_name, event_role, status, confidence, block = "摸膝-摸腰转换边界", "task_transition", "protocol_labeled", "高", ""
        elif index == 20:
            task_name, event_role, status, confidence, block = "前三动作结束边界", "task_end_boundary", "protocol_labeled", "高", ""
        elif subject in {"2026081401", "2026081403"} and 20 <= index <= 25:
            task_name, event_role, status, confidence, block = "握拳", "action_block_start", "protocol_labeled", "高", index - 19
        else:
            task_name, event_role, status, confidence, block = "未分类/异常标记", "unclassified", "unclassified", "低", ""
        previous_interval = ""
        if index > 1:
            previous_interval = f"与前标间隔 {float(times[index - 1] - times[index - 2]):.2f}s"
        if task_name in {"摸头", "摸膝", "摸腰", "握拳"}:
            evidence = f"固定动作顺序 + block 序号 {block}; {previous_interval}; 预期 block 间隔约24s"
        elif event_role == "task_transition":
            evidence = f"动作顺序边界; {previous_interval}; 预期转换间隔约45s"
        else:
            evidence = previous_interval or "记录第一个有效标"
        labels.append({
            "marker_index": index,
            "task_name": task_name,
            "event_role": event_role,
            "inference_status": status,
            "confidence": confidence,
            "block_index": block,
            "evidence": evidence,
            "timestamp_seconds": float(onset),
        })
    return labels


@dataclass
class EventSeries:
    times: np.ndarray
    source: str


MOVEMENT_ACTION_TASKS = {"摸头", "摸膝", "摸腰", "握拳"}


def _movement_event_score(events: EventSeries | None) -> int:
    """Count protocol action markers available in an event stream."""
    if events is None or not events.times.size:
        return 0
    labels = infer_movement_event_labels(events.times, "")
    return sum(label["task_name"] in MOVEMENT_ACTION_TASKS for label in labels)


def select_movement_event_reference(
    fnirs_events: EventSeries | None,
    eeg_events: EventSeries | None,
) -> tuple[EventSeries | None, EventSeries | None]:
    """Choose the best observed EEG/fNIRS stream for cross-modal alignment.

    Protocol coverage is the primary criterion.  fNIRS wins ties to preserve
    the historical preference, while the other stream is retained for an
    explicit marker-disagreement audit.
    """
    candidates = [(fnirs_events, "fNIRS"), (eeg_events, "EEG")]
    ranked = [
        (events, source, _movement_event_score(events))
        for events, source in candidates
        if events is not None and events.times.size
    ]
    if not ranked:
        return None, None
    ranked.sort(key=lambda item: (item[2], item[1] == "fNIRS"), reverse=True)
    primary = ranked[0][0]
    secondary = next((item[0] for item in ranked[1:] if item[0] is not None), None)
    return primary, secondary


def infer_emg_marker_rows(
    subject: str,
    reference_events: EventSeries | None,
    emg_time: np.ndarray,
    emg_sample_rate: float,
    secondary_events: EventSeries | None = None,
) -> list[dict[str, object]]:
    """Project EEG/fNIRS marker times onto the EMG frame-0 time base.

    EMG frame 0 is defined as ``emg_time[0]``.  Marker timestamps from the
    selected EEG/fNIRS reference are therefore shifted by the interval from
    that stream's first marker.  When both streams exist, their relative
    marker timing disagreement is retained in milliseconds for QC.
    """
    if reference_events is None or not reference_events.times.size or not emg_time.size:
        return []
    labels = infer_movement_event_labels(reference_events.times, subject)
    emg_origin = float(emg_time[0])
    sample_rate = float(emg_sample_rate)
    reference_origin = float(reference_events.times[0])
    secondary_times = secondary_events.times if secondary_events is not None else np.asarray([], dtype=float)
    rows: list[dict[str, object]] = []
    for label in labels:
        relative_seconds = float(label["timestamp_seconds"]) - reference_origin
        marker_time = emg_origin + relative_seconds
        frame_index = int(round(relative_seconds * sample_rate))
        disagreement_ms = np.nan
        if int(label["marker_index"]) <= secondary_times.size:
            secondary_relative = float(secondary_times[int(label["marker_index"]) - 1]) - float(secondary_times[0])
            disagreement_ms = abs(relative_seconds - secondary_relative) * 1000.0
        rows.append({
            "subject": subject,
            "marker_index": int(label["marker_index"]),
            "task_name": label["task_name"],
            "event_role": label["event_role"],
            "block_index": label["block_index"],
            "reference_marker_source": reference_events.source,
            "secondary_marker_source": secondary_events.source if secondary_events is not None else "",
            "reference_timestamp_seconds": float(label["timestamp_seconds"]),
            "emg_time_seconds": marker_time,
            "emg_frame_index": frame_index,
            "emg_frame_zero_definition": "emg_time[0]",
            "alignment_status": "inferred_from_eeg_or_fnirs_marker",
            "secondary_marker_disagreement_ms": disagreement_ms,
            "confidence": label["confidence"],
        })
    return rows


def _first(paths: Iterable[Path]) -> Path | None:
    paths = sorted(paths)
    return paths[0] if paths else None


def find_recording(subject: str, session: str, modality: str, task: str) -> Path | None:
    root = RAW_ROOT / subject / session
    if modality == "EEG":
        candidates = list((root / "EEG").glob(f"{task}/*.bdf"))
        recordings = [p for p in candidates if p.name.lower() != "evt.bdf"]
        # Some exports leave a tiny aborted data.1.bdf beside the formal
        # recording.  Analysis should select the largest complete BDF; the
        # chosen path is still recorded downstream for auditability.
        return max(recordings, key=lambda p: p.stat().st_size) if recordings else None
    if modality == "FNIRS":
        return _first((root / "FNIRS" / task).glob("*.nirs"))
    if modality == "EMG":
        return _first((root / "EMG").glob("*.csv"))
    raise ValueError(modality)


def _header_fields(path: Path) -> tuple[int, int, list[str], list[int]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode("ascii").strip())
        record_count = int(fixed[236:244].decode("ascii").strip())
        channel_count = int(fixed[252:256].decode("ascii").strip())
        extension = handle.read(header_bytes - 256)
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    fields: list[list[str]] = []
    cursor = 0
    for width in widths:
        fields.append([
            extension[cursor + i * width: cursor + (i + 1) * width]
            .decode("latin-1", errors="replace").strip()
            for i in range(channel_count)
        ])
        cursor += width * channel_count
    return header_bytes, record_count, fields[0], [int(x) for x in fields[8]]


def read_bdf_annotation(path: Path | None) -> EventSeries | None:
    """Read descriptions from a companion BDF annotation file.

    BioSemi exports an empty annotation at integer seconds and a second
    annotation carrying description ``1`` at each task trigger.  We keep only
    the described triggers and retain the empty source when no such event is
    present for transparent reporting.
    """
    if path is None or not path.exists():
        return None
    header_bytes, record_count, labels, samples = _header_fields(path)
    annotation_indices = [i for i, label in enumerate(labels) if "annotation" in label.lower()]
    if not annotation_indices:
        return None
    annotation_index = annotation_indices[0]
    bytes_per_record = 3 * sum(samples)
    record_bytes = path.read_bytes()[header_bytes:]
    chunks: list[bytes] = []
    for record in range(record_count):
        raw = record_bytes[record * bytes_per_record: (record + 1) * bytes_per_record]
        offset = 3 * sum(samples[:annotation_index])
        chunks.append(raw[offset: offset + 3 * samples[annotation_index]])
    stream = b"".join(chunks).replace(b"\x00", b"")
    # The BDF+ form is +onset [\x15duration] \x14 description \x14.
    # Hardware impedance checks (Start/Stop Impedance) must be excluded per docs/04_EVENT_MARKERS.md.
    matches = re.finditer(rb"\+([0-9.]+)(?:\x15[^\x14]*)?\x14([^\x14\x15]*)", stream)
    times = []
    for match in matches:
        desc = match.group(2).decode("utf-8", errors="replace").strip()
        if not desc or "impedance" in desc.casefold():
            continue
        times.append(float(match.group(1)))
    if not times:
        return None
    return EventSeries(np.asarray(times, dtype=float), "EEG evt.bdf description=1")


def read_fnirs_events(path: Path | None) -> EventSeries | None:
    if path is None or not path.exists():
        return None
    document = loadmat(path, variable_names=["t", "s"], simplify_cells=True)
    if "t" not in document or "s" not in document:
        return None
    time = np.ravel(np.asarray(document["t"], dtype=float))
    marker = np.asarray(document["s"])
    if marker.ndim == 1:
        indices = np.flatnonzero(marker != 0)
    else:
        # NirSmart often stores the same trigger in several status columns.
        indices = np.flatnonzero(np.any(marker != 0, axis=1))
    if not indices.size:
        return None
    # Some NirSmart exports repeat one trigger in three status columns a few
    # hundred milliseconds apart.  Collapse those duplicates into one onset.
    raw_times = time[indices]
    kept = [float(raw_times[0])]
    for value in raw_times[1:]:
        if float(value) - kept[-1] > 1.0:
            kept.append(float(value))
        elif float(value) < kept[-1]:
            kept[-1] = float(value)
    return EventSeries(np.asarray(kept, dtype=float), "fNIRS s marker (deduplicated)")


def butter_filter(data: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    high = min(high, fs * 0.45)
    sos = signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def robust_z(data: np.ndarray, axis: int = -1) -> np.ndarray:
    med = np.nanmedian(data, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(data - med), axis=axis, keepdims=True) * 1.4826
    return (data - med) / np.maximum(mad, 1e-9)


def eeg_features(path: Path, events: EventSeries | None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    header, physical, _ = main.read_bdf(path, None)
    labels = [str(x) for x in header.labels]
    scalp_indices = [i for i, label in enumerate(labels) if main.normalize_label(label) in {
        main.normalize_label(x) for x in main.FULL_EEG_CHANNELS
    }]
    scalp_labels = [labels[i] for i in scalp_indices]
    data_uv = physical[scalp_indices]
    data_uv -= np.median(data_uv, axis=0, keepdims=True)
    clean = butter_filter(data_uv, header.sample_rate, 1.0, 45.0)
    # Amplitude scale differs markedly between exports and several channels
    # contain transient movement artifacts.  Within-channel robust scaling
    # makes the exploratory network features reflect temporal organisation,
    # not the recorder's absolute unit or a few saturated samples.
    clean = np.clip(robust_z(clean, axis=1), -20.0, 20.0).astype(np.float32)
    fs = header.sample_rate
    bands = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    roi_names = {
        "frontal": ["Fz", "F3", "F4", "FCz", "FC1", "FC2"],
        "left_sensorimotor": ["C1", "C3", "CP1", "CP3"],
        "right_sensorimotor": ["C2", "C4", "CP2", "CP4"],
        "parietal": ["Pz", "P3", "P4", "POz"],
    }
    label_map = {main.normalize_label(label): i for i, label in enumerate(scalp_labels)}
    features: dict[str, float] = {}
    band_matrix: dict[str, np.ndarray] = {}
    # Welch on the full continuous record gives a stable individual signature.
    nperseg = min(int(fs * 4), clean.shape[1])
    freqs, psd = signal.welch(clean, fs=fs, nperseg=nperseg, axis=-1, detrend="linear")
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        values = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        band_matrix[band_name] = values
        for roi, wanted in roi_names.items():
            indices = [label_map[main.normalize_label(x)] for x in wanted if main.normalize_label(x) in label_map]
            if indices:
                features[f"eeg_{roi}_{band_name}"] = float(np.nanmean(values[indices]))
    features["eeg_global_alpha_beta_ratio"] = float(np.nanmean(band_matrix["alpha"]) / np.maximum(np.nanmean(band_matrix["beta"]), 1e-12))
    # Hilbert-envelope coupling is more robust than zero-lag raw correlation.
    roi_series: dict[str, np.ndarray] = {}
    for roi, wanted in roi_names.items():
        indices = [label_map[main.normalize_label(x)] for x in wanted if main.normalize_label(x) in label_map]
        if indices:
            roi_series[roi] = np.nanmean(clean[indices], axis=0)
    theta_env = {roi: np.abs(signal.hilbert(butter_filter(values[None, :], fs, 4, 8)[0])) for roi, values in roi_series.items()}
    beta_env = {roi: np.abs(signal.hilbert(butter_filter(values[None, :], fs, 13, 30)[0])) for roi, values in roi_series.items()}
    for left, right in [("frontal", "left_sensorimotor"), ("frontal", "right_sensorimotor"), ("left_sensorimotor", "right_sensorimotor"), ("left_sensorimotor", "parietal"), ("right_sensorimotor", "parietal")]:
        if left in theta_env and right in theta_env:
            features[f"eeg_theta_envcorr_{left}_{right}"] = float(np.corrcoef(theta_env[left], theta_env[right])[0, 1])
        if left in beta_env and right in beta_env:
            features[f"eeg_beta_envcorr_{left}_{right}"] = float(np.corrcoef(beta_env[left], beta_env[right])[0, 1])
    event_indices_used: list[int] = []
    if events is not None and events.times.size:
        epoch_len = int(round(10 * fs))
        pre_len = int(round(5 * fs))
        post_values: list[float] = []
        pre_values: list[float] = []
        for event_index, onset in enumerate(events.times):
            start = int(round(onset * fs))
            if start - pre_len < 0 or start + epoch_len > clean.shape[1]:
                continue
            pre = np.mean(theta_env["frontal"][start - pre_len:start]) if "frontal" in theta_env else np.nan
            post = np.mean(theta_env["frontal"][start:start + epoch_len]) if "frontal" in theta_env else np.nan
            pre_values.append(float(pre)); post_values.append(float(post))
            event_indices_used.append(event_index)
        if post_values:
            features["eeg_event_theta_frontal_delta"] = float(np.mean(post_values) - np.mean(pre_values))
            features["eeg_event_count_used"] = float(len(post_values))
        event_deltas = np.asarray(post_values) - np.asarray(pre_values)
    else:
        event_deltas = np.asarray([], dtype=float)
    extras = {"clean": clean, "fs": np.asarray([fs]), "roi_series": roi_series, "band_matrix": band_matrix, "freqs": freqs, "psd": psd, "event_theta_deltas": event_deltas, "event_indices_used": np.asarray(event_indices_used if events is not None else [], dtype=int)}
    return features, extras


def fnirs_features(path: Path, events: EventSeries | None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    recording = read_fnirs(path, None)
    fs = recording.sample_rate_hz
    intensity_850 = np.column_stack([c.intensity_b for c in recording.channels])
    intensity_730 = np.column_stack([c.intensity_a for c in recording.channels])
    def od(values: np.ndarray) -> np.ndarray:
        baseline = np.nanmedian(np.where(values > 0, values, np.nan), axis=0, keepdims=True)
        values = np.where(values > 0, values, np.nan)
        out = -np.log(np.maximum(values, 1e-12) / np.maximum(baseline, 1e-12))
        out = np.asarray(out, dtype=float)
        for col in range(out.shape[1]):
            valid = np.isfinite(out[:, col])
            if np.count_nonzero(valid) >= 2:
                out[:, col] = np.interp(np.arange(out.shape[0]), np.flatnonzero(valid), out[valid, col])
            else:
                out[:, col] = 0.0
        return out
    od850 = od(intensity_850)
    od730 = od(intensity_730)
    # A wavelength-balanced optical-density proxy; no DPF/extinction values are
    # assumed, so it is used for relative activity/connectivity only.
    proxy = (robust_z(od850, axis=0) - robust_z(od730, axis=0)) / np.sqrt(2.0)
    proxy = butter_filter(proxy.T, fs, 0.01, 0.2).T
    features: dict[str, float] = {}
    # fALFF-like fraction and low-frequency power per channel.
    freqs, psd = signal.welch(proxy, fs=fs, nperseg=min(int(fs * 64), proxy.shape[0]), axis=0, detrend="linear")
    low = (freqs >= 0.01) & (freqs <= 0.08)
    whole = (freqs >= 0.01) & (freqs <= 0.2)
    frac = np.trapezoid(psd[low], freqs[low], axis=0) / np.maximum(np.trapezoid(psd[whole], freqs[whole], axis=0), 1e-12)
    features["fnirs_falff_proxy_global"] = float(np.nanmean(frac))
    features["fnirs_lowfreq_proxy_global"] = float(np.nanmean(np.trapezoid(psd[low], freqs[low], axis=0)))
    # Network integration on channel series.  Correlation is Fisher-z
    # transformed before graph summaries to reduce compression at high r.
    corr = np.corrcoef(proxy.T)
    corr = np.clip(corr, -0.999, 0.999)
    zcorr = np.arctanh(corr)
    np.fill_diagonal(zcorr, 0.0)
    features["fnirs_network_mean_abs_fisher_z"] = float(np.mean(np.abs(zcorr[np.triu_indices_from(zcorr, 1)])))
    # Threshold to the strongest 15% of edges before graph efficiency; using a
    # complete weighted graph would make the efficiency identically one.
    upper = np.abs(zcorr[np.triu_indices_from(zcorr, 1)])
    edge_threshold = float(np.nanpercentile(upper, 85))
    graph = nx.from_numpy_array(np.where(np.abs(zcorr) >= edge_threshold, np.abs(zcorr), 0.0))
    features["fnirs_network_global_efficiency"] = float(nx.global_efficiency(graph))
    strengths = np.sum(np.abs(zcorr), axis=0)
    features["fnirs_network_hub_strength"] = float(np.max(strengths))
    event_indices_used: list[int] = []
    if events is not None and events.times.size:
        event_values: list[float] = []
        event_delta_matrix: list[np.ndarray] = []
        event_left: list[float] = []
        event_right: list[float] = []
        for event_index, onset in enumerate(events.times):
            start = int(round(onset * fs))
            pre0, pre1 = start - int(round(5 * fs)), start
            post0, post1 = start + int(round(2 * fs)), start + int(round(12 * fs))
            if pre0 < 0 or post1 > proxy.shape[0]:
                continue
            delta = np.nanmean(proxy[post0:post1], axis=0) - np.nanmean(proxy[pre0:pre1], axis=0)
            event_values.append(float(np.nanmean(delta)))
            event_delta_matrix.append(np.asarray(delta, dtype=float))
            positions = np.asarray([c.position for c in recording.channels])
            # Probe x coordinates are used only to create a reproducible
            # left/right split; the scalp orientation should be verified against
            # the digitised montage before making anatomical claims.
            event_left.append(float(np.nanmean(delta[positions[:, 0] < np.median(positions[:, 0])])) )
            event_right.append(float(np.nanmean(delta[positions[:, 0] >= np.median(positions[:, 0])])) )
            event_indices_used.append(event_index)
        if event_values:
            features["fnirs_event_proxy_delta"] = float(np.mean(event_values))
            features["fnirs_event_left_right_asymmetry"] = float(np.mean(event_left) - np.mean(event_right))
            features["fnirs_event_count_used"] = float(len(event_values))
        event_deltas = np.asarray(event_values)
        event_delta_vectors = np.vstack(event_delta_matrix) if event_delta_matrix else np.empty((0, proxy.shape[1]))
    else:
        event_deltas = np.asarray([], dtype=float)
        event_delta_vectors = np.empty((0, proxy.shape[1]))
    extras = {"proxy": proxy, "fs": np.asarray([fs]), "corr": corr, "frac": frac, "positions": np.asarray([c.position for c in recording.channels]), "freqs": freqs, "psd": psd, "event_proxy_deltas": event_deltas, "event_delta_vectors": event_delta_vectors, "event_left_deltas": np.asarray(event_left if events is not None and events.times.size else [], dtype=float), "event_right_deltas": np.asarray(event_right if events is not None and events.times.size else [], dtype=float), "event_indices_used": np.asarray(event_indices_used, dtype=int)}
    return features, extras


def eye_fnirs_features(path: Path, events: EventSeries | None = None) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compute a quality-gated, non-haemodynamic feature set for eye-task fNIRS.

    Eye-tracker infrared light is a known confound in these recordings.  The
    output therefore stays in optical-density/proxy space and carries the QC
    status and masks alongside every derived value.  No HbO/HbR conversion or
    protocol task labels are inferred from the (semantically unknown) marker
    stream.
    """
    recording = read_fnirs(path, None)
    fs = float(recording.sample_rate_hz)
    metrics = [assess_fnirs_channel(channel, fs) for channel in recording.channels]
    masks = [build_fnirs_sample_masks(channel, fs) for channel in recording.channels]
    wavelength_valid_mask = np.stack([item[0] for item in masks]).astype(bool)
    sample_mask = np.all(wavelength_valid_mask, axis=1)
    motion_mask = np.stack([item[1] for item in masks]).astype(bool)
    repair_mask = np.stack([item[2] for item in masks]).astype(bool)
    # Keep the helper's analysis mask as the source of truth while applying
    # each component explicitly so downstream consumers can audit the gate.
    analysis_mask = np.stack([item[3] for item in masks]).astype(bool)
    analysis_mask &= sample_mask & ~motion_mask & ~repair_mask
    # POOR channels are retained for QC counts but are never re-admitted to
    # the exploratory proxy branch. GOOD/CAUTION channels remain eligible with
    # their status exposed in the output for sensitivity review.
    channel_gate = np.asarray([metric.status != "POOR" for metric in metrics], dtype=bool)
    analysis_mask &= channel_gate[:, None]
    known_ir = bool(is_infrared_eye_tracking_task(path))
    quality_counts = {status: int(sum(metric.status == status for metric in metrics)) for status in ("GOOD", "CAUTION", "POOR")}
    mean_score = float(np.nanmean([metric.quality_score for metric in metrics]))
    poor_fraction = quality_counts["POOR"] / max(len(metrics), 1)
    quality_before = "POOR" if poor_fraction >= 0.20 or mean_score < 55 else "CAUTION" if quality_counts["POOR"] or quality_counts["CAUTION"] else "GOOD"
    session_status = "CAUTION" if known_ir and quality_before == "GOOD" else quality_before
    occlusion_status = "PARTIAL_REPORTED" if known_ir else "NOT_APPLICABLE"

    # Keep invalid/motion samples out of summaries. Interpolation is limited to
    # numerical stability for filtering and is never presented as physiology.
    intensity_a = np.column_stack([c.intensity_a for c in recording.channels])
    intensity_b = np.column_stack([c.intensity_b for c in recording.channels])
    def _od(values: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values, dtype=float)
        for col in range(values.shape[1]):
            positive = np.isfinite(values[:, col]) & (values[:, col] > 0)
            if np.count_nonzero(positive) == 0:
                continue
            baseline = float(np.median(values[positive, col]))
            raw = -np.log(np.where(positive, values[:, col], np.nan) / max(baseline, 1e-12))
            valid = np.isfinite(raw)
            if np.count_nonzero(valid) >= 2:
                out[:, col] = np.interp(np.arange(out.shape[0]), np.flatnonzero(valid), raw[valid])
        return out
    proxy = (robust_z(_od(intensity_a), axis=0) - robust_z(_od(intensity_b), axis=0)) / np.sqrt(2.0)
    proxy = butter_filter(proxy.T, fs, 0.01, 0.2).T
    # Keep the physical-channel layout for traceability, but never expose a
    # finite proxy for a POOR channel or a sample rejected by the combined
    # wavelength-validity, motion, repair, and channel gates.
    proxy[:, ~channel_gate] = np.nan
    proxy[~analysis_mask.T] = np.nan
    valid = analysis_mask & np.isfinite(proxy.T)
    channel_means = np.asarray([
        float(np.nanmean(proxy[:, i][valid[i]])) if np.any(valid[i]) else np.nan
        for i in range(proxy.shape[1])
    ])
    features: dict[str, object] = {
        "eye_fnirs_known_ir_interference": int(known_ir),
        "eye_fnirs_occlusion_reported": int(known_ir),
        "eye_fnirs_occlusion_status": occlusion_status,
        "eye_fnirs_quality_status": session_status,
        "eye_fnirs_quality_status_before_interference": quality_before,
        "eye_fnirs_analysis_branch": "EYE_IR_DEGRADED_MASKED" if known_ir else "STANDARD_MASKED",
        "eye_fnirs_quality_degradation": "KNOWN_IR_INTERFERENCE_PARTIALLY_OCCLUDED" if known_ir else "",
        "eye_fnirs_occlusion_mask_available": 0,
        "eye_fnirs_mean_quality_score": mean_score,
        "eye_fnirs_quality_good_channels": quality_counts["GOOD"],
        "eye_fnirs_quality_caution_channels": quality_counts["CAUTION"],
        "eye_fnirs_quality_poor_channels": quality_counts["POOR"],
        "eye_fnirs_channel_count": int(len(recording.channels)),
        "eye_fnirs_eligible_channel_count": int(np.count_nonzero(channel_gate)),
        "eye_fnirs_sample_valid_coverage": float(np.mean(sample_mask)),
        "eye_fnirs_motion_mask_coverage": float(np.mean(motion_mask)),
        "eye_fnirs_analysis_sample_coverage": float(np.mean(analysis_mask)),
        "eye_fnirs_proxy_global_mean": float(np.nanmean(channel_means)),
        "eye_fnirs_proxy_global_abs_mean": float(np.nanmean(np.abs(channel_means))),
        "eye_fnirs_event_count_observed": int(events.times.size) if events is not None else 0,
        "eye_fnirs_event_semantics": "unknown; no task labels inferred",
    }
    extras = {
        "proxy": proxy,
        "analysis_sample_mask": analysis_mask,
        "wavelength_valid_sample_mask": wavelength_valid_mask,
        "sample_valid_mask": sample_mask,
        "motion_artifact_mask": motion_mask,
        "repair_mask": repair_mask,
        "proxy_channel_gate": channel_gate,
        "fs": np.asarray([fs]),
        "positions": np.asarray([c.position for c in recording.channels]),
        "event_times": np.asarray(events.times if events is not None else [], dtype=float),
    }
    return features, extras


def emg_features(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    raw_series = read_emg_csv(path, None)
    mapping = resolve_emg_mapping([item.sensor_id for item in raw_series], path)
    # The primary feature space is logical muscle order.  The explicitly
    # confirmed shifted baseline is therefore eligible; unknown shifts remain
    # visible in QC but are excluded until a record-level mapping is supplied.
    if mapping["mapping_valid_for_primary_features"]:
        allowed = {int(value) for value in mapping["physical_to_logical"]}
    else:
        raise ValueError(
            f"EMG 记录映射未确认（{mapping['mapping_status']}）；保留 QC，不进入主肌肉特征。"
        )
    series = [item for item in raw_series if item.sensor_id in allowed and item.muscle_index is not None]
    if not series:
        raise ValueError("EMG 文件没有用户确认映射中的传感器。")
    features: dict[str, float] = {}
    envelopes: list[np.ndarray] = []
    line_noise: list[float] = []
    quality_status: list[str] = []
    quality_score: list[float] = []
    for item in series:
        fs = item.sample_rate_hz
        values = np.asarray(item.values, dtype=float)
        filtered = butter_filter(values[None, :] - np.nanmedian(values), fs, 20.0, min(450.0, fs * 0.45))[0]
        envelope = np.abs(signal.hilbert(filtered))
        envelopes.append(envelope)
        try:
            metric = assess_emg_channel(item)
            line_noise.append(float(metric.line_noise_power_percent))
            quality_status.append(str(getattr(metric, "status", "GOOD")))
            quality_score.append(float(getattr(metric, "quality_score", np.nan)))
        except Exception:
            line_noise.append(float("nan"))
            quality_status.append("UNKNOWN")
            quality_score.append(float("nan"))
        rms = float(np.sqrt(np.mean(filtered ** 2)))
        envelope_mean = float(np.mean(envelope))
        envelope_cv = float(np.std(envelope) / np.maximum(np.mean(envelope), 1e-12))
        logical_index = item.muscle_index
        # Only scored channels are eligible for logical-muscle features.
        # POOR and UNKNOWN channels remain available through the audit arrays
        # below, but must never become primary feature values.
        if logical_index is not None and quality_status[-1] in {"GOOD", "CAUTION"}:
            features[f"emg_muscle_{logical_index:02d}_rms"] = rms
            features[f"emg_muscle_{logical_index:02d}_envelope_mean"] = envelope_mean
            features[f"emg_muscle_{logical_index:02d}_envelope_cv"] = envelope_cv
        # Physical-ID columns are retained as audit-only fields.  They must
        # not be used for baseline/follow-up matching when profiles differ.
        features[f"emg_sensor_{item.sensor_id}_rms"] = rms
        features[f"emg_sensor_{item.sensor_id}_envelope_mean"] = envelope_mean
        features[f"emg_sensor_{item.sensor_id}_envelope_cv"] = envelope_cv
    # A few CSV rows are missing at the end of individual sensors.  Align to
    # the common support before computing coactivation/synergy features.
    common_length = min(len(values) for values in envelopes)
    envelope_all = np.vstack([values[:common_length] for values in envelopes])
    eligible_statuses = {"GOOD", "CAUTION"}
    good_indices = [
        i for i, value in enumerate(line_noise)
        if np.isfinite(value) and value < 20.0 and quality_status[i] in eligible_statuses
    ]
    if len(good_indices) < 3:
        # Relax only the line-noise threshold when necessary; never re-admit
        # POOR/UNKNOWN channels merely because fewer than three remain.
        good_indices = [i for i, status in enumerate(quality_status) if status in eligible_statuses]
    if not good_indices:
        # Keep an unscorable record visible without allowing NMF/correlation
        # routines to manufacture values from rejected channels.
        features.update({
            "emg_global_coactivation": float("nan"),
            "emg_coactivation_sensor_count": 0.0,
            "emg_coactivation_muscle_count": 0.0,
            "emg_coactivation_excluded_sensor_count": float(len(series)),
        })
        extras = {
            "envelope": np.empty((0, common_length), dtype=float),
            "time": series[0].time[:common_length],
            "fs": np.asarray([series[0].sample_rate_hz]),
            "synergy_w": np.empty((0, 0), dtype=float),
            "synergy_h": np.empty((0, 0), dtype=float),
            "sensor_ids": np.asarray([], dtype=int),
            "physical_sensor_ids": np.asarray([], dtype=int),
            "logical_muscle_indices": np.asarray([], dtype=int),
            "mapping_profile_id": np.asarray(mapping["mapping_profile_id"]),
            "mapping_status": np.asarray(mapping["mapping_status"]),
            "mapping_valid_for_primary_features": np.asarray(mapping["mapping_valid_for_primary_features"]),
            "line_noise_percent": np.asarray([], dtype=float),
            "excluded_physical_sensor_ids": np.asarray([s.sensor_id for s in series], dtype=int),
            "excluded_logical_muscle_indices": np.asarray([s.muscle_index if s.muscle_index is not None else -1 for s in series], dtype=int),
            "quality_status": np.asarray(quality_status),
            "quality_score": np.asarray(quality_score),
        }
        return features, extras
    envelope_matrix = envelope_all[good_indices]
    good_series = [series[i] for i in good_indices]
    # Muscle synergy is an interpretable low-rank description of coordinated
    # muscle recruitment, estimated separately per recording.
    scaled = np.maximum(envelope_matrix, 0)
    down = scaled[:, :: max(1, int(round(np.nanmedian([s.sample_rate_hz for s in series])) / 20))]
    n_components = min(3, down.shape[0], max(1, down.shape[1] // 50))
    model = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=500)
    w = model.fit_transform(down.T)
    h = model.components_
    features["emg_synergy_reconstruction_error"] = float(model.reconstruction_err_ / np.maximum(np.linalg.norm(down), 1e-12))
    for component in range(n_components):
        weights = h[component]
        order = np.argsort(weights)[::-1]
        top = good_series[int(order[0])]
        features[f"emg_synergy_{component + 1}_top_muscle_index"] = float(top.muscle_index if top.muscle_index is not None else -1)
        features[f"emg_synergy_{component + 1}_top_sensor"] = float(top.sensor_id)
        features[f"emg_synergy_{component + 1}_top_weight"] = float(weights[order[0]] / np.maximum(np.sum(weights), 1e-12))
    if envelope_matrix.shape[0] >= 2:
        coactivation = np.corrcoef(envelope_matrix)
        off_diag = coactivation[~np.eye(coactivation.shape[0], dtype=bool)]
        global_coactivation = float(np.mean(off_diag)) if off_diag.size else float("nan")
    else:
        global_coactivation = float("nan")
    features["emg_global_coactivation"] = global_coactivation
    features["emg_coactivation_sensor_count"] = float(len(good_indices))
    features["emg_coactivation_muscle_count"] = float(len(good_indices))
    features["emg_coactivation_excluded_sensor_count"] = float(len(series) - len(good_indices))
    extras = {
        "envelope": envelope_matrix,
        "time": series[0].time[:common_length],
        "fs": np.asarray([series[0].sample_rate_hz]),
        "synergy_w": w,
        "synergy_h": h,
        "sensor_ids": np.asarray([s.sensor_id for s in good_series]),
        "physical_sensor_ids": np.asarray([s.sensor_id for s in good_series]),
        "logical_muscle_indices": np.asarray([s.muscle_index if s.muscle_index is not None else -1 for s in good_series]),
        "mapping_profile_id": np.asarray(mapping["mapping_profile_id"]),
        "mapping_status": np.asarray(mapping["mapping_status"]),
        "mapping_valid_for_primary_features": np.asarray(mapping["mapping_valid_for_primary_features"]),
        "line_noise_percent": np.asarray([line_noise[i] for i in good_indices]),
        "excluded_physical_sensor_ids": np.asarray([s.sensor_id for i, s in enumerate(series) if i not in good_indices]),
        "excluded_logical_muscle_indices": np.asarray([s.muscle_index if s.muscle_index is not None else -1 for i, s in enumerate(series) if i not in good_indices]),
        "quality_status": np.asarray(quality_status),
        "quality_score": np.asarray(quality_score),
    }
    return features, extras


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_network_figure(path: Path, subject: str, session: str, corr: np.ndarray, positions: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    threshold = np.nanpercentile(np.abs(corr[np.triu_indices_from(corr, 1)]), 85)
    graph = nx.Graph()
    graph.add_nodes_from(range(corr.shape[0]))
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j]) >= threshold:
                graph.add_edge(i, j, weight=abs(corr[i, j]))
    xy = {i: (float(positions[i, 0]), float(positions[i, 1])) for i in range(len(positions))}
    widths = [1 + 4 * graph.edges[e].get("weight", 0) for e in graph.edges]
    nx.draw_networkx_nodes(graph, xy, node_size=55, node_color=np.arange(len(positions)), cmap="viridis", ax=ax)
    nx.draw_networkx_edges(graph, xy, width=widths, alpha=0.45, edge_color="#2f4858", ax=ax)
    ax.set_title(f"{subject} {session}: fNIRS proxy network (top 15% edges)")
    ax.set_xlabel("probe x (montage units; orientation to verify)")
    ax.set_ylabel("probe y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_timeline(path: Path, subject: str, session: str, eeg: dict[str, np.ndarray] | None, fnirs: dict[str, np.ndarray] | None, emg: dict[str, np.ndarray] | None, events: EventSeries | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), dpi=160, sharex=False)
    if eeg is not None:
        clean = eeg["clean"]
        fs = float(eeg["fs"][0])
        sample = max(1, int(fs / 10))
        axes[0].plot(np.arange(clean.shape[1])[::sample] / fs, np.nanmean(clean[:, ::sample], axis=0), lw=0.5, color="#1f4e79")
        axes[0].set_title("EEG mean scalp signal (filtered, re-referenced)")
        axes[0].set_ylabel("uV")
    if fnirs is not None:
        proxy = fnirs["proxy"]
        fs = float(fnirs["fs"][0])
        axes[1].plot(np.arange(proxy.shape[0]) / fs, np.nanmean(proxy, axis=1), lw=0.8, color="#c44e52")
        axes[1].set_title("fNIRS wavelength-balanced OD proxy")
        axes[1].set_ylabel("z proxy")
    if emg is not None:
        env = emg["envelope"]
        t = emg["time"]
        axes[2].plot(t, np.nanmean(env, axis=0), lw=0.8, color="#55a868")
        axes[2].set_title("sEMG mean envelope")
        axes[2].set_ylabel("source unit")
        axes[2].set_xlabel("seconds")
    if events is not None:
        for ax in axes:
            for onset in events.times:
                ax.axvline(onset, color="#777777", lw=0.35, alpha=0.35)
    fig.suptitle(f"{subject} {session}: multimodal timeline")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_delta_heatmap(path: Path, rows: list[dict[str, object]]) -> None:
    """Show baseline-to-second-session changes as a descriptive signature."""
    target_by_subject = {"2026081403": "sec_test", "2026081501": "sec_test"}
    selected = [
        ("fnirs_event_proxy_delta", "fNIRS event response"),
        ("fnirs_event_left_right_asymmetry", "fNIRS L-R asymmetry"),
        ("fnirs_network_mean_abs_fisher_z", "fNIRS network coupling"),
        ("rest_fnirs_network_mean_abs_fisher_z", "rest fNIRS coupling"),
        ("emg_global_coactivation", "EMG coactivation"),
        ("eeg_theta_envcorr_left_sensorimotor_right_sensorimotor", "EEG interhemispheric theta"),
        ("eye_eeg_frontal_theta", "eye EEG frontal theta"),
    ]
    values: list[list[float]] = []
    labels: list[str] = []
    for subject, target in target_by_subject.items():
        subject_rows = {str(r["session"]): r for r in rows if str(r["subject"]) == subject}
        if "base_line" not in subject_rows or target not in subject_rows:
            continue
        base, follow = subject_rows["base_line"], subject_rows[target]
        row_values: list[float] = []
        for key, _ in selected:
            a, b = base.get(key), follow.get(key)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                row_values.append(np.nan)
            else:
                scale = max(abs(float(a)), 1e-6)
                row_values.append((float(b) - float(a)) / scale * 100.0)
        values.append(row_values)
        labels.append(str(base["name"]))
    if not values:
        return
    matrix = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 3.5), dpi=160)
    vmax = max(50.0, float(np.nanpercentile(np.abs(matrix), 90)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(selected)), [label for _, label in selected], rotation=30, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("Baseline to second recording: multimodal exploratory signature")
    ax.set_xlabel("Percentage change; within-person descriptive only")
    fig.colorbar(image, ax=ax, label="change (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_delta_summary(path: Path, rows: list[dict[str, object]]) -> None:
    target_by_subject = {"2026081403": "sec_test", "2026081501": "sec_test"}
    selected = [
        "fnirs_event_proxy_delta", "fnirs_event_left_right_asymmetry",
        "fnirs_network_mean_abs_fisher_z", "rest_fnirs_network_mean_abs_fisher_z",
        "emg_global_coactivation", "eeg_theta_envcorr_left_sensorimotor_right_sensorimotor",
        "eye_eeg_frontal_theta",
    ]
    output: list[dict[str, object]] = []
    for subject, target in target_by_subject.items():
        subject_rows = {str(r["session"]): r for r in rows if str(r["subject"]) == subject}
        if "base_line" not in subject_rows or target not in subject_rows:
            continue
        base, follow = subject_rows["base_line"], subject_rows[target]
        for key in selected:
            a, b = base.get(key), follow.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                output.append({"subject": subject, "name": base["name"], "baseline_session": "base_line", "followup_session": target, "feature": key, "baseline": float(a), "followup": float(b), "absolute_change": float(b) - float(a), "percent_change": (float(b) - float(a)) / max(abs(float(a)), 1e-6) * 100.0})
    _write_rows(path, output)


def _save_event_states(out_root: Path, cache: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    """Find recurring event-response modes with an explicitly exploratory model."""
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for (subject, session), extra in cache.items():
        vectors = np.asarray(extra.get("event_delta_vectors", np.empty((0, 0))), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] < 8:
            continue
        x = robust_z(vectors, axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        pca = PCA(n_components=2, random_state=0)
        scores = pca.fit_transform(x)
        labels = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(scores)
        # Give state 1/2 a stable order by their mean PC1 score.
        means = {label: float(np.mean(scores[labels == label, 0])) for label in (0, 1)}
        if means[0] > means[1]:
            labels = 1 - labels
        silhouette = float(silhouette_score(scores, labels))
        for index, (pc, state) in enumerate(zip(scores, labels), start=1):
            rows.append({"subject": subject, "session": session, "event_index": index, "state": int(state + 1), "pc1": float(pc[0]), "pc2": float(pc[1])})
        summaries.append({"subject": subject, "session": session, "events": int(vectors.shape[0]), "pc1_variance_fraction": float(pca.explained_variance_ratio_[0]), "pc2_variance_fraction": float(pca.explained_variance_ratio_[1]), "silhouette": silhouette, "state_switches": int(np.count_nonzero(np.diff(labels))), "state2_fraction": float(np.mean(labels == 1)), "pc1_first_minus_last": float(np.mean(scores[:3, 0]) - np.mean(scores[-3:, 0]))})
        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=160)
        for state, color in [(0, "#4c78a8"), (1, "#e45756")]:
            mask = labels == state
            ax.scatter(np.flatnonzero(mask) + 1, scores[mask, 0], s=35, color=color, label=f"state {state + 1}")
        ax.plot(np.arange(1, len(scores) + 1), scores[:, 0], color="#555555", lw=0.7, alpha=0.5)
        ax.set_title(f"{SUBJECTS.get(subject, subject)} {session}: event-response states")
        ax.set_xlabel("event index")
        ax.set_ylabel("PCA score 1")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_root / f"fnirs_event_states_{subject}_{session}.png")
        plt.close(fig)
    _write_rows(out_root / "fnirs_event_states.csv", rows)
    _write_rows(out_root / "fnirs_event_state_summary.csv", summaries)


def _save_network_deltas(out_root: Path, cache: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    for subject in SUBJECTS:
        base = cache.get((subject, "base_line"))
        follow = cache.get((subject, "sec_test"))
        if base is None or follow is None:
            continue
        delta = np.asarray(follow["corr"]) - np.asarray(base["corr"])
        fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
        limit = max(0.15, float(np.nanpercentile(np.abs(delta), 98)))
        image = ax.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ticks = np.arange(0, delta.shape[0], max(1, delta.shape[0] // 11))
        ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=45)
        ax.set_yticks(ticks, [str(i + 1) for i in ticks])
        ax.set_xlabel("fNIRS channel")
        ax.set_ylabel("fNIRS channel")
        ax.set_title(f"{SUBJECTS[subject]}: second - baseline fNIRS connectivity")
        fig.colorbar(image, ax=ax, label="Pearson r change")
        fig.tight_layout()
        fig.savefig(out_root / f"fnirs_network_delta_{subject}.png")
        plt.close(fig)


def _save_emg_sensitivity(out_root: Path, cache: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    """Compare coactivation on the same logical muscles in both recordings.

    Per-recording quality gating can select different sensors.  This table
    makes that sensitivity visible instead of attributing sensor-set changes
    to physiology.
    """
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        base = cache.get((subject, "base_line"))
        follow = cache.get((subject, "sec_test"))
        if base is None or follow is None:
            continue
        base_ids = [int(value) for value in np.asarray(base.get("logical_muscle_indices", []))]
        follow_ids = [int(value) for value in np.asarray(follow.get("logical_muscle_indices", []))]
        common = sorted(set(base_ids).intersection(follow_ids))
        if len(common) < 3:
            continue
        values: dict[str, float] = {}
        for session, extra, ids in [
            ("base_line", base, base_ids),
            ("sec_test", follow, follow_ids),
        ]:
            index = [ids.index(muscle_index) for muscle_index in common]
            envelope = np.asarray(extra["envelope"])[index]
            corr = np.corrcoef(envelope)
            off_diag = corr[~np.eye(len(index), dtype=bool)]
            values[session] = float(np.mean(off_diag))
        delta = values["sec_test"] - values["base_line"]
        rows.append({
            "subject": subject,
            "name": SUBJECTS[subject],
            "baseline_session": "base_line",
            "followup_session": "sec_test",
            "common_logical_muscle_indices": ",".join(str(value) for value in common),
            "common_muscle_count": len(common),
            "baseline_physical_sensor_ids": ",".join(str(value) for value in np.asarray(base.get("physical_sensor_ids", []), dtype=int)),
            "followup_physical_sensor_ids": ",".join(str(value) for value in np.asarray(follow.get("physical_sensor_ids", []), dtype=int)),
            "baseline_mapping_profile_id": str(np.asarray(base.get("mapping_profile_id", ""))),
            "followup_mapping_profile_id": str(np.asarray(follow.get("mapping_profile_id", ""))),
            "physical_sensor_comparable": False,
            "logical_muscle_comparable": True,
            "baseline_common_sensor_coactivation": values["base_line"],
            "followup_common_sensor_coactivation": values["sec_test"],
            "absolute_change": delta,
            "percent_change": delta / max(abs(values["base_line"]), 1e-6) * 100.0,
        })
    _write_rows(out_root / "emg_common_sensor_sensitivity.csv", rows)


def _append_movement_subtask_summary(
    rows: list[dict[str, object]],
    subject: str,
    session: str,
    modality: str,
    labels: list[dict[str, object]],
    extra: dict[str, np.ndarray] | None,
) -> None:
    """Aggregate event responses after applying protocol-derived subtasks."""
    if extra is None or not labels:
        return
    used = np.asarray(extra.get("event_indices_used", []), dtype=int)
    if modality == "fNIRS":
        values = np.asarray(extra.get("event_proxy_deltas", []), dtype=float)
        left = np.asarray(extra.get("event_left_deltas", []), dtype=float)
        right = np.asarray(extra.get("event_right_deltas", []), dtype=float)
        metric = "fNIRS event response proxy"
    else:
        values = np.asarray(extra.get("event_theta_deltas", []), dtype=float)
        left = right = np.asarray([], dtype=float)
        metric = "EEG frontal theta event delta"
    if values.size == 0 or used.size != values.size:
        return
    for task_name in ("摸头", "摸膝", "摸腰", "握拳"):
        selected = [row_index for row_index, marker_index in enumerate(used) if marker_index < len(labels) and labels[marker_index]["task_name"] == task_name]
        if not selected:
            continue
        selected_values = values[selected]
        output: dict[str, object] = {
            "subject": subject,
            "name": SUBJECTS[subject],
            "session": session,
            "modality": modality,
            "task_name": task_name,
            "metric": metric,
            "event_count": int(len(selected)),
            "marker_indices": ",".join(str(int(used[i]) + 1) for i in selected),
            "mean_response_delta": float(np.nanmean(selected_values)),
            "median_response_delta": float(np.nanmedian(selected_values)),
            "sd_response_delta": float(np.nanstd(selected_values, ddof=1)) if len(selected) > 1 else 0.0,
            "inference_status": "protocol_labeled from confirmed fixed protocol order",
        }
        if modality == "fNIRS" and left.size == values.size and right.size == values.size:
            output["mean_left_right_asymmetry"] = float(np.nanmean(left[selected] - right[selected]))
        rows.append(output)


def _save_subtask_profile(out_root: Path, rows: list[dict[str, object]]) -> None:
    """Plot protocol-labeled action-specific fNIRS response profiles."""
    action_order = ["摸头", "摸膝", "摸腰"]
    subjects = list(SUBJECTS)
    fig, axes = plt.subplots(len(subjects), 2, figsize=(10, max(6.2, 2.6 * len(subjects))), dpi=160, squeeze=False)
    for row_index, subject in enumerate(subjects):
        subject_rows = [row for row in rows if row["subject"] == subject and row["modality"] == "fNIRS" and row["task_name"] in action_order]
        for col_index, value_key in enumerate(("mean_response_delta", "mean_left_right_asymmetry")):
            ax = axes[row_index, col_index]
            for session, color, offset in (("base_line", "#4c78a8", -0.18), ("sec_test", "#e45756", 0.18)):
                values = []
                for action in action_order:
                    match = next((r for r in subject_rows if r["session"] == session and r["task_name"] == action), None)
                    values.append(float(match[value_key]) if match and match.get(value_key) not in (None, "") else np.nan)
                x = np.arange(len(action_order), dtype=float) + offset
                ax.bar(x, values, width=0.34, color=color, label=session)
            ax.axhline(0, color="#555555", lw=0.7)
            ax.set_xticks(np.arange(len(action_order)), action_order)
            ax.set_title("事件响应代理" if col_index == 0 else "左右差异")
            ax.set_ylabel("fNIRS proxy")
            ax.grid(axis="y", alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(frameon=False)
        axes[row_index, 0].set_ylabel(f"{SUBJECTS[subject]}\nfNIRS proxy")
    fig.suptitle("Protocol-labeled movement subtask profiles (exploratory)")
    fig.tight_layout()
    fig.savefig(out_root / "movement_subtask_profile.png")
    plt.close(fig)


def _append_movement_emg_subtask_summary(
    rows: list[dict[str, object]],
    session: str,
    markers: list[dict[str, object]],
    extra: dict[str, np.ndarray] | None,
) -> None:
    """Summarize 5 s EMG windows using inferred EEG/fNIRS marker times."""
    if extra is None or not markers:
        return
    envelope = np.asarray(extra.get("envelope", []), dtype=float)
    time = np.asarray(extra.get("time", []), dtype=float)
    if envelope.ndim != 2 or envelope.shape[0] == 0 or time.ndim != 1 or envelope.shape[1] != time.size:
        return
    for marker in markers:
        if marker["task_name"] not in MOVEMENT_ACTION_TASKS:
            continue
        start_time = float(marker["emg_time_seconds"])
        mask = (time >= start_time) & (time < start_time + 5.0)
        if np.count_nonzero(mask) < max(20, envelope.shape[1] // 1000):
            continue
        window = envelope[:, mask]
        channel_mean = np.nanmean(window, axis=1)
        if window.shape[0] >= 2:
            corr = np.corrcoef(window)
            off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        else:
            off_diag = np.asarray([], dtype=float)
        rows.append({
            "subject": marker["subject"],
            "name": SUBJECTS[str(marker["subject"])],
            "session": session,
            "task_name": marker["task_name"],
            "block_index": marker["block_index"],
            "marker_index": marker["marker_index"],
            "reference_marker_source": marker["reference_marker_source"],
            "secondary_marker_source": marker["secondary_marker_source"],
            "emg_marker_time_seconds": marker["emg_time_seconds"],
            "emg_frame_index": marker["emg_frame_index"],
            "secondary_marker_disagreement_ms": marker["secondary_marker_disagreement_ms"],
            "window_start_seconds": start_time,
            "window_duration_seconds": float(time[mask][-1] - time[mask][0]) if np.count_nonzero(mask) > 1 else 0.0,
            "sensor_count": int(envelope.shape[0]),
            "physical_sensor_ids": ",".join(str(int(value)) for value in np.asarray(extra.get("physical_sensor_ids", []), dtype=int)),
            "logical_muscle_indices": ",".join(str(int(value)) for value in np.asarray(extra.get("logical_muscle_indices", []), dtype=int)),
            "excluded_physical_sensor_ids": ",".join(str(int(value)) for value in np.asarray(extra.get("excluded_physical_sensor_ids", []), dtype=int)),
            "mapping_profile_id": str(np.asarray(extra.get("mapping_profile_id", ""))),
            "mapping_status": str(np.asarray(extra.get("mapping_status", ""))),
            "mean_envelope": float(np.nanmean(channel_mean)),
            "mean_envelope_cv": float(np.nanstd(channel_mean) / max(abs(float(np.nanmean(channel_mean))), 1e-12)),
            "coactivation": float(np.nanmean(off_diag)) if off_diag.size else float("nan"),
            "inference_status": "EMG marker inferred from EEG/fNIRS marker; marker 1 mapped to EMG frame 0",
            "confidence": marker["confidence"],
        })


def _append_movement_bridge_summary(
    rows: list[dict[str, object]],
    subject: str,
    session: str,
    markers: list[dict[str, object]],
    fnirs_extra: dict[str, np.ndarray] | None,
    emg_extra: dict[str, np.ndarray] | None,
) -> None:
    """Pair fNIRS event responses with inferred EMG marker windows."""
    if fnirs_extra is None or emg_extra is None or not markers:
        return
    if not bool(np.asarray(emg_extra.get("mapping_valid_for_primary_features", False))):
        return
    fnirs_used = np.asarray(fnirs_extra.get("event_indices_used", []), dtype=int)
    fnirs_values = np.asarray(fnirs_extra.get("event_proxy_deltas", []), dtype=float)
    envelope = np.asarray(emg_extra.get("envelope", []), dtype=float)
    emg_time = np.asarray(emg_extra.get("time", []), dtype=float)
    if fnirs_used.size != fnirs_values.size or envelope.ndim != 2 or envelope.shape[0] == 0 or emg_time.ndim != 1 or envelope.shape[1] != emg_time.size:
        return
    fnirs_by_marker = {int(marker_index): float(value) for marker_index, value in zip(fnirs_used, fnirs_values)}
    emg_by_marker: dict[int, float] = {}
    for marker in markers:
        if marker["task_name"] not in MOVEMENT_ACTION_TASKS:
            continue
        start_time = float(marker["emg_time_seconds"])
        mask = (emg_time >= start_time) & (emg_time < start_time + 5.0)
        if np.count_nonzero(mask) >= max(20, envelope.shape[1] // 1000):
            emg_by_marker[int(marker["marker_index"]) - 1] = float(np.nanmean(envelope[:, mask]))
    for task_name in ("摸头", "摸膝", "摸腰", "握拳"):
        marker_indices = [int(marker["marker_index"]) - 1 for marker in markers if marker["task_name"] == task_name]
        paired = [(fnirs_by_marker[index], emg_by_marker[index]) for index in marker_indices if index in fnirs_by_marker and index in emg_by_marker]
        if len(paired) < 3:
            continue
        brain = np.asarray([pair[0] for pair in paired], dtype=float)
        muscle = np.asarray([pair[1] for pair in paired], dtype=float)
        rho, pvalue = stats.spearmanr(brain, muscle)
        rows.append({
            "subject": subject,
            "name": SUBJECTS[subject],
            "session": session,
            "task_name": task_name,
            "paired_blocks": int(len(paired)),
            "fNIRS_mean_response": float(np.nanmean(brain)),
            "EMG_mean_envelope": float(np.nanmean(muscle)),
            "emg_mapping_profile_id": str(np.asarray(emg_extra.get("mapping_profile_id", ""))),
            "emg_mapping_status": str(np.asarray(emg_extra.get("mapping_status", ""))),
            "emg_physical_sensor_ids": ",".join(str(int(value)) for value in np.asarray(emg_extra.get("physical_sensor_ids", []), dtype=int)),
            "emg_logical_muscle_indices": ",".join(str(int(value)) for value in np.asarray(emg_extra.get("logical_muscle_indices", []), dtype=int)),
            "emg_excluded_physical_sensor_ids": ",".join(str(int(value)) for value in np.asarray(emg_extra.get("excluded_physical_sensor_ids", []), dtype=int)),
            "spearman_rho": float(rho),
            "spearman_p_nominal": float(pvalue),
            "emg_marker_source": ";".join(sorted({str(marker["reference_marker_source"]) for marker in markers})),
            "interpretation_status": "within-session descriptive pairing; fNIRS hemodynamic window and EMG execution window are not causal evidence",
        })


def run(out_root: Path, data_root: Path | None = None) -> None:
    if data_root is not None:
        set_data_root(data_root)
    global SESSIONS
    SESSIONS, excluded_sessions = canonicalize_sessions(SESSIONS)
    out_root.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    movement_subtask_rows: list[dict[str, object]] = []
    movement_emg_rows: list[dict[str, object]] = []
    movement_emg_marker_rows: list[dict[str, object]] = []
    movement_bridge_rows: list[dict[str, object]] = []
    eye_fnirs_rows: list[dict[str, object]] = []
    fnirs_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    emg_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for subject, name in SUBJECTS.items():
        for session in SESSIONS[subject]:
            session_dir = out_root / subject / session
            session_dir.mkdir(parents=True, exist_ok=True)
            all_extras: dict[str, dict[str, np.ndarray]] = {}
            combined: dict[str, object] = {"subject": subject, "name": name, "session": session}
            motor_eeg = find_recording(subject, session, "EEG", "运动")
            motor_fnirs = find_recording(subject, session, "FNIRS", "运动")
            motor_emg = find_recording(subject, session, "EMG", "")
            evt_path = motor_eeg.parent / "evt.bdf" if motor_eeg else None
            eeg_events = read_bdf_annotation(evt_path)
            fnirs_events = read_fnirs_events(motor_fnirs)
            events, secondary_events = select_movement_event_reference(fnirs_events, eeg_events)
            movement_labels = infer_movement_event_labels(events.times, subject) if events is not None else []
            combined["movement_event_source"] = events.source if events is not None else "none"
            combined["movement_event_secondary_source"] = secondary_events.source if secondary_events is not None else "none"
            if events is not None:
                for index, onset in enumerate(events.times, start=1):
                    row = {"subject": subject, "session": session, "event_index": index, "onset_seconds": round(float(onset), 4), "source": events.source}
                    if movement_labels and index <= len(movement_labels):
                        row.update(movement_labels[index - 1])
                    event_rows.append(row)
            if motor_eeg:
                try:
                    eeg, eeg_extra = eeg_features(motor_eeg, eeg_events)
                    combined.update(eeg)
                    all_extras["eeg"] = eeg_extra
                except Exception as exc:
                    combined["eeg_error"] = str(exc)
            if motor_fnirs:
                try:
                    fnirs, fnirs_extra = fnirs_features(motor_fnirs, fnirs_events)
                    combined.update(fnirs)
                    all_extras["fnirs"] = fnirs_extra
                    fnirs_cache[(subject, session)] = fnirs_extra
                    _save_network_figure(session_dir / "fnirs_network.png", name, session, fnirs_extra["corr"], fnirs_extra["positions"])
                except Exception as exc:
                    combined["fnirs_error"] = str(exc)
            # Pair event-indexed EEG and fNIRS responses when both devices
            # carry triggers.  This is a within-session synchrony diagnostic;
            # it is not causal evidence and should be repeated across people.
            if "eeg" in all_extras and "fnirs" in all_extras:
                x = all_extras["eeg"].get("event_theta_deltas", np.asarray([]))
                y = all_extras["fnirs"].get("event_proxy_deltas", np.asarray([]))
                n = min(len(x), len(y))
                if n >= 5:
                    rho, pvalue = stats.spearmanr(np.asarray(x)[:n], np.asarray(y)[:n])
                    combined["bridge_event_spearman_rho"] = float(rho)
                    combined["bridge_event_spearman_p_nominal"] = float(pvalue)
                    combined["bridge_event_pairs"] = float(n)
            if motor_emg:
                try:
                    emg, emg_extra = emg_features(motor_emg)
                    combined.update(emg)
                    combined["emg_mapping_profile_id"] = str(np.asarray(emg_extra.get("mapping_profile_id", "")))
                    combined["emg_mapping_status"] = str(np.asarray(emg_extra.get("mapping_status", "")))
                    combined["emg_mapping_valid_for_primary_features"] = bool(np.asarray(emg_extra.get("mapping_valid_for_primary_features", False)))
                    all_extras["emg"] = emg_extra
                    emg_cache[(subject, session)] = emg_extra
                    if events is not None:
                        emg_markers = infer_emg_marker_rows(
                            subject,
                            events,
                            np.asarray(emg_extra.get("time", []), dtype=float),
                            float(np.asarray(emg_extra.get("fs", [0.0]), dtype=float)[0]),
                            secondary_events,
                        )
                        for marker in emg_markers:
                            marker["session"] = session
                        movement_emg_marker_rows.extend(emg_markers)
                        _append_movement_emg_subtask_summary(movement_emg_rows, session, emg_markers, emg_extra)
                except Exception as exc:
                    combined["emg_error"] = str(exc)
            # Eye-movement EEG supplies the cognitive-control side of the
            # bridge. Eye fNIRS is retained below as a separately labelled,
            # infrared-confounded proxy branch for sensitivity exploration.
            eye_eeg = find_recording(subject, session, "EEG", "眼动")
            if eye_eeg:
                try:
                    eye, _ = eeg_features(eye_eeg, None)
                    combined.update({f"eye_{key}": value for key, value in eye.items()})
                except Exception as exc:
                    combined["eye_eeg_error"] = str(exc)
            eye_fnirs = find_recording(subject, session, "FNIRS", "眼动")
            if eye_fnirs:
                try:
                    eye_events = read_fnirs_events(eye_fnirs)
                    eye, eye_extra = eye_fnirs_features(eye_fnirs, eye_events)
                    combined.update(eye)
                    eye_fnirs_rows.append({"subject": subject, "name": name, "session": session, "path": str(eye_fnirs), **eye})
                    all_extras["eye_fnirs"] = eye_extra
                except Exception as exc:
                    combined["eye_fnirs_error"] = str(exc)
            # Resting fNIRS gives a complementary low-frequency network state.
            # It is kept separate from task-evoked activity to avoid conflating
            # baseline connectivity with a motor response.
            rest_fnirs = find_recording(subject, session, "FNIRS", "静息态")
            if rest_fnirs:
                try:
                    rest, _ = fnirs_features(rest_fnirs, None)
                    combined.update({f"rest_{key}": value for key, value in rest.items()})
                except Exception as exc:
                    combined["rest_fnirs_error"] = str(exc)
            if motor_fnirs and fnirs_events is not None and "fnirs" in all_extras:
                _append_movement_subtask_summary(movement_subtask_rows, subject, session, "fNIRS", infer_movement_event_labels(fnirs_events.times, subject), all_extras["fnirs"])
            if motor_eeg and eeg_events is not None and "eeg" in all_extras:
                _append_movement_subtask_summary(movement_subtask_rows, subject, session, "EEG", infer_movement_event_labels(eeg_events.times, subject), all_extras["eeg"])
            if motor_fnirs and fnirs_events is not None and "fnirs" in all_extras and "emg" in all_extras:
                emg_markers = [row for row in movement_emg_marker_rows if row["subject"] == subject and row.get("session", session) == session]
                _append_movement_bridge_summary(movement_bridge_rows, subject, session, emg_markers, all_extras["fnirs"], all_extras["emg"])
            _save_timeline(session_dir / "multimodal_timeline.png", name, session, all_extras.get("eeg"), all_extras.get("fnirs"), all_extras.get("emg"), events)
            feature_rows.append(combined)
    _write_rows(out_root / "motor_multimodal_features.csv", feature_rows)
    _write_rows(out_root / "motor_event_markers.csv", event_rows)
    _write_rows(out_root / "movement_subtask_summary.csv", movement_subtask_rows)
    _write_rows(out_root / "movement_subtask_emg_summary.csv", movement_emg_rows)
    _write_rows(out_root / "movement_emg_markers.csv", movement_emg_marker_rows)
    _write_rows(out_root / "movement_subtask_bridge_summary.csv", movement_bridge_rows)
    _write_rows(out_root / "eye_fnirs_features.csv", eye_fnirs_rows)
    _save_subtask_profile(out_root, movement_subtask_rows)

    # A compact, deliberately descriptive baseline -> second-test comparison.
    comparison_rows: list[dict[str, object]] = []
    for row in feature_rows:
        if row["session"] not in {"base_line", "sec", "sec_test"}:
            continue
        comparison_rows.append({"subject": row["subject"], "name": row["name"], "session": row["session"], **{k: v for k, v in row.items() if k.startswith(("eeg_", "fnirs_", "emg_", "eye_", "rest_")) and not k.startswith("emg_sensor_") and isinstance(v, (int, float))}})
    _write_rows(out_root / "motor_feature_comparison.csv", comparison_rows)
    _write_delta_summary(out_root / "case_delta_summary.csv", comparison_rows)
    _save_delta_heatmap(out_root / "case_delta_heatmap.png", comparison_rows)
    _save_event_states(out_root, fnirs_cache)
    _save_network_deltas(out_root, fnirs_cache)
    _save_emg_sensitivity(out_root, emg_cache)
    paired_subjects = sorted(
        subject for subject, sessions in SESSIONS.items()
        if "base_line" in sessions and "sec_test" in sessions
    )
    metadata = {
        "scope": f"{len(SUBJECTS)} subjects; {len(paired_subjects)} paired baseline/sec_test subjects",
        "raw_root": str(RAW_ROOT),
        "sessions": SESSIONS,
        "excluded_duplicate_sessions": excluded_sessions,
        "subject_count": len(SUBJECTS),
        "paired_subjects": paired_subjects,
        "caution": f"{len(paired_subjects)} paired subjects are the independent longitudinal units; windows/events are descriptive within-session units, not independent subjects.",
        "fNIRS_proxy": "wavelength-balanced optical-density proxy; no DPF/extinction coefficients; probe left/right orientation must be verified.",
        "EMG": "All EMG recordings belong to the movement task. EMG marker times are inferred from the selected observed EEG/fNIRS movement marker stream with marker 1 mapped to EMG frame 0; secondary-stream disagreement is retained in movement_emg_markers.csv.",
        "EEG": "event-locked features use companion evt.bdf description=1 when present.",
        "movement_subtasks": "Automatic movement marker timestamps are treated as accurate. Task names follow the confirmed fixed protocol order: 摸头, 摸膝, 摸腰; 2026081401 and 2026081403 baseline markers after the first 20 are labeled 握拳. Labels are protocol_labeled, not encoded by marker value.",
        "movement_bridge": "fNIRS event response windows are paired to 5 s EMG execution windows by automatic marker time and fixed-protocol task label; pairing is descriptive and not causal.",
        "eye_fnirs": "眼动 fNIRS 作为独立降级敏感性分支：仅输出 optical-density proxy、质量状态、红外干扰/遮挡声明与 analysis_sample_mask 覆盖率；不转换 HbO/HbR，不从未知语义 marker 推断任务标签。",
    }
    (out_root / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=None, help="raw-data root (default: historical J: path)")
    args = parser.parse_args()
    run(args.out, data_root=args.data_root)
    print(f"derived outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main_cli()
