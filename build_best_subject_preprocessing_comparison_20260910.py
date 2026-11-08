"""Build transparent before/after preprocessing visuals for one movement visit.

The visit is selected objectively from the fixed seven-participant local
dataset, using only completeness and signal-quality evidence already produced
by the verified preprocessing stage.  This is a teaching/quality-control
artifact: it never makes a treatment or mechanism claim.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat

import fnirs_qc
import main as eeg_io
from emg_qc import read_emg_csv


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_repro_20260903"
STAGE = ROOT / "outputs" / "preprocessed" / "stage_20260904_eye_fnirs_imu_verified"
EEG_QC = ROOT / "outputs" / "segment_eeg_quality_paired7_20260903" / "task_segment_summary.csv"
MBLL_AUDIT = ROOT / "outputs" / "mbll_reviewed_20260908" / "movement_marker_audit.csv"
OUT = ROOT / "outputs" / "best_subject_preprocessing_comparison_20260910"

SENSOR_RE = re.compile(r"Avanti sensor\s+(\d+):\s*(EMG|Acc|Gyro|Mag)\s+\d+", re.I)


def parse_header(path: Path) -> dict[int, dict[str, list[tuple[int, int, str]]]]:
    """CSV header -> async timestamp/value columns; dependency-free equivalent."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    result: dict[int, dict[str, list[tuple[int, int, str]]]] = {}
    for value_col, label in enumerate(header):
        match = SENSOR_RE.search(label or "")
        if not match or value_col == 0:
            continue
        sensor = int(match.group(1)); stream = {"emg":"EMG","acc":"Acc","gyro":"Gyro","mag":"Mag"}[match.group(2).lower()]
        result.setdefault(sensor, {}).setdefault(stream, []).append((value_col-1, value_col, label.strip()))
    return result


def _stream_arrays(path: Path, columns: list[tuple[int, int, str]]) -> np.ndarray:
    """Read only requested async time/value pairs, without pandas dependency."""
    indices = [i for pair in columns for i in pair[:2]]
    out: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader=csv.reader(f); next(reader)
        for row in reader:
            values=[]
            for index in indices:
                try: values.append(float(row[index])) if index < len(row) and row[index] else values.append(float("nan"))
                except ValueError: values.append(float("nan"))
            out.append(values)
    return np.asarray(out,dtype=float)


@dataclass
class EventSeries:
    times: np.ndarray
    source: str


def infer_movement_event_labels(times: np.ndarray, subject: str) -> list[dict[str, Any]]:
    """Assign tasks only from the verified fixed protocol order, never value 1."""
    actions = {3:("摸头",1),4:("摸头",2),5:("摸头",3),6:("摸头",4),7:("摸头",5),
               9:("摸膝",1),10:("摸膝",2),11:("摸膝",3),12:("摸膝",4),13:("摸膝",5),
               15:("摸腰",1),16:("摸腰",2),17:("摸腰",3),18:("摸腰",4),19:("摸腰",5)}
    result=[]
    for index,onset in enumerate(np.asarray(times,dtype=float),start=1):
        if index in actions:
            task,block=actions[index]; role="action_block_start"
        elif index == 1:
            task,block,role="运动任务同步起点","","sync_start"
        elif index == 2:
            task,block,role="首个动作前准备/边界","","pre_first_action_boundary"
        elif index in {8,14,20}:
            task,block,role="动作转换/结束边界","","boundary"
        else:
            task,block,role="未分类/异常标记","","unclassified"
        result.append({"marker_index":index,"task_name":task,"block_index":block,"event_role":role,"timestamp_seconds":float(onset)})
    return result


def read_fnirs_events(path: Path) -> EventSeries | None:
    if not path.exists():
        return None
    doc=loadmat(path,variable_names=["t","s"],simplify_cells=True)
    if "t" not in doc or "s" not in doc:
        return None
    time=np.ravel(np.asarray(doc["t"],dtype=float)); marks=np.asarray(doc["s"])
    indices=np.flatnonzero(marks!=0) if marks.ndim==1 else np.flatnonzero(np.any(marks!=0,axis=1))
    if not indices.size: return None
    kept=[float(time[indices[0]])]
    for value in time[indices[1:]]:
        if float(value)-kept[-1]>1.0: kept.append(float(value))
        elif float(value)<kept[-1]: kept[-1]=float(value)
    return EventSeries(np.asarray(kept),"fNIRS s marker (deduplicated)")


def read_bdf_annotation(path: Path) -> EventSeries | None:
    """Minimal companion-BDF event reader; only nonempty descriptions survive."""
    if not path.exists(): return None
    with path.open("rb") as f:
        fixed=f.read(256); header_bytes=int(fixed[184:192].decode("ascii").strip()); count=int(fixed[236:244].decode("ascii").strip()); nchan=int(fixed[252:256].decode("ascii").strip()); ext=f.read(header_bytes-256)
    widths=(16,80,8,8,8,8,8,80,8,32); fields=[]; cursor=0
    for width in widths:
        fields.append([ext[cursor+i*width:cursor+(i+1)*width].decode("latin-1",errors="replace").strip() for i in range(nchan)]); cursor+=width*nchan
    ann=[i for i,label in enumerate(fields[0]) if "annotation" in label.lower()]
    if not ann: return None
    samples=[int(x) for x in fields[8]]; idx=ann[0]; record_bytes=path.read_bytes()[header_bytes:]; width=3*sum(samples); chunks=[]
    for rec in range(count):
        raw=record_bytes[rec*width:(rec+1)*width]; offset=3*sum(samples[:idx]); chunks.append(raw[offset:offset+3*samples[idx]])
    stream=b"".join(chunks).replace(b"\x00",b"")
    matches=re.finditer(rb"\+([0-9.]+)(?:\x15[^\x14]*)?\x14([^\x14\x15]*)",stream)
    times=[float(m.group(1)) for m in matches if m.group(2).strip()]
    return EventSeries(np.asarray(times),"EEG evt.bdf description=1") if times else None


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 250,
})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def teacher_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Teacher-facing ranking with no raw filename, name, or birth-date field.

    The audit table is preserved separately because it is needed to trace an
    analysis.  This reduced table is the only selection table intended for a
    slide or teacher report.
    """
    subjects = {subject: index + 1 for index, subject in enumerate(sorted({str(r["subject"]) for r in rows}))}
    keep = (
        "subject", "session", "eligible_complete_multimodal", "selection_quality_score_0_100",
        "core_movement_action_markers", "eeg_protocol_complete", "eeg_event_alignment",
        "eeg_execution_broad_artifact_fraction", "eeg_execution_bad_channel_fraction",
        "fnirs_kept_channels", "fnirs_analysis_sample_percent", "emg_mapping_valid",
        "emg_sensor_count", "emg_mean_quality_score", "imu_acc_gyro_complete", "imu_complete_sensor_count",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        selected = {key: row.get(key, "") for key in keep}
        selected["anonymous_relative_record_id"] = f"participant_{subjects[str(row['subject'])]:02d}_{str(row['session']).replace('base_line', 'baseline').replace('sec_test', 'followup')}"
        output.append(selected)
    return output


def stage_jsons(subject: str, session: str, modality: str) -> list[tuple[Path, dict[str, Any]]]:
    base = STAGE / subject / session / modality
    result = []
    for path in base.rglob("preprocessing_summary.json"):
        try:
            result.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return result


def select_summary(subject: str, session: str, modality: str, task_token: str = "运动") -> tuple[Path, dict[str, Any]] | None:
    candidates = [(p, j) for p, j in stage_jsons(subject, session, modality) if task_token in str(j.get("task", ""))]
    if not candidates:
        return None
    # The verified MBLL branch used the lexicographically first stage-approved
    # source.  Keep that same convention here; raw filename is not a task label.
    return sorted(candidates, key=lambda item: str(item[1].get("source_relative_path", item[0])))[0]


def visit_eeg_qc() -> dict[tuple[str, str], dict[str, Any]]:
    rows = list(csv.DictReader(EEG_QC.open(encoding="utf-8-sig", newline="")))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subject"], row["session"]), []).append(row)
    for (subject, session), group in grouped.items():
        primary = [row for row in group if row.get("task") in {"摸头", "摸膝", "摸腰"}]
        if len(primary) != 3:
            continue
        def mean(name: str) -> float:
            values = []
            for row in primary:
                try:
                    value = float(row.get(name, ""))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            return float(np.mean(values)) if values else float("nan")
        result[(str(subject), str(session))] = {
            "eeg_execution_broad_artifact_fraction": mean("execution_broad_bad_epoch_fraction_mean"),
            "eeg_execution_bad_channel_fraction": mean("execution_bad_channel_fraction_mean"),
            "eeg_protocol_complete": str(primary[0]["visit_protocol_complete"]).lower() == "true",
            "eeg_event_alignment": str(primary[0]["event_alignment"]),
            "eeg_visit_status": str(primary[0]["visit_status"]),
        }
    return result


def core_marker_counts() -> dict[tuple[str, str], int]:
    rows = list(csv.DictReader(MBLL_AUDIT.open(encoding="utf-8-sig", newline="")))
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["subject"], row["session"])
        if row.get("analysis_status") == "included_core_protocol_action":
            counts[key] = counts.get(key, 0) + 1
    return counts


def imu_coverage(csv_path: Path, sensors: list[int]) -> tuple[bool, int]:
    try:
        fields = parse_header(csv_path)
    except Exception:
        return False, 0
    complete = 0
    for sensor in sensors:
        streams = fields.get(int(sensor), {})
        if "Acc" in streams and "Gyro" in streams:
            complete += 1
    return complete == len(sensors), complete


def make_selection() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    eeg = visit_eeg_qc()
    markers = core_marker_counts()
    all_rows: list[dict[str, Any]] = []
    descriptors: dict[str, tuple[Path, dict[str, Any]]] = {}
    for subject_path in sorted(STAGE.iterdir()):
        if not subject_path.is_dir() or not re.fullmatch(r"\d{10}", subject_path.name):
            continue
        subject = subject_path.name
        for session_path in sorted(subject_path.iterdir()):
            if not session_path.is_dir() or session_path.name not in {"base_line", "sec_test"}:
                continue
            session = session_path.name
            eeg_info = select_summary(subject, session, "EEG")
            fnirs_info = select_summary(subject, session, "FNIRS")
            emg_info = select_summary(subject, session, "EMG")
            key = (subject, session)
            q = eeg.get(key, {})
            row: dict[str, Any] = {
                "subject": subject, "session": session,
                "eeg_available": eeg_info is not None,
                "fnirs_available": fnirs_info is not None,
                "emg_available": emg_info is not None,
                "core_movement_action_markers": markers.get(key, 0),
                **q,
            }
            if fnirs_info:
                _, f = fnirs_info
                ms = f.get("mask_summary", {})
                row.update(
                    fnirs_kept_channels=int(ms.get("kept_channel_count", 0)),
                    fnirs_analysis_sample_percent=float(ms.get("mean_analysis_sample_percent", 0.0)),
                    fnirs_source_relative_path=f.get("source_relative_path", ""),
                )
            else:
                row.update(fnirs_kept_channels=0, fnirs_analysis_sample_percent=0.0, fnirs_source_relative_path="")
            if emg_info:
                _, e = emg_info
                sensors = [int(x) for x in e.get("sensor_ids", [])]
                raw_csv = RAW / str(e.get("source_relative_path", ""))
                imu_ok, imu_count = imu_coverage(raw_csv, sensors) if raw_csv.exists() else (False, 0)
                scores = [float(x) for x in dict(e.get("channel_quality_scores", {})).values()]
                row.update(
                    emg_mapping_valid=bool(e.get("mapping_valid_for_primary_features", False)),
                    emg_sensor_count=len(sensors),
                    emg_mean_quality_score=float(np.mean(scores)) if scores else 0.0,
                    imu_acc_gyro_complete=imu_ok, imu_complete_sensor_count=imu_count,
                    emg_source_relative_path=e.get("source_relative_path", ""),
                )
            else:
                row.update(emg_mapping_valid=False, emg_sensor_count=0, emg_mean_quality_score=0.0,
                           imu_acc_gyro_complete=False, imu_complete_sensor_count=0, emg_source_relative_path="")
            required = bool(row["eeg_available"] and row["fnirs_available"] and row["emg_available"]
                            and row.get("eeg_protocol_complete", False) and row["core_movement_action_markers"] == 15
                            and row["emg_mapping_valid"] and row["emg_sensor_count"] == 7 and row["imu_acc_gyro_complete"])
            # Quality score in [0, 100]. Completeness is an eligibility gate,
            # rather than a component that can be traded against poor signals.
            score = (
                0.30 * max(0.0, 1.0 - float(row.get("eeg_execution_broad_artifact_fraction", 1.0)))
                + 0.20 * max(0.0, 1.0 - float(row.get("eeg_execution_bad_channel_fraction", 1.0)))
                + 0.25 * min(1.0, float(row["fnirs_kept_channels"]) / 44.0)
                + 0.15 * min(1.0, float(row["fnirs_analysis_sample_percent"]) / 100.0)
                + 0.10 * min(1.0, float(row["emg_mean_quality_score"]) / 100.0)
            )
            row["eligible_complete_multimodal"] = required
            row["selection_quality_score_0_100"] = round(100.0 * score, 3)
            row["selection_note"] = "eligible" if required else "not selected: incomplete protocol, missing modality, unresolved EMG mapping, or incomplete IMU fields"
            all_rows.append(row)
            if eeg_info:
                descriptors[f"{subject}/{session}/EEG"] = eeg_info
            if fnirs_info:
                descriptors[f"{subject}/{session}/FNIRS"] = fnirs_info
            if emg_info:
                descriptors[f"{subject}/{session}/EMG"] = emg_info
    eligible = [r for r in all_rows if r["eligible_complete_multimodal"]]
    if not eligible:
        raise RuntimeError("No complete multimodal visit met the predefined selection criteria.")
    winner = sorted(eligible, key=lambda r: (-float(r["selection_quality_score_0_100"]), r["subject"], r["session"]))[0]
    return all_rows, winner, descriptors


def action_window(recording: Path, subject: str) -> tuple[float, str, int]:
    events = read_bdf_annotation(recording.parent / "evt.bdf")
    if events is None:
        # This is not a marker-value interpretation: it uses the already
        # verified fallback convention only if an EEG companion event file is absent.
        candidates = sorted((recording.parents[2] / "FNIRS").rglob("*.nirs"))
        events = read_fnirs_events(candidates[0]) if candidates else None
    if events is None:
        raise RuntimeError(f"No event stream for {recording}")
    labels = infer_movement_event_labels(events.times, subject)
    first = next(x for x in labels if x["task_name"] == "摸头")
    return float(first["timestamp_seconds"]), "摸头（第 1 次）", int(first["marker_index"])


def band_envelope(values: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, values)
    env = np.abs(signal.hilbert(filtered))
    win = max(3, int(round(fs * 0.5)))
    return signal.savgol_filter(env, win + 1 if win % 2 == 0 else win, 2, mode="interp")


def shade_action(ax: plt.Axes) -> None:
    ax.axvspan(0, 5, color="#D55E00", alpha=0.13, label="动作：摸头（0–5 s）")
    ax.axvspan(5, 20, color="#56B4E9", alpha=0.10, label="动作后静息（5–20 s）")
    ax.axvline(0, color="#D55E00", lw=1.1)
    ax.axvline(5, color="#D55E00", lw=0.9, ls="--")


def save(fig: plt.Figure, name: str) -> str:
    png = OUT / f"{name}.png"
    svg = OUT / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png.relative_to(OUT))


def plot_selection(rows: list[dict[str, Any]], winner: dict[str, Any]) -> str:
    eligible = [r for r in rows if r["eligible_complete_multimodal"]]
    eligible = sorted(eligible, key=lambda r: float(r["selection_quality_score_0_100"]))
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    labels = [f"{r['subject']}\n{r['session']}" for r in eligible]
    vals = [r["selection_quality_score_0_100"] for r in eligible]
    colors = ["#0072B2" if (r["subject"], r["session"]) == (winner["subject"], winner["session"]) else "#9AA0A6" for r in eligible]
    bars = ax.barh(np.arange(len(eligible)), vals, color=colors)
    ax.set_yticks(np.arange(len(eligible)), labels)
    ax.set_xlim(0, 100); ax.set_xlabel("综合数据质量分（0–100，完整性为准入条件）")
    ax.set_title("代表性个例的客观选择：仅在完整的 EEG + fNIRS + EMG + IMU 访视中比较")
    ax.grid(axis="x", alpha=0.2)
    for b, r in zip(bars, eligible):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height()/2,
                f"{b.get_width():.1f} | EEG广泛伪迹 {100*r['eeg_execution_broad_artifact_fraction']:.1f}% | fNIRS {r['fnirs_kept_channels']}/44", va="center", fontsize=8)
    ax.text(0.01, -0.19, "评分由 EEG 动作期广泛伪迹与坏导联比例、fNIRS 保留通道/有效样本比例、EMG 质量分组成；\n未满足 15 个核心动作标记、完整 EEG 方案、7 块标准肌肉映射或 Acc+Gyro 完整者不进入比较。", transform=ax.transAxes, fontsize=8.6)
    return save(fig, "01_subject_selection")


def process_eeg(subject: str, session: str, summary: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    raw_path = RAW / str(summary["source_relative_path"])
    npz_path = STAGE / subject / session / "EEG" / "运动" / "eeg_cleaned.npz"
    clean = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in clean["labels"]]
    fs_clean = float(clean["sample_rate_hz"])
    bad = np.asarray(clean["bad_channel_mask"], dtype=bool)
    preferred = ["C3", "C4", "Cz", "FC3", "FC4"]
    channel = next((name for name in preferred if name in labels and not bad[labels.index(name)]), None)
    if channel is None:
        channel = labels[int(np.flatnonzero(~bad)[0])]
    ci = labels.index(channel)
    header, raw_all, _ = eeg_io.read_bdf(raw_path, None)
    lookup = {eeg_io.normalize_label(x): i for i, x in enumerate(header.labels)}
    raw_i = lookup[eeg_io.normalize_label(channel)]
    raw = raw_all[raw_i].astype(float)
    fs_raw = float(header.sample_rate)
    cleaned = np.asarray(clean["data_uv"], dtype=float)[ci]
    onset, task, _ = action_window(raw_path, subject)
    left, right = max(0.0, onset - 5.0), min((len(cleaned)-1)/fs_clean, onset + 20.0)
    iraw = slice(int(round(left*fs_raw)), int(round(right*fs_raw)))
    iclean = slice(int(round(left*fs_clean)), int(round(right*fs_clean)))
    tx = np.arange(iraw.start, iraw.stop) / fs_raw - onset
    ty = np.arange(iclean.start, iclean.stop) / fs_clean - onset
    # Thin raw curves for readability, without altering the source data.
    thin = max(1, int(fs_raw/250))
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(tx[::thin], raw[iraw][::thin], color="#9E9E9E", lw=0.55, label="原始单导联（未重参考/未滤波）")
    axes[0].set_ylabel("电位（µV）")
    axes[0].set_title(f"EEG：{channel}，{task}对齐的 25 秒片段——处理前")
    axes[1].plot(ty, cleaned[iclean], color="#0072B2", lw=0.7, label="处理后：重参考 + 50 Hz 陷波 + 0.5–45 Hz 带通")
    axes[1].set_ylabel("电位（µV）")
    axes[1].set_title("处理后波形：异常导联已标记；无效 2 秒窗不进入统计")
    for ax in axes:
        shade_action(ax); ax.grid(alpha=0.18); ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("相对动作开始时间（s）")
    axes[0].text(0.01, 0.04, "波峰、波谷是该导联的瞬时电位起伏；不是“好/坏疗效”的直接证据。", transform=axes[0].transAxes, fontsize=8.6,
                 bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=.92))
    files = [save(fig, "02_eeg_time_domain_before_after")]
    # Bands figure, in the same cleaned source and identical time segment.
    segment = cleaned[iclean]
    tseg = ty
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 6.9), constrained_layout=True)
    colors = {"θ 4–8 Hz": "#CC79A7", "α 8–13 Hz": "#009E73", "β 13–30 Hz": "#E69F00"}
    envelopes: dict[str, np.ndarray] = {}
    for name, band in [("θ 4–8 Hz", (4,8)), ("α 8–13 Hz", (8,13)), ("β 13–30 Hz", (13,30))]:
        envelopes[name] = band_envelope(segment, fs_clean, *band)
        axes[0].plot(tseg, envelopes[name], color=colors[name], lw=1.1, label=name)
    axes[0].set_ylabel("Hilbert 包络（µV）")
    axes[0].set_title(f"处理后 EEG 的节律强度：{channel}（包络越高，表示该频段瞬时振幅更大）")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    a = segment[(tseg >= 0) & (tseg < 5)]
    r = segment[(tseg >= 5) & (tseg < 20)]
    f1, p1 = signal.welch(a, fs_clean, nperseg=min(len(a), int(fs_clean*2)))
    f2, p2 = signal.welch(r, fs_clean, nperseg=min(len(r), int(fs_clean*2)))
    axes[1].semilogy(f1[(f1>=1)&(f1<=40)], p1[(f1>=1)&(f1<=40)], color="#D55E00", lw=1.1, label="动作期 0–5 s")
    axes[1].semilogy(f2[(f2>=1)&(f2<=40)], p2[(f2>=1)&(f2<=40)], color="#56B4E9", lw=1.1, label="动作后静息 5–20 s")
    for low, high, name in [(4,8,"θ"),(8,13,"α"),(13,30,"β")]:
        axes[1].axvspan(low, high, color=colors[f"{name} {low}–{high} Hz"], alpha=.10)
        axes[1].text((low+high)/2, max(np.nanmax(p1),np.nanmax(p2))*0.6, name, ha="center", fontsize=9)
    axes[1].set(xlabel="频率（Hz）", ylabel="功率谱密度（µV²/Hz）", title="同一片段的频谱：峰值表示该频率附近的振荡能量更集中")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.18)
    shade_action(axes[0])
    axes[0].grid(alpha=.18)
    axes[1].grid(alpha=.18)
    axes[0].text(0.01, .04, "α、β的“高低”需在预先规定的任务/静息或治疗前后比较中解释；单张个例图不能证明功能改善。", transform=axes[0].transAxes, fontsize=8.5,
                 bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=.92))
    files.append(save(fig, "03_eeg_theta_alpha_beta"))
    metrics = [
        {"modality":"EEG", "measure":"representative_channel", "value":channel, "unit":"label"},
        {"modality":"EEG", "measure":"source_sample_rate", "value":fs_raw, "unit":"Hz"},
        {"modality":"EEG", "measure":"processed_sample_rate", "value":fs_clean, "unit":"Hz"},
        {"modality":"EEG", "measure":"line_noise_median_before", "value":summary.get("line_noise_median_before_uv"), "unit":"µV"},
        {"modality":"EEG", "measure":"line_noise_median_after", "value":summary.get("line_noise_median_after_uv"), "unit":"µV"},
        {"modality":"EEG", "measure":"valid_2s_epoch_count", "value":summary.get("valid_2s_epochs"), "unit":"epochs"},
        {"modality":"EEG", "measure":"total_2s_epoch_count", "value":summary.get("total_2s_epochs"), "unit":"epochs"},
    ]
    return files, metrics


def process_fnirs(subject: str, session: str, summary: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    raw_path = RAW / str(summary["source_relative_path"])
    stage_dir = next(path.parent for path, doc in stage_jsons(subject, session, "FNIRS") if doc.get("source_relative_path") == summary["source_relative_path"])
    npz_path = stage_dir / "fnirs_cleaned_od.npz"
    clean = np.load(npz_path, allow_pickle=True)
    rec = fnirs_qc.read_fnirs(raw_path, None)
    labels = [str(x) for x in clean["labels"]]
    mask = np.asarray(clean["analysis_sample_mask"], dtype=bool)
    # Use the channel with highest usable share (tie -> lower label), thereby
    # avoiding manual selection of an unusually dramatic waveform.
    usable = mask.mean(axis=1)
    ci = int(np.argmax(usable)); label = labels[ci]
    raw_channel = next(ch for ch in rec.channels if ch.label == label)
    time = np.asarray(clean["time_seconds"], dtype=float)
    od = np.asarray(clean["optical_density"], dtype=float)[ci]
    onset_events = read_fnirs_events(raw_path)
    onset_label = next(x for x in infer_movement_event_labels(onset_events.times, subject) if x["task_name"] == "摸头")
    onset = float(onset_label["timestamp_seconds"] - rec.time[0])
    left, right = max(0.0, onset - 5), min(float(time[-1]), onset+20)
    sel = (time>=left)&(time<=right)
    rel_t = time[sel]-onset
    raw_a = np.asarray(raw_channel.intensity_a)[sel]
    raw_b = np.asarray(raw_channel.intensity_b)[sel]
    # Relative intensity is more legible than arbitrary raw detector scale,
    # but it is still raw intensity: no physiological concentration claim.
    raw_a_pct = 100*(raw_a/np.nanmedian(raw_a)-1)
    raw_b_pct = 100*(raw_b/np.nanmedian(raw_b)-1)
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 8.9), sharex=True, constrained_layout=True)
    axes[0].plot(rel_t, raw_a_pct, color="#7A5195", lw=.9, label="730 nm 原始强度（片段中位数归一化）")
    axes[0].plot(rel_t, raw_b_pct, color="#EF5675", lw=.9, label="850 nm 原始强度（片段中位数归一化）")
    axes[0].set(ylabel="相对原始强度（%）", title=f"fNIRS：{label}，{onset_label['task_name']}第1次——原始双波长光强")
    axes[1].plot(rel_t, od[0,sel], color="#7A5195", lw=.95, label="730 nm：修复突变 + 0.01–0.2 Hz")
    axes[1].plot(rel_t, od[1,sel], color="#EF5675", lw=.95, label="850 nm：修复突变 + 0.01–0.2 Hz")
    axes[1].set_ylabel("光密度 OD（a.u.）")
    axes[1].set_title("处理后：只保留通过通道与样本质量筛选的数据")
    # Relative MBLL using the staged, filtered natural OD values. The source
    # filename provides age only as unverified metadata; flag it in legend.
    name_dates = __import__("mbll_relative_sensitivity_reviewed_20260908").parse_filename_dates(raw_path)
    mbll = __import__("mbll_relative_sensitivity_reviewed_20260908")
    dpf = mbll.dpf_values("age_formula", float(name_dates["age_years"]))
    distance = np.linalg.norm(rec.source_positions[raw_channel.source-1] - rec.detector_positions[raw_channel.detector-1])
    concentrations, cond = mbll.unmix_natural_od(np.moveaxis(od[:,sel], 0, 1)[...,None], dpf, np.asarray([distance]))
    hbo, hbr = concentrations[:,0,0], concentrations[:,1,0]
    hbo -= np.nanmedian(hbo[rel_t<0]); hbr -= np.nanmedian(hbr[rel_t<0])
    axes[2].plot(rel_t, hbo, color="#D55E00", lw=1.15, label="相对 ΔHbO")
    axes[2].plot(rel_t, hbr, color="#0072B2", lw=1.15, label="相对 ΔHbR")
    axes[2].axhline(0, color="#555", lw=.7)
    axes[2].set(ylabel="相对浓度变化（µM）", title="相对 MBLL 换算：用于展示，不是绝对血红蛋白浓度")
    for ax in axes:
        shade_action(ax); ax.grid(alpha=.18); ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("相对动作开始时间（s）")
    axes[2].text(.01, .04, "上升/下降只代表该通道在所选基线下的相对估计变化；不能单独等同于脑区激活或疗效。", transform=axes[2].transAxes, fontsize=8.3,
                 bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=.92))
    return [save(fig, "04_fnirs_intensity_od_mbll")], [
        {"modality":"fNIRS", "measure":"representative_channel", "value":label, "unit":"label"},
        {"modality":"fNIRS", "measure":"kept_channels", "value":summary["mask_summary"]["kept_channel_count"], "unit":"channels"},
        {"modality":"fNIRS", "measure":"analysis_samples", "value":summary["mask_summary"]["mean_analysis_sample_percent"], "unit":"percent"},
        {"modality":"fNIRS", "measure":"source_detector_distance", "value":distance, "unit":"cm (coordinate-derived)"},
        {"modality":"fNIRS", "measure":"mbll_design_condition", "value":cond, "unit":"condition number"},
        {"modality":"fNIRS", "measure":"dpf_730", "value":dpf[0], "unit":"unitless"},
        {"modality":"fNIRS", "measure":"dpf_850", "value":dpf[1], "unit":"unitless"},
    ]


def process_emg_imu(subject: str, session: str, summary: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    raw_path = RAW / str(summary["source_relative_path"])
    series = read_emg_csv(raw_path, None)
    # Sensor 3 (anterior deltoid) is chosen programmatically by best quality
    # score; it remains a plotted example, not a claim of prime mover.
    stage_npz = next((STAGE / subject / session / "EMG").rglob("emg_cleaned.npz"))
    clean = np.load(stage_npz, allow_pickle=True)
    quality = np.asarray(clean["channel_quality_score"], dtype=float)
    ci = int(np.argmax(quality))
    sensor_id = int(clean["sensor_ids"][ci]); muscle = str(clean["logical_muscle_labels"][ci])
    raw_series = next(x for x in series if x.sensor_id == sensor_id)
    fs_raw = float(raw_series.sample_rate_hz)
    data = np.asarray(clean["data_source_unit"], dtype=float)[ci]
    fs = float(clean["sample_rate_hz"])
    # EMG has no trustworthy hardware task marker, so select a fixed
    # mid-recording window by time rather than falsely aligning it to EEG/fNIRS.
    center = min(120.0, float(clean["time_seconds"][-1]) / 2)
    left, right = max(0., center-3.), min(float(clean["time_seconds"][-1]), center+3.)
    rsel = (raw_series.time-raw_series.time[0]>=left)&(raw_series.time-raw_series.time[0]<=right)
    ctime = np.asarray(clean["time_seconds"], dtype=float)
    csel=(ctime>=left)&(ctime<=right)
    rect = np.abs(data)
    sos_lp = signal.butter(4, 5.0, btype="lowpass", fs=fs, output="sos")
    envelope = signal.sosfiltfilt(sos_lp, rect)
    fig, axes = plt.subplots(2,1,figsize=(12.2,6.6),sharex=True,constrained_layout=True)
    axes[0].plot((raw_series.time-raw_series.time[0])[rsel], raw_series.values[rsel], lw=.4,color="#9E9E9E",label="原始 sEMG（源文件单位）")
    axes[0].set(ylabel="幅值（源单位）", title=f"sEMG：传感器 {sensor_id} / {muscle}——处理前（6 秒示例片段）")
    axes[1].plot(ctime[csel], data[csel], lw=.55,color="#0072B2",label="20–450 Hz 带通 + 50 Hz谐波陷波")
    axes[1].plot(ctime[csel], envelope[csel], lw=1.25,color="#D55E00",label="全波整流 + 5 Hz 低通包络")
    axes[1].set(ylabel="幅值（源单位）",xlabel="记录内时间（s）",title="处理后：包络线反映短时间内的肌肉活动强弱，非动作学位移")
    for ax in axes: ax.grid(alpha=.18); ax.legend(loc="upper right",fontsize=8)
    axes[1].text(.01,.04,"EMG 峰值/包络升高通常表示该肌肉电活动更强；需有可信同步事件才可归因于某一次动作。", transform=axes[1].transAxes,fontsize=8.4,bbox=dict(boxstyle="round",fc="white",ec="#999999",alpha=.92))
    files=[save(fig,"05_emg_before_after_envelope")]
    # IMU uses the same selected physical sensor. We display source-unit motion
    # intensity and gravity-based roll/pitch only, without integrating yaw.
    fields=parse_header(raw_path); acc=_stream_arrays(raw_path,fields[sensor_id]["Acc"]); gyro=_stream_arrays(raw_path,fields[sensor_id]["Gyro"])
    ta,av=acc[:,0],acc[:,1::2]; valid=np.isfinite(ta)&np.isfinite(av).all(axis=1)
    ta,av=ta[valid],av[valid]; norm=np.linalg.norm(av,axis=1)
    # 1-s bins for readable intensity / tilt proxy.
    starts=np.arange(np.floor(ta.min()),np.ceil(ta.max()),1.0); rows=[]
    for start in starts:
        z=(ta>=start)&(ta<start+1)
        if z.sum()<5: continue
        a=av[z]; n=norm[z]
        rows.append((start, np.sqrt(np.mean((n-np.median(n))**2)), np.median(np.degrees(np.arctan2(a[:,1],a[:,2]))), np.median(np.degrees(np.arctan2(-a[:,0],np.sqrt(a[:,1]**2+a[:,2]**2)))), np.mean((n>=.75)&(n<=1.25))))
    rows=np.asarray(rows); vis=(rows[:,0]>=left)&(rows[:,0]<=right)
    fig,axes=plt.subplots(2,1,figsize=(12.2,5.8),sharex=True,constrained_layout=True)
    axes[0].plot(rows[vis,0],rows[vis,1],color="#6A3D9A",lw=1.2); axes[0].set(ylabel="动态加速度 RMS\n（源单位）",title=f"IMU：同一传感器 {sensor_id} 的运动强度（1 秒窗）")
    axes[1].plot(rows[vis,0],rows[vis,2],color="#009E73",label="roll 倾角代理")
    axes[1].plot(rows[vis,0],rows[vis,3],color="#E69F00",label="pitch 倾角代理")
    axes[1].set(ylabel="角度代理（°）",xlabel="记录内时间（s）",title="只有重力占主导的时段才可把这些曲线作为姿态倾角代理")
    for ax in axes: ax.grid(alpha=.18)
    axes[1].legend(loc="upper right",fontsize=8)
    axes[1].text(.01,.04,"不能解算绝对 yaw 或完整关节角：坐标系、单位、传感器绑定位姿及磁校准信息未提供。",transform=axes[1].transAxes,fontsize=8.4,bbox=dict(boxstyle="round",fc="white",ec="#999999",alpha=.92))
    files.append(save(fig,"06_imu_motion_intensity_tilt_proxy"))
    return files,[
        {"modality":"EMG","measure":"representative_sensor","value":sensor_id,"unit":"sensor id"},
        {"modality":"EMG","measure":"representative_muscle","value":muscle,"unit":"label"},
        {"modality":"EMG","measure":"raw_sample_rate","value":fs_raw,"unit":"Hz"},
        {"modality":"EMG","measure":"processed_sample_rate","value":fs,"unit":"Hz"},
        {"modality":"EMG","measure":"mapping_status","value":str(clean["mapping_status"]),"unit":"status"},
        {"modality":"IMU","measure":"tilt_proxy_gravity_consistent_fraction_median","value":float(np.median(rows[:,4])),"unit":"fraction"},
        {"modality":"IMU","measure":"interpretation","value":"TILT_PROXY_ONLY; no yaw/absolute pose","unit":"status"},
    ]


def readme(winner: dict[str,Any], figure_files: list[str]) -> str:
    return f"""# 代表性受试者的预处理前后对比

## 这份材料回答什么

这不是疗效图，而是一份**数据处理工作展示**。它用一个按照预先写明的质量规则挑选出的完整访视，直观说明原始 EEG、fNIRS、EMG 和 IMU 数据如何变成可做统计的分析数据。

选中的记录：`{winner['subject']} / {winner['session']}`；综合质量分 `{winner['selection_quality_score_0_100']:.1f}/100`。

## 选择规则（先定规则，再选个例）

只有同时满足下列条件的运动任务记录才进入排名：EEG 的 15 个核心动作标记齐全且整段记录完整；fNIRS 的 15 个预先定义动作标记齐全；EMG 的 7 个标准传感器映射可用；每个传感器都有 Acc 和 Gyro 数据。随后按 EEG 动作期广泛伪迹比例（30%）、EEG 坏导联比例（20%）、fNIRS 保留通道比例（25%）、fNIRS 有效样本比例（15%）和 EMG 平均质量分（10%）计算 0–100 分。完整性是准入条件，不能用很高的单项分弥补缺失。

## 如何讲图

- **EEG 时域图**：灰线是原始单导联，蓝线是重参考、50 Hz 陷波和 0.5–45 Hz 滤波后的同一导联。波峰波谷是瞬时电位起伏；更整洁只说明干扰被压低、信号更适合分析，并不等于治疗有效。
- **EEG θ/α/β 图**：上图的包络越高，代表该频段在此刻的振幅更大；下图的频谱峰代表该频率附近振荡能量更集中。它们需要在预先规定的任务、静息或前后比较中解释，不能从单张个例图得出认知改善。
- **fNIRS 图**：第一行是两种波长的原始光强，第二行是转换为光密度、校正突变并做 0.01–0.2 Hz 滤波后的曲线，第三行是相对 MBLL 的 ΔHbO/ΔHbR 展示。第三行是依赖消光系数、DPF 和源探距离假设的相对估计，不是绝对浓度，也不能单独称为“脑激活”。
- **EMG 图**：蓝线是 20–450 Hz 带通并去除 50 Hz 谐波后的原始肌电；红线是整流后低通的包络。红线高说明短时间内该肌肉的电活动更强，不是关节位移或完整动作学。
- **IMU 图**：第一行是动态运动强度，第二行是重力主导时的 roll/pitch 倾角代理。由于没有坐标系、单位、绑带方位和磁校准信息，不能解算绝对朝向、yaw 或完整关节角。

## 统一预处理步骤与理由

| 模态 | 做了什么 | 为什么这样做 | 仍保留的限制 |
|---|---|---|---|
| EEG | 中位数重参考、50 Hz 陷波、四阶零相位 0.5–45 Hz 带通、2 秒窗伪迹/坏导联标记 | 降低共模偏移和工频，保留常用 θ/α/β 范围；不把坏段偷偷修成好段 | EOG/ECG 实际未连接，未做眼电或心电回归；插值导联只用于可视化 |
| fNIRS | 非正/缺失样本剔除、双波长通道质控、突变检测和修复、0.01–0.2 Hz 滤波、样本 mask | 把光强变为可比较的 OD，降低突跳和高频噪声，并明确哪些样本不能用于统计 | 无短程通道；滤波不等于完全去除系统生理噪声；MBLL 参数存在假设 |
| EMG | 50 Hz 及谐波陷波、20–450 Hz 带通、重采样；展示时另算整流低通包络 | 去工频、保留肌电主要频带；包络便于看肌肉活动强弱 | CSV 未声明幅值单位，EMG 没有可信动作标记，不能假装与脑信号逐秒同步 |
| IMU | 按各自时间戳读取 Acc/Gyro；1 秒窗动态强度和重力倾角代理 | 不强行积分会漂移的陀螺；先展示最保守、可解释的信息 | 不提供绝对姿态、yaw、完整运动学或关节角 |

## 图文件

{chr(10).join('- `' + f + '`' for f in figure_files)}

## 不应这样讲

不要说“处理后更平滑所以治疗有效”，不要说“HbO 上升就是某脑区激活”，不要说“EMG 就是完整运动学”，不要把图中单个波峰波谷直接说成疗效或机制。正确说法是：**预处理把已知噪声、工频、异常通道与异常样本显式管理，使后续在受试者层面比较任务和前后变化成为可能。**
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, winner, descriptors = make_selection()
    write_csv(OUT / "subject_selection.csv", sorted(rows, key=lambda r: (not bool(r["eligible_complete_multimodal"]), -float(r["selection_quality_score_0_100"]))))
    write_csv(OUT / "teacher_subject_selection_anonymized.csv", sorted(teacher_selection_rows(rows), key=lambda r: (not bool(r["eligible_complete_multimodal"]), -float(r["selection_quality_score_0_100"]))))
    (OUT / "figure_contract.md").write_text("""# 图表合同\n\n核心结论：代表性记录说明处理流程如何提高信号可读性和质量可追溯性，不能用来证明治疗有效。\n\n图表类型：多模态定量网格。\n\n证据层级：第一层为透明的个例选择；第二层为每个模态同一时间片段的原始/处理后对比；第三层为质量门控与不可作出的解释。\n\n审稿风险：单个个例不是群体推断；fNIRS 的相对 MBLL 参数有假设；EMG 没有可信动作标记；EOG/ECG 未连接；IMU 不是完整运动学。\n""", encoding="utf-8")
    eeg_s = descriptors[f"{winner['subject']}/{winner['session']}/EEG"][1]
    fnirs_s = descriptors[f"{winner['subject']}/{winner['session']}/FNIRS"][1]
    emg_s = descriptors[f"{winner['subject']}/{winner['session']}/EMG"][1]
    figures = [plot_selection(rows, winner)]
    metrics: list[dict[str, Any]] = []
    paths, m = process_eeg(winner["subject"], winner["session"], eeg_s); figures += paths; metrics += m
    paths, m = process_fnirs(winner["subject"], winner["session"], fnirs_s); figures += paths; metrics += m
    paths, m = process_emg_imu(winner["subject"], winner["session"], emg_s); figures += paths; metrics += m
    for row in metrics:
        row.update(subject=winner["subject"], session=winner["session"])
    write_csv(OUT / "figure_metrics.csv", metrics)
    (OUT / "README.md").write_text(readme(winner, figures), encoding="utf-8")
    manifest = {"subject":winner["subject"], "session":winner["session"], "figures":figures,
                "selection_rule":"completeness gate plus fixed weighted score described in README and subject_selection.csv",
                "teacher_selection_table":"teacher_subject_selection_anonymized.csv (no source paths, names, or birth dates)",
                "scope":"quality-control and processing demonstration; not an efficacy, biomarker, or causal-mechanism analysis"}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
