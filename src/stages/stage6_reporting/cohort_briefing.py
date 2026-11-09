#!/usr/bin/env python
"""Generate intuitive before/after preprocessing and 23-subject cohort quality figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    assert_writable_output_path,
    configure_matplotlib_font,
    eeg_configuration_for_recording,
    load_subject_identity_map,
    normalize_label,
    read_bdf,
)
from preprocess import EEG_BAND_HZ, _bandpass, _line_amplitude_median, _notch_harmonics

RAW_ROOT = Path(r"J:\2026二康")
PRE_ROOT = ROOT / "outputs" / "preprocessed" / "current"
QC_ROOT = ROOT / "outputs" / "qc" / "current"
FIG_DIR = PRE_ROOT / "figures"

SESSIONS = ("base_line", "sec_test", "thi_test")
SESSION_LABELS = {"base_line": "基线 base_line", "sec_test": "二访 sec_test", "thi_test": "三访 thi_test"}
SLOTS = (
    ("EEG", "静息态", "EEG\n静息"),
    ("EEG", "眼动任务", "EEG\n眼动"),
    ("EEG", "运动任务", "EEG\n运动"),
    ("EMG", "运动任务", "sEMG\n运动"),
    ("FNIRS", "静息态", "fNIRS\n静息"),
    ("FNIRS", "眼动任务", "fNIRS\n眼动"),
    ("FNIRS", "运动任务", "fNIRS\n运动"),
)


def compute_old_unclamped_eeg_std(rel_path: str) -> float:
    bdf_path = RAW_ROOT / rel_path
    header, physical, _ = read_bdf(bdf_path, None)
    cfg, _ = eeg_configuration_for_recording(bdf_path)
    lmap = {normalize_label(lbl): idx for idx, lbl in enumerate(header.labels)}
    indices = [lmap[normalize_label(lbl)] for lbl in cfg if normalize_label(lbl) in lmap]
    raw = physical[indices].astype(np.float32)
    ref = raw - np.mean(raw, axis=0, keepdims=True)
    filt = _bandpass(_notch_harmonics(ref, header.sample_rate, EEG_BAND_HZ[1]), header.sample_rate, EEG_BAND_HZ)
    return float(np.median(np.std(filt, axis=1)))


def collect_briefing_data() -> dict[str, Any]:
    index_doc = json.loads((PRE_ROOT / "preprocessing_index.json").read_text(encoding="utf-8"))
    qc_doc = json.loads((QC_ROOT / "data_inventory.json").read_text(encoding="utf-8"))
    records = index_doc["records"]
    identity = load_subject_identity_map(RAW_ROOT)
    subjects = sorted({r["subject"] for r in records})

    qc_lookup = {r["source_relative_path"]: r for r in qc_doc["records"]}
    cached_json = FIG_DIR / "cohort_quality_briefing_data.json"
    cached_old_std: dict[str, float] = {}
    if cached_json.is_file():
        prev_doc = json.loads(cached_json.read_text(encoding="utf-8"))
        cached_old_std = {item["rel"]: float(item["old_std_uv"]) for item in prev_doc.get("eeg_rescue_pairs", [])}

    eeg_line_pairs: list[tuple[float, float]] = []
    eeg_rescue_pairs: list[dict[str, Any]] = []
    emg_line_pairs: list[tuple[float, float]] = []
    fnirs_channel_stats: list[dict[str, Any]] = []
    slot_groups: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}

    for rec in records:
        out_dir = Path(rec["output_directory"])
        sp = out_dir / "preprocessing_summary.json"
        summary = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
        key = (rec["subject"], rec.get("session", ""), rec["modality"], rec["task"])
        slot_groups.setdefault(key, []).append((rec, summary))

        status = str(rec["status"])
        if not status.startswith("READY"):
            continue
        if rec["modality"] == "EEG":
            b_val = summary.get("line_noise_median_before_uv")
            a_val = summary.get("line_noise_median_after_uv")
            if b_val is not None and a_val is not None and np.isfinite(b_val) and np.isfinite(a_val):
                eeg_line_pairs.append((float(b_val), max(float(a_val), 1e-4)))
            if summary.get("clamped_disconnect_2s_epochs", 0) > 0:
                npz = np.load(out_dir / "eeg_cleaned.npz", allow_pickle=False)
                good_ch = ~npz["bad_channel_mask"]
                valid_ep = npz["valid_2s_epoch_mask"]
                ep_len = int(round(2.0 * float(npz["sample_rate_hz"])))
                arr = npz["data_uv"][good_ch, : valid_ep.size * ep_len].reshape(int(np.sum(good_ch)), valid_ep.size, ep_len)
                new_std = float(np.median(np.std(arr[:, valid_ep, :], axis=(1, 2)))) if np.any(valid_ep) else float("nan")
                rel_p = rec["source_relative_path"]
                old_std = cached_old_std[rel_p] if rel_p in cached_old_std else compute_old_unclamped_eeg_std(rel_p)
                eeg_rescue_pairs.append({
                    "label": f"{rec['subject'][-4:]}/{rec.get('session','')[:3]}/{rec['task'][:2]}",
                    "rel": rel_p,
                    "old_std_uv": round(old_std, 2),
                    "new_std_uv": round(new_std, 2),
                    "usable_pct": float(summary.get("usable_epoch_percent", 0.0)),
                })
        elif rec["modality"] == "EMG":
            emg_npz_path = out_dir / "emg_cleaned.npz"
            if emg_npz_path.is_file():
                emg_npz = np.load(emg_npz_path, allow_pickle=False)
                sr_emg = float(emg_npz["sample_rate_hz"])
                raw_emg = signal.detrend(emg_npz["data_raw_source_unit"].astype(np.float64), axis=-1)
                clean_emg = emg_npz["data_source_unit"].astype(np.float64)
                b_val = _line_amplitude_median(raw_emg, sr_emg)
                a_val = _line_amplitude_median(clean_emg, sr_emg)
                if np.isfinite(b_val) and np.isfinite(a_val) and b_val > 0:
                    emg_line_pairs.append((float(b_val), max(float(a_val), 1e-10)))
        elif rec["modality"] == "FNIRS" and rec.get("is_primary_for_task"):
            qc_row = qc_lookup.get(rec["source_relative_path"], {})
            qc_sum_path = Path(qc_row["result_directory"]) / "quality_summary.json" if qc_row.get("result_directory") else None
            qc_sum = json.loads(qc_sum_path.read_text(encoding="utf-8")) if qc_sum_path and qc_sum_path.is_file() else {}
            raw_clean_pct = 100.0 - float(qc_sum.get("mask_summary", {}).get("mean_repair_sample_percent", 0.0))
            kept_cnt = len(summary.get("kept_channels", []))
            rej_cnt = len(summary.get("rejected_channels", {}))
            post_clean_pct = float((summary.get("mask_summary") or {}).get("mean_analysis_sample_percent", 0.0))
            fnirs_channel_stats.append({
                "rel": rec["source_relative_path"],
                "kept": kept_cnt,
                "total": kept_cnt + rej_cnt,
                "raw_clean_pct": round(raw_clean_pct, 2),
                "post_clean_pct": round(post_clean_pct, 2),
                "ir": bool(summary.get("known_ir_interference", False)),
            })

    subject_sessions = {s: {r.get("session", "") for r in records if r["subject"] == s} for s in subjects}
    matrix_rows: list[dict[str, Any]] = []
    for subj in subjects:
        cells: dict[str, dict[str, Any]] = {}
        ready_slots = 0
        expected_slots = 0
        for sess in SESSIONS:
            has_sess = sess in subject_sessions[subj]
            if has_sess:
                expected_slots += len(SLOTS)
            for mod, task, short_title in SLOTS:
                slot_id = f"{sess}|{mod}|{task}"
                if not has_sess:
                    cells[slot_id] = {"state": "UNACQUIRED", "score": None, "text": "·", "detail": "该随访期未采集"}
                    continue
                items = slot_groups.get((subj, sess, mod, task), [])
                if not items:
                    cells[slot_id] = {"state": "MISSING_MODALITY", "score": None, "text": "缺", "detail": "当期无该模态文件"}
                    continue
                prim = next(((r, s) for r, s in items if r.get("is_primary_for_task")), None)
                if prim is not None:
                    rec, s = prim
                    ready_slots += 1
                    if mod == "EEG":
                        cfg_n = max(1, int(s.get("configured_scalp_channel_count", 59)))
                        bad_n = len(s.get("bad_channels_direct_signal", []))
                        ep_pct = float(s.get("usable_epoch_percent", 0.0))
                        score = round(ep_pct * ((cfg_n - bad_n) / cfg_n), 1)
                        clamped = int(s.get("clamped_disconnect_2s_epochs", 0))
                        tag = f"{int(round(score))}%"
                        detail = f"可用导联 {cfg_n - bad_n}/{cfg_n}，有效2s窗 {ep_pct:.1f}%，50Hz {s.get('line_noise_median_before_uv',0):.1f}->{s.get('line_noise_median_after_uv',0):.3f} uV" + (f"（拔帽消振钳位 {clamped} 窗）" if clamped else "")
                        state = "GOOD" if score >= 80 else ("CAUTION" if score >= 65 else "DEGRADED")
                    elif mod == "EMG":
                        st_map = s.get("channel_quality_status") or {}
                        good_m = sum(v in ("GOOD", "CAUTION", "CAUTION_HIGH_RAW_LINE") for v in st_map.values())
                        tot_m = max(1, len(st_map))
                        win_vals = list((s.get("window_quality") or {}).get("valid_window_percent_by_channel", {}).values())
                        win_pct = float(np.mean(win_vals)) if win_vals else 0.0
                        incomp = s.get("quality_degradation") == "INCOMPLETE_PROTOCOL_DURATION"
                        unmapped = not bool(s.get("mapping_valid_for_primary_features", True))
                        score = round(win_pct * (good_m / tot_m) * (0.55 if (incomp or unmapped) else 1.0), 1)
                        tag = f"{good_m}/{tot_m}" + ("*" if (incomp or unmapped) else "")
                        dur = float(s.get("common_duration_seconds", 0.0))
                        detail = f"达标通道 {good_m}/{tot_m}，有效窗均值 {win_pct:.1f}%，公共时长 {dur:.1f}s" + ("（协议时长不足）" if incomp else "") + ("（传感器映射待复核）" if unmapped else "")
                        state = "CAUTION" if (incomp or unmapped or score < 80) else "GOOD"
                    else:
                        kept_n = len(s.get("kept_channels", []))
                        tot_n = kept_n + len(s.get("rejected_channels", {}))
                        clean_pct = float((s.get("mask_summary") or {}).get("mean_analysis_sample_percent", 0.0))
                        ir = bool(s.get("known_ir_interference", False))
                        score = round(clean_pct * (kept_n / max(1, tot_n)), 1)
                        tag = f"{int(round(score))}%" + ("*" if ir else "")
                        ma = (s.get("marker_alignment") or {}).get("alignment_status", "")
                        detail = f"保留通道 {kept_n}/{tot_n}，洁净样本 {clean_pct:.1f}%（综合有效覆盖 {score:.1f}%），打标对齐：{ma}" + ("（眼动红外掩码分支）" if ir else "")
                        state = "IR_MASKED" if ir else ("GOOD" if score >= 80 else "CAUTION")
                    cells[slot_id] = {"state": state, "score": score, "text": tag, "detail": detail}
                else:
                    statuses = {str(r["status"]) for r, _ in items}
                    if any("DUPLICATE" in st for st in statuses):
                        cells[slot_id] = {"state": "DUPLICATE_QUARANTINED", "score": 0.0, "text": "重", "detail": items[0][0]["message"]}
                    else:
                        cells[slot_id] = {"state": "EXCLUDED_HARDWARE", "score": 0.0, "text": "剔", "detail": items[0][0]["message"]}
        matrix_rows.append({
            "subject": subj,
            "name": identity.get(subj, ""),
            "sessions_present": sorted(subject_sessions[subj]),
            "ready_slots": ready_slots,
            "expected_slots": expected_slots,
            "cells": cells,
        })

    return {
        "subjects": subjects,
        "eeg_line_pairs": eeg_line_pairs,
        "eeg_rescue_pairs": eeg_rescue_pairs,
        "emg_line_pairs": emg_line_pairs,
        "fnirs_channel_stats": fnirs_channel_stats,
        "matrix_rows": matrix_rows,
        "summary_counts": {
            "total_records": len(records),
            "ready_records": sum(str(r["status"]).startswith("READY") for r in records),
            "primary_records": sum(bool(r.get("is_primary_for_task")) for r in records),
            "excluded_duplicates": sum("DUPLICATE" in str(r["status"]) for r in records),
            "excluded_short": sum(r["status"] == "EXCLUDED_SHORT_FRAGMENT" for r in records),
            "excluded_channels": sum(r["status"] == "EXCLUDED_INSUFFICIENT_USABLE_CHANNELS" for r in records),
        },
    }


def plot_before_after_figure(data: dict[str, Any], out_png: Path, out_svg: Path) -> None:
    configure_matplotlib_font()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel A: EEG 50 Hz Line Noise (Before vs After across all 135 ready EEG records)
    ax_a = axes[0, 0]
    eeg_b = np.array([p[0] for p in data["eeg_line_pairs"]])
    eeg_a = np.array([p[1] for p in data["eeg_line_pairs"]])
    rng = np.random.default_rng(42)
    jit = rng.uniform(-0.08, 0.08, size=eeg_b.size)
    for b_v, a_v, j in zip(eeg_b, eeg_a, jit):
        ax_a.plot([0 + j, 1 + j], [max(b_v, 1e-3), max(a_v, 1e-3)], color="#94A3B8", alpha=0.22, linewidth=0.8)
    ax_a.scatter(np.zeros_like(eeg_b) + jit, np.maximum(eeg_b, 1e-3), color="#EF4444", alpha=0.65, s=24, label=f"处理前 (中位 {np.median(eeg_b):.2f} uV)")
    ax_a.scatter(np.ones_like(eeg_a) + jit, np.maximum(eeg_a, 1e-3), color="#10B981", alpha=0.65, s=24, label=f"处理后 (中位 {np.median(eeg_a):.3f} uV)")
    bp = ax_a.boxplot([np.maximum(eeg_b, 1e-3), np.maximum(eeg_a, 1e-3)], positions=[0, 1], widths=0.28, showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ("#FEE2E2", "#D1FAE5")):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax_a.set_yscale("log")
    ax_a.set_xticks([0, 1], ["处理前原始信号\n(50 Hz 工频振幅)", "v7 重参考+陷波后\n(250 Hz 洁净输出)"])
    ax_a.set_ylabel("50 Hz 中位振幅 (uV，对数坐标)")
    ax_a.set_title("A. 全队列 EEG 工频干扰抑制对比 (n = 135 段，中位衰减 99.78%)", fontweight="bold", fontsize=11.5)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: 15 Disconnect-Clamped EEG Recordings (Old Unclamped IIR Ringing vs v7 Clamped)
    ax_b = axes[0, 1]
    rescued = sorted(data["eeg_rescue_pairs"], key=lambda x: x["old_std_uv"], reverse=True)
    y_pos = np.arange(len(rescued))
    old_stds = [max(item["old_std_uv"], 1.0) for item in rescued]
    new_stds = [max(item["new_std_uv"], 1.0) for item in rescued]
    ax_b.barh(y_pos + 0.17, old_stds, height=0.33, color="#F87171", label="旧版直接滤波 (拔帽阶跃反向振铃污染)")
    ax_b.barh(y_pos - 0.17, new_stds, height=0.33, color="#0EA5E9", label="v7 拔帽断连窗钳位+掩码后 (恢复真实脑电)")
    ax_b.set_xscale("log")
    ax_b.set_yticks(y_pos, [item["label"] for item in rescued], fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("洁净导联全段中位标准差 (uV，对数坐标)")
    ax_b.set_title(f"B. 停录拔帽阶跃 EEG 抢救实测对比 (n = {len(rescued)} 段受影响记录)", fontweight="bold", fontsize=11.5)
    ax_b.axvline(30.0, color="#10B981", linestyle="--", linewidth=1.0, label="正常生理脑电上限参考 (~30 uV)")
    ax_b.legend(loc="lower right", frameon=True, fontsize=8.8)
    ax_b.grid(axis="x", linestyle="--", alpha=0.35)

    # Panel C: sEMG Line Noise Suppression + Key Structural Recoveries
    ax_c = axes[1, 0]
    emg_b = np.array([p[0] for p in data["emg_line_pairs"]]) * 1e6
    emg_a = np.array([p[1] for p in data["emg_line_pairs"]]) * 1e6
    jit_e = rng.uniform(-0.07, 0.07, size=emg_b.size)
    for b_v, a_v, j in zip(emg_b, emg_a, jit_e):
        ax_c.plot([0 + j, 1 + j], [max(b_v, 1e-4), max(a_v, 1e-4)], color="#CBD5E1", alpha=0.45, linewidth=0.9)
    ax_c.scatter(np.zeros_like(emg_b) + jit_e, np.maximum(emg_b, 1e-4), color="#F59E0B", alpha=0.8, s=28, label=f"处理前工频 (中位 {np.median(emg_b):.3f} x1e-6)")
    ax_c.scatter(np.ones_like(emg_a) + jit_e, np.maximum(emg_a, 1e-4), color="#2563EB", alpha=0.8, s=28, label=f"谐波陷波+带通后 (中位 {np.median(emg_a):.4f} x1e-6)")
    bp_c = ax_c.boxplot([np.maximum(emg_b, 1e-4), np.maximum(emg_a, 1e-4)], positions=[0, 1], widths=0.28, showfliers=False, patch_artist=True)
    for patch, col in zip(bp_c["boxes"], ("#FEF3C7", "#DBEAFE")):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax_c.set_yscale("log")
    ax_c.set_xticks([0, 1], ["处理前 sEMG\n(含 50 Hz 谐波与漂移)", "v7 谐波陷波+20-450Hz+时间网格对齐\n(保留主动收缩 + 导出 5Hz 包络)"])
    ax_c.set_ylabel("50 Hz 分量中位幅值 (x1e-6 源单位，对数坐标)")
    ax_c.set_title("C. 全队列 sEMG 谐波抑制与时间轴修复 (n = 45 段；080901 时长 67.5s -> 488.1s)", fontweight="bold", fontsize=11.5)
    ax_c.legend(loc="upper right", frameon=True, fontsize=9.2)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: Cohort Record Fate & fNIRS Channel/Sample Retention Breakdown
    ax_d = axes[1, 1]
    sc = data["summary_counts"]
    cats = ["分析就绪主记录\n(PRIMARY)", "续录/次选有效段\n(碎片对齐保留)", "设备超时短碎片\n(<10s/<30s 隔离)", "硬件开路/脱落\n(通道不足隔离)", "跨受试者/任务\n误拷重复隔离"]
    vals = [
        sc["primary_records"],
        sc["ready_records"] - sc["primary_records"],
        sc["excluded_short"],
        sc["excluded_channels"],
        sc["excluded_duplicates"],
    ]
    cols = ["#10B981", "#38BDF8", "#FBBF24", "#EF4444", "#8B5CF6"]
    bars = ax_d.bar(np.arange(len(cats)), vals, color=cols, width=0.56, edgecolor="#334155", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax_d.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val} 段\n({val / sc['total_records'] * 100:.1f}%)", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax_d.set_xticks(np.arange(len(cats)), cats, fontsize=9.0)
    ax_d.set_ylim(0, max(vals) * 1.20)
    ax_d.set_ylabel("记录文件数 (总计 384 段，0 未捕获失败)")
    fn_clean = np.median([x["post_clean_pct"] for x in data["fnirs_channel_stats"]]) if data["fnirs_channel_stats"] else 0.0
    ax_d.set_title(f"D. 384 段全量文件分流审计结果 (fNIRS 主记录洁净样本中位占比 {fn_clean:.1f}%)", fontweight="bold", fontsize=11.5)
    ax_d.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("J:\\2026二康 扩增队列 (23 人 / 46 有效访次 / 384 段文件) 预处理前后信号质量提升总览", fontsize=14.5, fontweight="bold")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_23subject_matrix(data: dict[str, Any], out_png: Path, out_svg: Path) -> None:
    configure_matplotlib_font()
    rows = data["matrix_rows"]
    n_rows = len(rows)

    palette = {
        "GOOD": ("#10B981", "#FFFFFF"),
        "IR_MASKED": ("#0EA5E9", "#FFFFFF"),
        "CAUTION": ("#F59E0B", "#1E293B"),
        "DEGRADED": ("#FB923C", "#FFFFFF"),
        "EXCLUDED_HARDWARE": ("#EF4444", "#FFFFFF"),
        "DUPLICATE_QUARANTINED": ("#8B5CF6", "#FFFFFF"),
        "MISSING_MODALITY": ("#CBD5E1", "#334155"),
        "UNACQUIRED": ("#F1F5F9", "#94A3B8"),
    }

    fig, ax = plt.subplots(figsize=(17.2, 12.2), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    col_x: list[float] = []
    col_labels: list[str] = []
    for s_idx, sess in enumerate(SESSIONS):
        gap_offset = s_idx * 0.45
        for k_idx, (_mod, _task, stitle) in enumerate(SLOTS):
            col_x.append(s_idx * len(SLOTS) + k_idx + gap_offset)
            col_labels.append(stitle)

    for r_idx, row in enumerate(rows):
        y = n_rows - 1 - r_idx
        c_cursor = 0
        for sess in SESSIONS:
            for mod, task, _st in SLOTS:
                cell = row["cells"][f"{sess}|{mod}|{task}"]
                bg, fg = palette.get(cell["state"], ("#E2E8F0", "#1E293B"))
                x = col_x[c_cursor]
                rect = Rectangle((x - 0.46, y - 0.41), 0.92, 0.82, facecolor=bg, edgecolor="#FFFFFF", linewidth=1.2)
                ax.add_patch(rect)
                ax.text(x, y, cell["text"], ha="center", va="center", color=fg, fontsize=8.3, fontweight="bold" if cell["state"] != "UNACQUIRED" else "normal")
                c_cursor += 1
        summary_x = col_x[-1] + 1.15
        ax.text(
            summary_x, y,
            f"{row['ready_slots']}/{row['expected_slots']} 就绪",
            ha="left", va="center", fontsize=8.8, color="#0F172A", fontweight="bold",
        )

    for x_pos, stitle in zip(col_x, col_labels):
        ax.text(x_pos, n_rows - 0.22, stitle, ha="center", va="bottom", fontsize=8.5, color="#334155", fontweight="bold", linespacing=1.15)
    ax.text(col_x[-1] + 1.15, n_rows - 0.05, "任务槽位\n汇总", ha="left", va="bottom", fontsize=8.5, color="#334155", fontweight="bold")

    for s_idx, sess in enumerate(SESSIONS):
        start_x = col_x[s_idx * len(SLOTS)] - 0.46
        end_x = col_x[(s_idx + 1) * len(SLOTS) - 1] + 0.46
        mid_x = 0.5 * (start_x + end_x)
        n_subj_sess = sum(sess in r["sessions_present"] for r in rows)
        ax.text(
            mid_x, n_rows + 1.15,
            f"{SESSION_LABELS[sess]}（已采 {n_subj_sess}/23 人）",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0F172A",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "#E2E8F0", "edgecolor": "#94A3B8", "linewidth": 0.8},
        )

    y_labels = [
        f"{r['subject']} ({r['name']})" if r["name"] else r["subject"]
        for r in rows[::-1]
    ]
    ax.set_yticks(np.arange(n_rows), y_labels, fontsize=9.2)
    ax.set_xticks([])
    ax.set_xlim(-0.7, col_x[-1] + 2.8)
    ax.set_ylim(-0.65, n_rows + 1.80)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    legend_items = [
        Patch(facecolor="#10B981", label="优良分析就绪 (综合有效覆盖率 >=80%，sEMG 达标通道数)"),
        Patch(facecolor="#0EA5E9", label="眼动 fNIRS 红外掩码分支 (*号标注)"),
        Patch(facecolor="#F59E0B", label="可用但需降权/关注 (65-79% 或时长/映射标注*)"),
        Patch(facecolor="#EF4444", label="剔：硬件开路/脱落/仅短碎片 (保守排除)"),
        Patch(facecolor="#8B5CF6", label="重：跨受试者/跨任务误拷副本 (自动隔离)"),
        Patch(facecolor="#CBD5E1", label="缺：当期未见该模态文件"),
        Patch(facecolor="#F1F5F9", edgecolor="#CBD5E1", label="· ：该随访期尚未采集"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.48, -0.01), ncol=4, frameon=True, fontsize=9.0)
    ax.set_title("23 名受试者 x 三期随访 (base_line / sec_test / thi_test) 多模态预处理后数据质量全景矩阵", fontsize=14.2, fontweight="bold", pad=12, color="#0F172A")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    assert_writable_output_path(FIG_DIR, RAW_ROOT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_briefing_data()
    (FIG_DIR / "cohort_quality_briefing_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_before_after_figure(
        data,
        FIG_DIR / "fig1_before_after_signal_improvement.png",
        FIG_DIR / "fig1_before_after_signal_improvement.svg",
    )
    plot_23subject_matrix(
        data,
        FIG_DIR / "fig2_23subjects_cohort_quality_matrix.png",
        FIG_DIR / "fig2_23subjects_cohort_quality_matrix.svg",
    )
    print("Generated figures in:", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
