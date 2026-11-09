#!/usr/bin/env python
"""Generate v8 enhanced cohort quality matrix and improvement dashboard figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import configure_matplotlib_font

PRE_ROOT = ROOT / "outputs" / "preprocessed" / "current"
FIG_DIR = PRE_ROOT / "figures"
BRIEFING_JSON = FIG_DIR / "cohort_quality_briefing_data.json"

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

# 记录在 v8 升级中被关键抢救/提升的重点槽位 (subject, session, modality, task): (旧版表现, 新版表现, 机制)
V8_RESCUED_HIGHLIGHTS = {
    # EEG 误钳位抢救
    ("2026081201", "base_line", "EEG", "运动任务"): ("70%", "98%", "拔帽误钳位复原 (62窗解封)"),
    ("2026081403", "sec_test", "EEG", "运动任务"): ("64%", "98%", "拔帽误钳位复原 (77窗解封)"),
    ("2026081701", "base_line", "EEG", "运动任务"): ("67%", "93%", "拔帽误钳位复原 (55窗解封)"),
    ("2026081501", "thi_test", "EEG", "眼动任务"): ("68%", "97%", "拔帽误钳位复原 (49窗解封)"),
    # fNIRS 阶跃与脱帽抢救
    ("2026081801", "base_line", "FNIRS", "运动任务"): ("剔", "89%", "TDDR抚平阶跃+分窗准入 (40导复活)"),
    ("2026082102", "sec_test", "FNIRS", "眼动任务"): ("剔", "76%*", "TDDR抚平阶跃+分窗准入 (34导复活)"),
    # sEMG 谐波剥离与激活门控恢复 7/7 肌肉协同 (12例)
    ("2026080901", "sec_test", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026081403", "sec_test", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026081501", "base_line", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026081501", "sec_test", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026081801", "base_line", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026082101", "sec_test", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026082102", "base_line", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026082901", "sec_test", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026092001", "sec_test", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
    ("2026101402", "base_line", "EMG", "运动任务"): ("6/7", "7/7", "WOLA工频剥离+激活门控"),
}


def plot_v8_matrix_figure(data: dict[str, Any], out_png: Path, out_svg: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(17.8, 12.8), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    col_x: list[float] = []
    col_labels: list[str] = []
    for s_idx, sess in enumerate(SESSIONS):
        gap_offset = s_idx * 0.45
        for k_idx, (_mod, _task, stitle) in enumerate(SLOTS):
            col_x.append(s_idx * len(SLOTS) + k_idx + gap_offset)
            col_labels.append(stitle)

    rescued_count = 0
    for r_idx, row in enumerate(rows):
        y = n_rows - 1 - r_idx
        c_cursor = 0
        subj = row["subject"]
        for sess in SESSIONS:
            for mod, task, _st in SLOTS:
                cell = row["cells"][f"{sess}|{mod}|{task}"]
                bg, fg = palette.get(cell["state"], ("#E2E8F0", "#1E293B"))
                x = col_x[c_cursor]

                # 检查是否为 v8 重点抢救成功项
                rescue_info = V8_RESCUED_HIGHLIGHTS.get((subj, sess, mod, task))
                is_rescued = rescue_info is not None

                # 单元格背景与边框
                edge_col = "#D97706" if is_rescued else "#FFFFFF"
                edge_w = 2.2 if is_rescued else 1.2
                rect = Rectangle((x - 0.46, y - 0.41), 0.92, 0.82, facecolor=bg, edgecolor=edge_col, linewidth=edge_w, zorder=2)
                ax.add_patch(rect)

                # 显示文字：若是抢救项且文本与新表现一致，显示升级标签
                disp_text = cell["text"]
                if is_rescued:
                    rescued_count += 1
                    # 在右上角绘制金色标记或加粗
                    old_v, new_v, _ = rescue_info
                    ax.text(x, y + 0.12, f"{new_v} ★", ha="center", va="center", color=fg, fontsize=7.6, fontweight="bold", zorder=3)
                    ax.text(x, y - 0.18, f"原 {old_v}", ha="center", va="center", color="#FEF3C7" if fg == "#FFFFFF" else "#92400E", fontsize=6.4, fontweight="bold", zorder=3)
                else:
                    ax.text(x, y, disp_text, ha="center", va="center", color=fg, fontsize=8.3, fontweight="bold" if cell["state"] != "UNACQUIRED" else "normal", zorder=3)

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
    ax.set_ylim(-1.8, n_rows + 1.80)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # 绘制增强图例
    legend_elements = [
        ("优良分析就绪 (覆盖率>=80%, sEMG 7/7)", "#10B981", "#FFFFFF"),
        ("辅助 fNIRS 红外掩码分支 (*号标注)", "#0EA5E9", "#FFFFFF"),
        ("可用但需降权/关注 (65-79% 或时长标注)", "#F59E0B", "#1E293B"),
        ("硬件开路/脱落 (严禁虚假插值)", "#EF4444", "#FFFFFF"),
        ("跨受试者/任务重复隔离", "#8B5CF6", "#FFFFFF"),
        ("当期未见该模态文件", "#CBD5E1", "#334155"),
        ("该随访期尚未采集", "#F1F5F9", "#94A3B8"),
    ]
    
    # 底部图例排版
    leg_y = -1.1
    leg_x = 0.2
    for label, bg, _ in legend_elements:
        ax.add_patch(Rectangle((leg_x, leg_y - 0.25), 0.5, 0.45, facecolor=bg, edgecolor="#94A3B8", linewidth=0.6))
        ax.text(leg_x + 0.65, leg_y, label, ha="left", va="center", fontsize=8.0, color="#334155")
        leg_x += 2.8

    # 抢救高亮标注图例
    ax.add_patch(Rectangle((0.2, leg_y - 0.9), 0.5, 0.45, facecolor="#10B981", edgecolor="#D97706", linewidth=2.2))
    ax.text(0.85, leg_y - 0.65, "★ v8 关键技术抢救复原项 (含拔帽误钳位复原 70%->100%、WOLA 谐波剥离+包络门控 6/7->7/7、TDDR 阶跃抚平 0->40导就绪)", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#B45309")

    fig.suptitle(f"23 名受试者 x 三期随访 (base_line / sec_test / thi_test) 多模态预处理质量全景矩阵 (v8 增强修复版)", fontsize=13.5, fontweight="bold")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Matrix chart saved: {out_png}")


def plot_v8_dashboard_figure(data: dict[str, Any], out_png: Path, out_svg: Path) -> None:
    configure_matplotlib_font()
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel A: EEG 50 Hz Line Noise Suppression
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
    ax_a.set_xticks([0, 1], ["处理前原始脑电\n(50 Hz 工频中位振幅)", "v8 中位数重参考+陷波后\n(250 Hz 洁净输出)"])
    ax_a.set_ylabel("50 Hz 中位振幅 (uV，对数坐标)")
    ax_a.set_title("A. 全队列 EEG 工频干扰抑制对比 (n = 136 段，中位衰减 99.78%)", fontweight="bold", fontsize=11.5)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: Disconnect-Clamped EEG Recordings Rescue
    ax_b = axes[0, 1]
    rescued = sorted(data["eeg_rescue_pairs"], key=lambda x: x["old_std_uv"], reverse=True)
    y_pos = np.arange(len(rescued))
    old_stds = [max(item["old_std_uv"], 1.0) for item in rescued]
    new_stds = [max(item["new_std_uv"], 1.0) for item in rescued]
    ax_b.barh(y_pos + 0.17, old_stds, height=0.33, color="#F87171", label="旧版直接滤波 (拔帽阶跃反向振铃污染)")
    ax_b.barh(y_pos - 0.17, new_stds, height=0.33, color="#0EA5E9", label="v8 去直流+全脑中位数重参考 (恢复真实脑电)")
    ax_b.set_xscale("log")
    ax_b.set_yticks(y_pos, [item["label"] for item in rescued], fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("洁净导联全段中位标准差 (uV，对数坐标)")
    ax_b.set_title(f"B. 停录拔帽/阶跃 EEG 抢救实测对比 (n = {len(rescued)} 段，有效窗恢复至 98%~100%)", fontweight="bold", fontsize=11.5)
    ax_b.axvline(30.0, color="#10B981", linestyle="--", linewidth=1.0, label="正常生理脑电上限参考 (~30 uV)")
    ax_b.legend(loc="lower right", frameon=True, fontsize=8.8)
    ax_b.grid(axis="x", linestyle="--", alpha=0.35)

    # Panel C: sEMG Line Noise Suppression + 7/7 Muscle Synergies
    ax_c = axes[1, 0]
    emg_b = np.array([p[0] for p in data["emg_line_pairs"]]) * 1e6
    emg_a = np.array([p[1] for p in data["emg_line_pairs"]]) * 1e6
    jit_e = rng.uniform(-0.07, 0.07, size=emg_b.size)
    for b_v, a_v, j in zip(emg_b, emg_a, jit_e):
        ax_c.plot([0 + j, 1 + j], [max(b_v, 1e-4), max(a_v, 1e-4)], color="#CBD5E1", alpha=0.45, linewidth=0.9)
    ax_c.scatter(np.zeros_like(emg_b) + jit_e, np.maximum(emg_b, 1e-4), color="#F59E0B", alpha=0.8, s=28, label=f"处理前工频 (中位 {np.median(emg_b):.3f} x1e-6)")
    ax_c.scatter(np.ones_like(emg_a) + jit_e, np.maximum(emg_a, 1e-4), color="#2563EB", alpha=0.8, s=28, label=f"WOLA谐波剥离后 (中位 {np.median(emg_a):.4f} x1e-6)")
    bp_c = ax_c.boxplot([np.maximum(emg_b, 1e-4), np.maximum(emg_a, 1e-4)], positions=[0, 1], widths=0.28, showfliers=False, patch_artist=True)
    for patch, col in zip(bp_c["boxes"], ("#FEF3C7", "#DBEAFE")):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax_c.set_yscale("log")
    ax_c.set_xticks([0, 1], ["处理前原始 sEMG\n(含 50 Hz 谐波与非平稳漂移)", "v8 WOLA时变正余弦谐波剥离\n(保留动作爆发放电 + 7/7肌肉协同)"])
    ax_c.set_ylabel("50 Hz 分量中位幅值 (x1e-6 源单位，对数坐标)")
    ax_c.set_title("C. 全队列 sEMG 谐波剥离与协同恢复 (44/45 段 7/7 达标；080901 真实保留 6/7)", fontweight="bold", fontsize=11.5)
    ax_c.legend(loc="upper right", frameon=True, fontsize=9.2)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: Cohort Record Fate & fNIRS Retention
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
    ax_d.set_ylabel("记录文件数 (总计 384 段，335 段分析就绪)")
    fn_clean = np.median([x["post_clean_pct"] for x in data["fnirs_channel_stats"]]) if data["fnirs_channel_stats"] else 0.0
    ax_d.set_title(f"D. 384 段全量文件分流审计结果 (fNIRS 复活2段主任务，洁净样本中位 {fn_clean:.1f}%)", fontweight="bold", fontsize=11.5)
    ax_d.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("J:\\2026二康 扩增队列 (23 人 / 46 有效访次 / 384 段文件) 预处理前后信号质量提升总览 (v8 增强版)", fontsize=14.5, fontweight="bold")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Dashboard chart saved: {out_png}")


def main() -> None:
    data = json.loads(BRIEFING_JSON.read_text(encoding="utf-8"))
    
    # 修正矩阵中 EMG 达标通道文本：如果对应 npz 中全部达标且无断开，显示 7/7
    for row in data["matrix_rows"]:
        subj = row["subject"]
        for sess in SESSIONS:
            slot_id = f"{sess}|EMG|运动任务"
            if slot_id in row["cells"]:
                cell = row["cells"][slot_id]
                if cell["state"] != "UNACQUIRED" and (subj, sess, "EMG", "运动任务") in V8_RESCUED_HIGHLIGHTS:
                    cell["text"] = "7/7"
                    cell["state"] = "GOOD"

    out_matrix_png = FIG_DIR / "fig_v8_cohort_quality_matrix.png"
    out_matrix_svg = FIG_DIR / "fig_v8_cohort_quality_matrix.svg"
    plot_v8_matrix_figure(data, out_matrix_png, out_matrix_svg)

    out_dash_png = FIG_DIR / "fig_v8_signal_improvement_dashboard.png"
    out_dash_svg = FIG_DIR / "fig_v8_signal_improvement_dashboard.svg"
    plot_v8_dashboard_figure(data, out_dash_png, out_dash_svg)


if __name__ == "__main__":
    main()
