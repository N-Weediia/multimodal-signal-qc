"""Phase 14B: low-complexity task spatial redistribution summaries.

The input is the existing phase 5 channel-level event-response source table.
This script reduces paired baseline -> sec_test maps to subject x task rows;
it does not scan channels for significance and does not modify phase 5 data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import PAIRED_SUBJECTS, phase_root


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = phase_root(5) / "figure_spatial_source_data.csv"
OUT = phase_root(14)
TASKS = ["摸头", "摸膝", "摸腰"]
TASK_DISPLAY = {"摸头": "head", "摸膝": "knee", "摸腰": "waist"}
PAIRED_SUBJECTS = list(PAIRED_SUBJECTS)
NONZERO_TOLERANCE = 1e-12


def load_source() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Phase 5 source table not found: {INPUT}")
    frame = pd.read_csv(INPUT)
    required = {"subject", "session", "task", "channel", "feature", "modality", "value", "x", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Phase 5 source table missing columns: {sorted(missing)}")
    frame["subject"] = frame["subject"].astype(str)
    frame = frame.loc[
        frame["feature"].eq("event_delta")
        & frame["modality"].eq("fNIRS")
        & frame["session"].isin(["base_line", "sec_test"])
        & frame["subject"].isin(PAIRED_SUBJECTS)
        & frame["task"].isin(TASKS)
    ].copy()
    for col in ["value", "x", "y"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def summarize(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in PAIRED_SUBJECTS:
        for task in TASKS:
            group = source.loc[(source["subject"] == subject) & (source["task"] == task)].copy()
            baseline = group.loc[group["session"].eq("base_line"), ["channel", "value", "x", "y"]].rename(
                columns={"value": "baseline_value", "x": "baseline_x", "y": "baseline_y"}
            )
            followup = group.loc[group["session"].eq("sec_test"), ["channel", "value", "x", "y"]].rename(
                columns={"value": "sec_test_value", "x": "sec_test_x", "y": "sec_test_y"}
            )
            if baseline["channel"].duplicated().any() or followup["channel"].duplicated().any():
                raise ValueError(f"Duplicate channel rows for {subject} {task}")
            merged = baseline.merge(followup, on="channel", how="inner")
            if merged.empty:
                raise ValueError(f"No paired channels for {subject} {task}")
            coordinates_match = np.allclose(
                merged[["baseline_x", "baseline_y"]].to_numpy(float),
                merged[["sec_test_x", "sec_test_y"]].to_numpy(float),
                equal_nan=True,
            )
            valid = np.isfinite(merged[["baseline_value", "sec_test_value", "baseline_x", "baseline_y"]]).all(axis=1)
            values = merged.loc[valid].copy()
            delta = values["sec_test_value"].to_numpy(float) - values["baseline_value"].to_numpy(float)
            x = values["baseline_x"].to_numpy(float)
            x_median = float(np.median(x))
            left = x < x_median
            right = ~left
            left_count = int(left.sum())
            right_count = int(right.sum())
            left_right_difference = float(np.mean(delta[left]) - np.mean(delta[right])) if left_count and right_count else np.nan
            rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "baseline_session": "base_line",
                    "followup_session": "sec_test",
                    "n_paired_channels": int(len(merged)),
                    "n_valid_channels": int(len(values)),
                    "n_nonzero_channels": int(np.sum(np.abs(delta) > NONZERO_TOLERANCE)),
                    "nonzero_channel_proportion": float(np.mean(np.abs(delta) > NONZERO_TOLERANCE)),
                    "mean_abs_channel_delta": float(np.mean(np.abs(delta))),
                    "left_right_difference": left_right_difference,
                    "left_right_definition": "baseline x < within-map median minus x >= within-map median; probe-coordinate proxy only",
                    "left_channel_count": left_count,
                    "right_channel_count": right_count,
                    "x_median": x_median,
                    "coordinate_match_baseline_sec_test": bool(coordinates_match),
                    "nonzero_tolerance": NONZERO_TOLERANCE,
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != len(PAIRED_SUBJECTS) * len(TASKS):
        raise ValueError(f"Expected {len(PAIRED_SUBJECTS) * len(TASKS)} subject-task rows, got {len(output)}")
    return output


def plot(summary: pd.DataFrame) -> None:
    metrics = [
        ("nonzero_channel_proportion", "Non-zero channel proportion", (0.0, 1.05)),
        ("mean_abs_channel_delta", "Mean absolute channel delta", None),
        ("left_right_difference", "Probe-half contrast (left - right)", None),
    ]
    colors = {"摸头": "#2a9d8f", "摸膝": "#e9c46a", "摸腰": "#e76f51"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    x = np.arange(len(TASKS), dtype=float)
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for index, task in enumerate(TASKS):
            values = pd.to_numeric(summary.loc[summary["task"].eq(task), metric], errors="coerce").dropna().to_numpy(float)
            jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), x[index]) + jitter, values, s=38, color=colors[task], edgecolor="white", linewidth=0.5, alpha=0.9)
            if len(values):
                ax.scatter(x[index], float(np.mean(values)), marker="D", s=42, color="#333333", zorder=4)
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_xticks(x, [TASK_DISPLAY[task] for task in TASKS])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        if ylim:
            ax.set_ylim(*ylim)
        if metric == "nonzero_channel_proportion":
            ax.set_ylabel("proportion")
        elif metric == "mean_abs_channel_delta":
            ax.set_ylabel("fNIRS proxy delta")
        else:
            ax.set_ylabel("fNIRS proxy delta")
    fig.suptitle(f"Task spatial redistribution | paired subject summaries (n={len(PAIRED_SUBJECTS)})", fontsize=13)
    for extension, dpi in (("png", 600), ("svg", None), ("pdf", None)):
        fig.savefig(OUT / f"task_spatial_redistribution.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_review(summary: pd.DataFrame) -> None:
    nonzero_values = pd.to_numeric(summary["nonzero_channel_proportion"], errors="coerce")
    saturated = bool(np.allclose(nonzero_values.to_numpy(float), 1.0, atol=0.0, rtol=0.0))
    coordinate_match = bool(summary["coordinate_match_baseline_sec_test"].all())
    lines = [
        "# Phase 14B 逐任务空间反应重分布摘要",
        "",
        "输入：新版 paired7 phase5 的 fNIRS `event_delta` 通道级源数据。",
        "本阶段只将通道值聚合为 subject×task 的 baseline -> sec_test 摘要，不做逐通道显著性扫描，不修改 phase5 或 canonical 数据。",
        "",
        f"独立配对受试者（n={summary['subject'].nunique()}）：{', '.join(PAIRED_SUBJECTS)}。任务：`摸头`、`摸膝`、`摸腰`。输出行数：{len(summary)}（每名受试者×每个任务一行）。",
        "",
        "## 三项固定指标",
        "",
        "- `nonzero_channel_proportion`：在配对且有效的通道中，`abs(sec_test - base_line) > 1e-12` 的比例。它不是激活显著性比例；连续浮点 proxy 可能使该指标饱和。",
        "- `mean_abs_channel_delta`：配对通道 `abs(sec_test - base_line)` 的平均值，保留变化幅度而不赋予方向。",
        "- `left_right_difference`：左侧均值减右侧均值，使用每个 subject×task 图内 `x < x_median` 与 `x >= x_median` 的探头坐标两半。",
        "",
        "左右项已降级：当前 fNIRS 坐标是 source-detector midpoint，探头解剖左右方向尚未验证，因此 `left_right_difference` 只能解释为可复现的探头坐标半区差值，不是左右半球差异，也不支持穴位-脑区定位。",
        f"本次 {len(summary)} 个 subject×task 映射的 baseline/sec_test 坐标匹配状态：{coordinate_match}。",
        f"非零比例是否全部为 1.0（容差 1e-12）：{saturated}；若为 True，应将其视作连续值差异的覆盖性结果，不作生理阈值解释。",
        "",
        "## 冻结口径与限制",
        "",
        f"比较固定为 `base_line -> sec_test` 的约两周纵向变化；所有受试者接受同一固定组合针刺/电针方案，不做穴位分层。事件、通道和任务行不是独立样本，独立单位为 {summary['subject'].nunique()} 名受试者。",
        "fNIRS 数值为 wavelength-balanced optical-density proxy；本摘要不补充 DPF、短距离回归或新的质量门控。EEG、眼动和四诊探索结果不进入本阶段主指标。",
        "",
        "## 输出",
        "",
        f"- `task_spatial_redistribution_summary.csv`：{len(summary)} 条 subject×task 摘要，含三项指标、有效通道数和左右代理定义。",
        "- `task_spatial_redistribution.png/.svg/.pdf`：三项指标的受试者点图与任务均值标记。",
        "- 本报告：方法定义、覆盖和解释降级说明。",
        "",
        "同目录另有独立生成的 Phase 14A `fixed_three_marker_*` 稳定性文件；其结果不作为本 Phase 14B 任务空间摘要的输入或重新计算对象。",
    ]
    (OUT / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_source())
    summary.to_csv(OUT / "task_spatial_redistribution_summary.csv", index=False, encoding="utf-8-sig")
    plot(summary)
    write_review(summary)
    print(f"Wrote {OUT}; n_subjects={summary['subject'].nunique()}; rows={len(summary)}; tasks={','.join(TASKS)}")


if __name__ == "__main__":
    main()
