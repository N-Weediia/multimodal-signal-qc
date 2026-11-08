"""Phase 15A: fixed within-task spatial pairwise contrasts.

Uses the Phase 14B subject-level summaries only. No channel is an independent
unit and no significance testing is performed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import phase_root

ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(15)
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = [("head-knee", "摸头", "摸膝"), ("waist-knee", "摸腰", "摸膝"), ("head-waist", "摸头", "摸腰")]
METRICS = ["mean_abs_channel_delta", "left_right_difference"]


def main() -> None:
    src = phase_root(14) / "task_spatial_redistribution_summary.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    d = pd.read_csv(src)
    d["subject"] = d["subject"].astype(str)
    rows = []
    for label, task_a, task_b in PAIRS:
        for metric in METRICS:
            a = d[d.task.eq(task_a)][["subject", metric]].rename(columns={metric: "a"})
            b = d[d.task.eq(task_b)][["subject", metric]].rename(columns={metric: "b"})
            x = a.merge(b, on="subject", how="inner")
            x["delta"] = x.a - x.b
            vals = pd.to_numeric(x.delta, errors="coerce").dropna().to_numpy(float)
            rows.append({"contrast": label, "task_a": task_a, "task_b": task_b, "metric": metric, "definition": f"{task_a} minus {task_b}", "n": len(vals), "mean": np.mean(vals) if len(vals) else np.nan, "median": np.median(vals) if len(vals) else np.nan, "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan, "positive_count": int((vals > 0).sum()), "negative_count": int((vals < 0).sum()), "zero_count": int((vals == 0).sum()), "direction_consistency": float(max((vals > 0).mean(), (vals < 0).mean())) if len(vals) else np.nan, "subject_values": "; ".join(f"{sid}:{v:.6g}" for sid, v in zip(x.subject, x.delta))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "task_pairwise_contrasts.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, constrained_layout=True)
    colors = {"mean_abs_channel_delta": "#1f4e79", "left_right_difference": "#e76f51"}
    labels = {"mean_abs_channel_delta": "Mean absolute channel delta", "left_right_difference": "Left-right difference"}
    for j, metric in enumerate(METRICS):
        sub = out[out.metric.eq(metric)]
        for i, label in enumerate([p[0] for p in PAIRS]):
            r = sub[sub.contrast.eq(label)].iloc[0]
            vals = np.array([float(z.split(":", 1)[1]) for z in r.subject_values.split("; ") if z])
            ax[j].scatter(np.full(len(vals), i), vals, s=42, color=colors[metric], edgecolor="white", linewidth=.6)
            ax[j].plot(i, r["mean"], marker="_", markersize=18, color="black", mew=2)
        ax[j].axhline(0, color="0.5", lw=.7); ax[j].set_title(labels[metric]); ax[j].set_ylabel("task A minus task B"); ax[j].set_xticks(range(3), [p[0] for p in PAIRS], rotation=20, ha="right")
    fig.suptitle(f"Within-task spatial contrasts | n={d.subject.nunique()} subjects | descriptive", fontsize=14)
    for ext, dpi in [("png", 600), ("svg", None), ("pdf", None)]: fig.savefig(OUT / f"task_pairwise_contrasts.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    report = "# Phase 15A 任务内空间摘要配对对比\n\n"
    report += f"对每位受试者先完成 baseline→sec_test 的任务空间摘要，再按固定对比计算：head-knee=摸头−摸膝，waist-knee=摸腰−摸膝，head-waist=摸头−摸腰。独立单位为受试者（n={d.subject.nunique()}），不把通道作为样本。\n\n"
    report += "`mean_abs_channel_delta` 是变化幅度指标；本报告的方向一致率仅对该差值相对于另一任务的大小关系描述，并不表示神经激活方向。`left_right_difference` 是探头坐标左右差值的任务对比，不能等同于脑半球因果效应。\n\n"
    report += "未进行显著性检验、逐通道扫描或多重比较校正；结果用于动作空间指纹的低复杂度探索。任务名称按固定 protocol-labeled 方案，marker timestamps automatic/accurate。\n\n"
    report += out.to_markdown(index=False) + "\n"
    (OUT / "task_pairwise_contrasts.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}; rows={len(out)}; subjects={d.subject.nunique()}")


if __name__ == "__main__":
    main()
