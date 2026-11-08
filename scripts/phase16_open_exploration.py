"""Phase 16: low-complexity open exploration on the paired-seven queue."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import PAIRED_SUBJECTS, phase_root

ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(16)
OUT.mkdir(parents=True, exist_ok=True)


def loo_rho(x, y):
    vals = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))]
    return float(np.nanmin(vals)), float(np.nanmax(vals)), ",".join(f"{v:.3f}" for v in vals)


def main():
    m = pd.read_csv(phase_root(11) / "nof1_delta_matrix.csv")
    m.subject = m.subject.astype(str)
    m = m[m.subject.isin(PAIRED_SUBJECTS)].set_index("subject")
    t = pd.read_csv(phase_root(14) / "task_spatial_redistribution_summary.csv")
    t.subject = t.subject.astype(str)
    t = t[t.subject.isin(PAIRED_SUBJECTS)]
    wide = t.pivot(index="subject", columns="task", values="mean_abs_channel_delta")
    wide["head_minus_knee"] = wide["摸头"] - wide["摸膝"]
    rows = []
    # Candidate 1: complete task-space contrast, magnitude only.
    x = wide["head_minus_knee"].dropna()
    rows.append({"candidate": "task_space_head_minus_knee_abs", "domain": "动作特异性空间指纹", "n": len(x), "summary_statistic": "mean/median and sign of task contrast", "estimate": float(x.mean()), "median": float(x.median()), "direction_consistency": float((x > 0).mean()), "loo_range": "", "subject_values": "; ".join(f"{sid}:{v:.6g}" for sid, v in x.items()), "interpretation": "摸头与摸膝的平均绝对通道变化幅度存在可量化差异", "limitation": "绝对幅度差不代表激活方向；探头中点坐标代理，不作脑区/穴位归因"})
    # Candidate 2: rest-to-movement bridge.
    x = m["rest_network"].reindex(wide.index).to_numpy(float); y = m["movement_fnirs"].reindex(wide.index).to_numpy(float)
    rho = float(spearmanr(x, y).statistic); lo, hi, lv = loo_rho(x, y)
    rows.append({"candidate": "rest_to_movement_rank_bridge", "domain": "静息—运动桥接", "n": len(x), "summary_statistic": "Spearman rho; leave-one-out rho", "estimate": rho, "median": np.nan, "direction_consistency": np.nan, "loo_range": f"[{lo:.3f},{hi:.3f}] ({lv})", "subject_values": "; ".join(f"{sid}:rest={a:.6g},movement={b:.6g}" for sid, a, b in zip(wide.index, x, y)), "interpretation": "静息网络整合变化与运动 fNIRS 响应变化的秩关系", "limitation": "小样本相关仅为假设生成；无因果方向，fNIRS 为光密度 proxy"})
    # Candidate 3: dynamic-state to task-space bridge.
    x = m["occupancy_entropy_norm_delta"].reindex(wide.index).to_numpy(float); y = wide["head_minus_knee"].to_numpy(float)
    rho = float(spearmanr(x, y).statistic); lo, hi, lv = loo_rho(x, y)
    rows.append({"candidate": "state_entropy_to_task_space", "domain": "动态状态—动作空间桥接", "n": len(x), "summary_statistic": "Spearman rho; leave-one-out rho", "estimate": rho, "median": np.nan, "direction_consistency": np.nan, "loo_range": f"[{lo:.3f},{hi:.3f}] ({lv})", "subject_values": "; ".join(f"{sid}:entropy={a:.6g},head-knee={b:.6g}" for sid, a, b in zip(wide.index, x, y)), "interpretation": "状态占用熵变化与摸头—摸膝空间幅度对比存在负向秩关系", "limitation": "session-local 数学状态；任务对比为摘要差值；不命名为认知/经络状态"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    head, bridge, dynamic = (out.iloc[index] for index in range(3))
    report = f"""# Phase 16 开放探索候选（n={len(wide)} 工作队列）

本轮保留七名完整 baseline→sec_test 配对受试者，包括已经更正二次测试数据的 2026081401。所有数据先聚合到 subject-level。候选不是确认性结果，不把任务、通道或事件作为样本。

## 候选 1：摸头—摸膝空间幅度差

`mean_abs_channel_delta(摸头) - mean_abs_channel_delta(摸膝)` 的均值为 {head.estimate:+.5f}，中位数为 {head['median']:+.5f}，正向比例为 {head.direction_consistency:.0%}。它直接回应动作特异性空间指纹，但这是绝对变化幅度，不表示激活方向。

## 候选 2：静息网络—运动响应秩桥接

`rest_network` 与 `movement_fnirs` 的 Spearman rho={bridge.estimate:+.2f}；LOO 结果见 `candidate_summary.csv`。可作为“静息网络整合变化是否伴随运动响应变化”的假设生成桥接，但不能宣称因果。

## 候选 3：动态状态—动作空间桥接

`occupancy_entropy_norm_delta` 与摸头—摸膝空间幅度差的 Spearman rho={dynamic.estimate:+.2f}；LOO 结果见 `candidate_summary.csv`。该关系回应动态状态与动作表型的连接，但 state 是 session-local 数学状态，不能命名为认知、治疗或经络状态。

## 排除项

眼动/四诊前后配对覆盖仍有限；EMG bridge 有效覆盖不足或质量不均；EEG 质量偏低。因此本轮不把它们包装成确认性候选。未进行逐通道扫描、深度学习或显著性包装。

任务名称 protocol-labeled，marker timestamps automatic/accurate；比较是约两周纵向、固定组合针刺/电针方案，不按合谷、四神聪、百会等单穴位分层。

下一步只建议对上述候选做预先固定的复核图/LOO 稳定性检查，待增加配对受试者后再考虑验证性统计。
"""
    (OUT / "open_exploration_candidates.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}; candidates={len(out)}; n={len(wide)}")


if __name__ == "__main__":
    main()
