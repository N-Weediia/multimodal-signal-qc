"""Leave-one-subject-out stability for the frozen three-layer marker set."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import PAIRED_SUBJECTS, phase_root


ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root("14_stability")
MARKERS = {
    "rest_network": "rest_network",
    "spatial_channel_change": "spatial_channel_change",
    "occupancy_entropy_norm_delta": "occupancy_entropy_norm_delta",
}


def robust_z(x: pd.Series) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else values.std(ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - median) / scale


def read_matrix() -> pd.DataFrame:
    path = phase_root(11) / "nof1_delta_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["subject"] = frame["subject"].astype(str)
    frame = frame[frame.subject.isin(PAIRED_SUBJECTS)].copy()
    required = {"subject", *MARKERS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing fixed-marker columns: {sorted(missing)}")
    if set(frame.subject) != set(PAIRED_SUBJECTS):
        raise ValueError("paired-seven marker matrix is incomplete")
    return frame[["subject", *MARKERS.values()]].sort_values("subject").reset_index(drop=True)


def make_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_sets = [("n7_all_paired", "", frame)]
    sample_sets.extend(
        (f"loo_without_{subject}", subject, frame[frame.subject.ne(subject)])
        for subject in PAIRED_SUBJECTS
    )
    for sample_set, held_out, subset in sample_sets:
        for _, source in subset.iterrows():
            for marker, column in MARKERS.items():
                value = pd.to_numeric(source[column], errors="coerce")
                signed = marker != "spatial_channel_change"
                if pd.isna(value):
                    direction = "missing"
                elif not signed:
                    direction = "magnitude_only"
                elif value > 0:
                    direction = "positive"
                elif value < 0:
                    direction = "negative"
                else:
                    direction = "zero"
                rows.append(
                    {
                        "sample_set": sample_set,
                        "held_out_subject": held_out,
                        "subject": str(source.subject),
                        "marker": marker,
                        "value": value,
                        "direction": direction,
                        "directional_marker": signed,
                    }
                )
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sample_set, held_out, marker), group in long.groupby(
        ["sample_set", "held_out_subject", "marker"], sort=False, dropna=False
    ):
        values = pd.to_numeric(group.value, errors="coerce").dropna().to_numpy(float)
        signed = bool(group.directional_marker.iloc[0])
        nonzero = values[np.sign(values) != 0]
        median = float(np.median(values)) if len(values) else np.nan
        concordance = (
            float(np.mean(np.sign(nonzero) == np.sign(median)))
            if signed and len(nonzero)
            else np.nan
        )
        rows.append(
            {
                "sample_set": sample_set,
                "held_out_subject": held_out,
                "marker": marker,
                "directional_marker": signed,
                "n": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "median": median,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "n_positive": int((values > 0).sum()) if signed else np.nan,
                "n_negative": int((values < 0).sum()) if signed else np.nan,
                "direction_concordance": concordance,
                "note": "signed direction" if signed else "magnitude only; no direction consistency",
            }
        )
    output = pd.DataFrame(rows)
    full_mean = output[output.sample_set.eq("n7_all_paired")].set_index("marker")["mean"]
    output["mean_shift_from_full"] = output.apply(
        lambda row: row["mean"] - full_mean.get(row["marker"], np.nan), axis=1
    )
    return output


def plot(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = list(MARKERS)
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.2, 1]}, constrained_layout=True
    )
    matrix = frame.set_index("subject")[[MARKERS[name] for name in order]].T
    matrix.index = order
    z = matrix.apply(robust_z, axis=1)
    limit = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))
    image = axes[0].imshow(z, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0].set_yticks(
        range(len(order)),
        ["rest network (signed)", "spatial change (magnitude)", "occupancy entropy (signed)"],
    )
    axes[0].set_xticks(range(len(matrix.columns)), [sid[-4:] for sid in matrix.columns])
    axes[0].set_title("Three-layer subject fingerprints (robust z)")
    fig.colorbar(image, ax=axes[0], shrink=0.78, label="within-marker robust z")

    full = summary[summary.sample_set.eq("n7_all_paired")].set_index("marker")
    loo = summary[summary.sample_set.str.startswith("loo_without_")]
    x = np.arange(len(order))
    for index, marker in enumerate(order):
        values = loo.loc[loo.marker.eq(marker), "mean"].to_numpy(float)
        axes[1].plot([index, index], [np.nanmin(values), np.nanmax(values)], color="#6b7280", lw=4, alpha=0.55)
        axes[1].scatter(np.full(len(values), index), values, s=24, color="#6b7280", alpha=0.75)
        axes[1].scatter(index, full.loc[marker, "mean"], marker="D", s=55, color="#1f4e79", zorder=4)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xticks(x, ["rest network", "spatial change\n(magnitude)", "occupancy entropy"])
    axes[1].set_ylabel("raw delta mean")
    axes[1].set_title("Full n=7 mean and leave-one-out range")
    fig.suptitle("Fixed three-layer marker stability | participant-level sensitivity", fontsize=13)
    for extension, dpi in (("png", 600), ("svg", None), ("pdf", None)):
        fig.savefig(OUT / f"fixed_three_marker_stability.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = read_matrix()
    long = make_long(frame)
    marker_summary = summarize(long)
    long.to_csv(OUT / "fixed_three_marker_stability.csv", index=False, encoding="utf-8-sig")
    marker_summary.to_csv(OUT / "fixed_three_marker_summary.csv", index=False, encoding="utf-8-sig")
    plot(frame, marker_summary)
    signed = marker_summary[
        marker_summary.sample_set.str.startswith("loo_without_") & marker_summary.directional_marker
    ]
    direction_stable = signed.groupby("marker")["mean"].apply(
        lambda values: bool((values > 0).all() or (values < 0).all())
    )
    report = f"""# 固定三层指标留一法稳定性

输入为新版 paired7 的受试者级指纹矩阵。完整队列包含 {', '.join(PAIRED_SUBJECTS)}；`2026081401` 使用更正后的二次测试数据，不再排除。

固定指标为静息 fNIRS 网络整合变化（有符号）、任务空间通道变化幅度（magnitude only）和动态状态占用熵变化（有符号）。对每个指标同时报告完整 n=7 与七次 n=6 leave-one-subject-out 均值；事件、通道和任务均未作为独立样本。

有符号指标的 LOO 均值方向稳定性：{direction_stable.to_dict()}。空间指标为绝对变化幅度，不计算正负方向一致性。

本分析只判断单个受试者对描述性趋势的影响，不能证明疗效、机制或穴位特异性。fNIRS 仍是 optical-density proxy，动态状态是 session-local 数学状态。
"""
    (OUT / "stability_summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}; n={len(frame)}; markers={len(MARKERS)}; loo_sets={len(PAIRED_SUBJECTS)}")


if __name__ == "__main__":
    main()
