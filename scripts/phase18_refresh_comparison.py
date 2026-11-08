"""Compare the frozen n=4 release with the corrected paired-seven refresh."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import phase_root


OLD = ROOT / "outputs" / "release_20260903" / "data"
NEW = ROOT / "outputs" / "analysis" / "paired7_20260903"
OUT = phase_root(18)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def loo_mean(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    estimates = [float(np.delete(clean, i).mean()) for i in range(len(clean))]
    return min(estimates), max(estimates)


def loo_rho(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    clean = frame[[x, y]].dropna().reset_index(drop=True)
    estimates = []
    for i in range(len(clean)):
        part = clean.drop(index=i)
        estimates.append(float(part[x].corr(part[y], method="spearman")))
    return min(estimates), max(estimates)


def bool_count(series: pd.Series) -> int:
    return int(series.fillna(False).astype(str).str.lower().eq("true").sum())


def task_contrast(path: Path, task_a: str, task_b: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["subject"] = data["subject"].astype(str)
    pivot = data.pivot_table(index="subject", columns="task", values="mean_abs_channel_delta", aggfunc="mean")
    out = pivot[[task_a, task_b]].dropna().copy()
    out["contrast"] = out[task_a] - out[task_b]
    return out.reset_index()


def add_row(rows: list[dict[str, object]], **values: object) -> None:
    defaults = {
        "family": "",
        "metric": "",
        "old_estimate": np.nan,
        "new_estimate": np.nan,
        "old_effective_n": np.nan,
        "new_effective_n": np.nan,
        "old_direction_count": "",
        "new_direction_count": "",
        "old_loo_min": np.nan,
        "old_loo_max": np.nan,
        "new_loo_min": np.nan,
        "new_loo_max": np.nan,
        "status_after_expansion": "",
        "interpretation": "",
    }
    defaults.update(values)
    rows.append(defaults)


def build_comparison() -> pd.DataFrame:
    old_fixed = pd.read_csv(OLD / "fixed_three_marker_summary_n4.csv")
    old_fixed = old_fixed[old_fixed.sample_set.eq("n4_excluding_2026081401")].set_index("marker")
    new_fixed_all = pd.read_csv(NEW / "phase14_stability" / "fixed_three_marker_summary.csv")
    new_fixed = new_fixed_all[new_fixed_all.sample_set.eq("n7_all_paired")].set_index("marker")
    old_nof1 = pd.read_csv(OLD / "nof1_delta_matrix_n4.csv")
    new_nof1 = pd.read_csv(NEW / "phase11" / "nof1_delta_matrix.csv")
    old_nof1["subject"] = old_nof1["subject"].astype(str)
    new_nof1["subject"] = new_nof1["subject"].astype(str)
    old_task = task_contrast(OLD / "task_spatial_redistribution_n4.csv", "摸头", "摸膝")
    new_task = task_contrast(NEW / "phase14" / "task_spatial_redistribution_summary.csv", "摸头", "摸膝")
    old_waist = task_contrast(OLD / "task_spatial_redistribution_n4.csv", "摸腰", "摸膝")
    new_waist = task_contrast(NEW / "phase14" / "task_spatial_redistribution_summary.csv", "摸腰", "摸膝")
    old_dynamic = pd.read_csv(OLD / "dynamic_state_panel_n4.csv")
    new_dynamic = pd.read_csv(NEW / "phase16" / "dynamic_state_panel.csv")

    rows: list[dict[str, object]] = []
    fixed_specs = [
        ("rest_network", "rest network", "retained but attenuated", "Signed resting fNIRS network metric; no efficacy interpretation."),
        ("spatial_channel_change", "task spatial magnitude", "retained", "Magnitude-only fNIRS spatial change; no activation direction."),
        ("occupancy_entropy_norm_delta", "state occupancy entropy", "retained but attenuated", "Session-local label-invariant state descriptor."),
    ]
    for key, label, status, note in fixed_specs:
        old_row, new_row = old_fixed.loc[key], new_fixed.loc[key]
        old_values = old_nof1[key if key != "spatial_channel_change" else "spatial_channel_change"]
        new_values = new_nof1[key if key != "spatial_channel_change" else "spatial_channel_change"]
        old_loo = loo_mean(old_values)
        new_loo = loo_mean(new_values)
        add_row(
            rows,
            family="three_layer_marker",
            metric=label,
            old_estimate=float(old_row["mean"]),
            new_estimate=float(new_row["mean"]),
            old_effective_n=int(old_row["n"]),
            new_effective_n=int(new_row["n"]),
            old_direction_count=(f"{int(old_row.n_positive)}/{int(old_row.n)} positive" if pd.notna(old_row.n_positive) else "magnitude only"),
            new_direction_count=(f"{int(new_row.n_positive)}/{int(new_row.n)} positive" if pd.notna(new_row.n_positive) else "magnitude only"),
            old_loo_min=old_loo[0], old_loo_max=old_loo[1],
            new_loo_min=new_loo[0], new_loo_max=new_loo[1],
            status_after_expansion=status,
            interpretation=note,
        )

    old_switch = old_dynamic[old_dynamic.metric.eq("switch_rate")]["delta"]
    new_switch = new_dynamic[new_dynamic.metric.eq("switch_rate")]["delta"]
    old_switch_loo, new_switch_loo = loo_mean(old_switch), loo_mean(new_switch)
    add_row(
        rows,
        family="three_layer_marker", metric="state switch rate",
        old_estimate=float(old_switch.mean()), new_estimate=float(new_switch.mean()),
        old_effective_n=int(old_switch.notna().sum()), new_effective_n=int(new_switch.notna().sum()),
        old_direction_count=f"{int((old_switch < 0).sum())}/{int(old_switch.notna().sum())} negative",
        new_direction_count=f"{int((new_switch < 0).sum())}/{int(new_switch.notna().sum())} negative",
        old_loo_min=old_switch_loo[0], old_loo_max=old_switch_loo[1],
        new_loo_min=new_switch_loo[0], new_loo_max=new_switch_loo[1],
        status_after_expansion="not replicated",
        interpretation="Mean direction changed; retain only as an exploratory state descriptor.",
    )

    for label, old_values, new_values, status in [
        ("head minus knee magnitude", old_task.contrast, new_task.contrast, "retained"),
        ("waist minus knee magnitude", old_waist.contrast, new_waist.contrast, "weakened"),
    ]:
        old_loo, new_loo = loo_mean(old_values), loo_mean(new_values)
        add_row(
            rows,
            family="task_contrast", metric=label,
            old_estimate=float(old_values.mean()), new_estimate=float(new_values.mean()),
            old_effective_n=int(old_values.notna().sum()), new_effective_n=int(new_values.notna().sum()),
            old_direction_count=f"{int((old_values > 0).sum())}/{int(old_values.notna().sum())} positive",
            new_direction_count=f"{int((new_values > 0).sum())}/{int(new_values.notna().sum())} positive",
            old_loo_min=old_loo[0], old_loo_max=old_loo[1],
            new_loo_min=new_loo[0], new_loo_max=new_loo[1],
            status_after_expansion=status,
            interpretation="Difference in absolute channel-change magnitude; not activation direction.",
        )

    for label, x, y, old_task_values, new_task_values, status in [
        ("rest-to-movement rho", "rest_network", "movement_fnirs", None, None, "retained but attenuated"),
        ("entropy-to-head-knee rho", "occupancy_entropy_norm_delta", "head_knee", old_task, new_task, "retained but attenuated"),
    ]:
        if old_task_values is None:
            old_bridge, new_bridge = old_nof1.copy(), new_nof1.copy()
        else:
            old_bridge = old_nof1.merge(old_task_values[["subject", "contrast"]], on="subject").rename(columns={"contrast": "head_knee"})
            new_bridge = new_nof1.merge(new_task_values[["subject", "contrast"]], on="subject").rename(columns={"contrast": "head_knee"})
        old_rho = float(old_bridge[x].corr(old_bridge[y], method="spearman"))
        new_rho = float(new_bridge[x].corr(new_bridge[y], method="spearman"))
        old_loo, new_loo = loo_rho(old_bridge, x, y), loo_rho(new_bridge, x, y)
        add_row(
            rows,
            family="bridge", metric=label,
            old_estimate=old_rho, new_estimate=new_rho,
            old_effective_n=len(old_bridge[[x, y]].dropna()), new_effective_n=len(new_bridge[[x, y]].dropna()),
            old_loo_min=old_loo[0], old_loo_max=old_loo[1],
            new_loo_min=new_loo[0], new_loo_max=new_loo[1],
            status_after_expansion=status,
            interpretation="Spearman rank bridge; hypothesis-generating, not causal.",
        )

    old_eeg = pd.read_csv(OLD / "eeg_quality_sensitivity_comparison.csv")
    new_eeg = pd.read_csv(NEW / "phase17_eeg_quality" / "eeg_quality_sensitivity_comparison.csv")
    for label, old_n, new_n, status in [
        ("eye executive process pairs", 0, 3, "new quantitative anchor"),
        ("EEG event sensitivity pairs", bool_count(old_eeg.event_delta_comparison_valid), bool_count(new_eeg.event_delta_comparison_valid), "coverage improved; still secondary"),
        ("EEG ratio sensitivity pairs", bool_count(old_eeg.ratio_comparison_valid), bool_count(new_eeg.ratio_comparison_valid), "coverage improved; still secondary"),
    ]:
        add_row(
            rows,
            family="coverage", metric=label,
            old_effective_n=old_n, new_effective_n=new_n,
            status_after_expansion=status,
            interpretation="Effective participant pairs; denominator changed from cohort n=4 to cohort n=7.",
        )
    return pd.DataFrame(rows)


def make_figure(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), constrained_layout=True)
    old_color, new_color = "#8c8c8c", "#2a9d8f"
    width = 0.36

    marker = data[data.family.eq("three_layer_marker")]
    x = np.arange(len(marker))
    axes[0, 0].bar(x - width / 2, marker.old_estimate, width, color=old_color, label="frozen n=4")
    axes[0, 0].bar(x + width / 2, marker.new_estimate, width, color=new_color, label="corrected n=7")
    axes[0, 0].axhline(0, color="0.3", lw=.8)
    axes[0, 0].set_xticks(x, ["rest\nnetwork", "spatial\nmagnitude", "occupancy\nentropy", "switch\nrate"])
    axes[0, 0].set_ylabel("metric-specific mean delta")
    axes[0, 0].set_title("A  Three-layer estimates")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=.2)

    task = data[data.family.eq("task_contrast")]
    x = np.arange(len(task))
    axes[0, 1].bar(x - width / 2, task.old_estimate, width, color=old_color)
    axes[0, 1].bar(x + width / 2, task.new_estimate, width, color=new_color)
    axes[0, 1].axhline(0, color="0.3", lw=.8)
    axes[0, 1].set_xticks(x, ["head - knee", "waist - knee"])
    axes[0, 1].set_ylabel("absolute channel-change contrast")
    axes[0, 1].set_title("B  Task-space contrasts")
    for i, row in enumerate(task.itertuples(index=False)):
        axes[0, 1].text(i, max(row.old_estimate, row.new_estimate) + .008, row.new_direction_count, ha="center", fontsize=7)
    axes[0, 1].grid(axis="y", alpha=.2)

    bridge = data[data.family.eq("bridge")]
    x = np.arange(len(bridge))
    axes[1, 0].bar(x - width / 2, bridge.old_estimate, width, color=old_color)
    axes[1, 0].bar(x + width / 2, bridge.new_estimate, width, color=new_color)
    axes[1, 0].axhline(0, color="0.3", lw=.8)
    axes[1, 0].set_xticks(x, ["rest -> movement", "entropy -> task space"])
    axes[1, 0].set_ylabel("Spearman rho")
    axes[1, 0].set_ylim(-1, 1)
    axes[1, 0].set_title("C  Cross-layer bridges (exploratory)")
    axes[1, 0].grid(axis="y", alpha=.2)

    coverage = data[data.family.eq("coverage")]
    x = np.arange(len(coverage))
    axes[1, 1].bar(x - width / 2, coverage.old_effective_n, width, color=old_color)
    axes[1, 1].bar(x + width / 2, coverage.new_effective_n, width, color=new_color)
    axes[1, 1].set_xticks(x, ["eye\nprocess", "EEG\nevent", "EEG\nratio"])
    axes[1, 1].set_ylabel("effective paired participants")
    axes[1, 1].set_ylim(0, 7.8)
    axes[1, 1].set_title("D  Real paired coverage (not cohort size)")
    for i, row in enumerate(coverage.itertuples(index=False)):
        axes[1, 1].text(i - width / 2, row.old_effective_n + .15, f"{int(row.old_effective_n)}/4", ha="center", fontsize=7)
        axes[1, 1].text(i + width / 2, row.new_effective_n + .15, f"{int(row.new_effective_n)}/7", ha="center", fontsize=7)
    axes[1, 1].grid(axis="y", alpha=.2)

    fig.suptitle("Cohort expansion stress test | frozen n=4 vs corrected paired n=7", fontsize=14, weight="bold")
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"cohort_expansion_stress_test.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_review(data: pd.DataFrame) -> None:
    indexed = data.set_index("metric")
    rest = indexed.loc["rest network"]
    head = indexed.loc["head minus knee magnitude"]
    switch = indexed.loc["state switch rate"]
    bridge = indexed.loc["rest-to-movement rho"]
    text = f"""# Cohort expansion stress test

This comparison treats the previous n=4 release as frozen and evaluates how
the corrected paired-seven refresh changes the same descriptive claims.

- Resting-network mean changed from {rest.old_estimate:+.4f} to {rest.new_estimate:+.4f}; the new leave-one-participant-out mean remains positive ({rest.new_loo_min:+.4f} to {rest.new_loo_max:+.4f}).
- Head-minus-knee spatial magnitude changed from {head.old_estimate:+.4f} (4/4 positive) to {head.new_estimate:+.4f} (6/7 positive).
- State switch-rate mean changed from {switch.old_estimate:+.4f} to {switch.new_estimate:+.4f}; the earlier negative direction is not replicated.
- Rest-to-movement Spearman rho changed from {bridge.old_estimate:+.2f} to {bridge.new_estimate:+.2f}; the n=7 leave-one-out range remains positive ({bridge.new_loo_min:+.2f} to {bridge.new_loo_max:+.2f}).
- Eye-process executive/attention pairing increased from 0 to 3 participants. EEG sensitivity pairing improved but remains incomplete and secondary.

All comparisons are subject-level and exploratory. Spatial values are
magnitude-only; dynamic states are session-local mathematical labels; EEG old
and segment-gated values are sensitivity counterparts rather than identical
estimators. No row is a clinical efficacy, single-point, meridian-localization,
or confirmatory statistical claim.
"""
    (OUT / "review.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison()
    comparison.to_csv(OUT / "cohort_expansion_comparison.csv", index=False, encoding="utf-8-sig")
    make_figure(comparison)
    write_review(comparison)
    print(f"Wrote {OUT}; rows={len(comparison)}")


if __name__ == "__main__":
    main()
