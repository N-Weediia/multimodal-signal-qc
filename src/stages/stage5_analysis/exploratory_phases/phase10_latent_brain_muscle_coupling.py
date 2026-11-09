"""Phase 10: latent fNIRS-event to EMG block coupling.

Each fNIRS event response is represented by the session-local PC1 score from
the existing exploratory state decomposition. It is paired with the EMG
5-second execution window at the same protocol marker. Because PC1 signs are
not comparable across independent session fits, the primary descriptor is the
absolute within-task Spearman correlation. Blocks are first reduced to a
subject/session/task descriptor, so they never inflate the independent n.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, phase_root


ROOT = Path(__file__).resolve().parents[1]
STATE = CANONICAL_ROOT / "fnirs_event_states.csv"
EMG = CANONICAL_ROOT / "movement_subtask_emg_summary.csv"
OUT = phase_root(10)
TASKS = ["摸头", "摸膝", "摸腰"]
TASK_LABELS = {"摸头": "touch-head", "摸膝": "touch-knee", "摸腰": "touch-waist"}
PAIRED = list(PAIRED_SUBJECTS)


def load_joined() -> pd.DataFrame:
    state = pd.read_csv(STATE)
    emg = pd.read_csv(EMG)
    state["subject"] = state["subject"].astype(str)
    emg["subject"] = emg["subject"].astype(str)
    state = state[state.session.isin(["base_line", "sec_test"])].copy()
    emg = emg[emg.session.isin(["base_line", "sec_test"]) & emg.task_name.isin(TASKS)].copy()
    joined = emg.merge(
        state[["subject", "session", "event_index", "pc1", "state"]],
        left_on=["subject", "session", "marker_index"],
        right_on=["subject", "session", "event_index"],
        how="inner",
    )
    return joined


def load_coupling() -> pd.DataFrame:
    joined = load_joined()
    rows: list[dict[str, object]] = []
    for (subject, session, task), group in joined.groupby(["subject", "session", "task_name"], sort=True):
        group = group.sort_values("block_index")
        result: dict[str, object] = {"subject": subject, "session": session, "task_name": task, "n_blocks": int(len(group))}
        result["pc1_state_switch_count"] = int(np.count_nonzero(np.diff(group["state"].to_numpy(dtype=int)))) if len(group) > 1 else 0
        for emg_col, out_name in [("mean_envelope", "envelope"), ("coactivation", "coactivation")]:
            x = pd.to_numeric(group["pc1"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(group[emg_col], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if good.sum() >= 3 and np.unique(x[good]).size > 1 and np.unique(y[good]).size > 1:
                rho, p = spearmanr(x[good], y[good])
                result[f"{out_name}_rho"] = float(rho)
                result[f"{out_name}_abs_rho"] = float(abs(rho))
                result[f"{out_name}_p_nominal"] = float(p)
            else:
                result[f"{out_name}_rho"] = np.nan
                result[f"{out_name}_abs_rho"] = np.nan
                result[f"{out_name}_p_nominal"] = np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def exact_signflip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    observed = abs(float(values.mean()))
    null = [abs(float((values * np.asarray(signs)).mean())) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def paired_delta(coupling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for task in TASKS:
        for metric in ["envelope_abs_rho", "coactivation_abs_rho"]:
            x = coupling[(coupling.subject.isin(PAIRED)) & coupling.task_name.eq(task)].pivot(index="subject", columns="session", values=metric)
            if "base_line" not in x or "sec_test" not in x:
                continue
            delta = (x["sec_test"] - x["base_line"]).reindex(PAIRED)
            finite = delta[np.isfinite(delta)].to_numpy(dtype=float)
            summary.append({
                "task_name": task,
                "task_label": TASK_LABELS[task],
                "metric": metric,
                "n_subjects": int(len(finite)),
                "mean_delta": float(np.mean(finite)) if len(finite) else np.nan,
                "median_delta": float(np.median(finite)) if len(finite) else np.nan,
                "positive_count": int((finite > 0).sum()) if len(finite) else 0,
                "negative_count": int((finite < 0).sum()) if len(finite) else 0,
                "exact_signflip_p_mean": exact_signflip(finite),
            })
            for subject in PAIRED:
                rows.append({"subject": subject, "task_name": task, "metric": metric, "baseline": x.loc[subject, "base_line"] if subject in x.index else np.nan, "sec_test": x.loc[subject, "sec_test"] if subject in x.index else np.nan, "delta": delta.loc[subject] if subject in delta.index else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def make_figure(coupling: pd.DataFrame, delta: pd.DataFrame, summary: pd.DataFrame, joined: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    metrics = ["envelope_abs_rho", "coactivation_abs_rho"]
    colors = ["#1b9e77", "#d95f02"]
    ax = axs[0, 0]
    x = np.arange(len(TASKS))
    width = .34
    for j, metric in enumerate(metrics):
        s = summary[summary.metric.eq(metric)].set_index("task_name").reindex(TASKS)
        vals = s["mean_delta"].to_numpy(dtype=float)
        ax.bar(x + (j - .5) * width, vals, width, color=colors[j], label=metric.replace("_abs_rho", ""))
        for xi, value in zip(x + (j - .5) * width, vals):
            if np.isfinite(value):
                ax.text(xi, value, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xticks(x, [TASK_LABELS[t] for t in TASKS], rotation=20, ha="right")
    ax.set_ylabel("delta in |within-task Spearman rho|")
    ax.set_title("Latent brain-muscle coupling change")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[0, 1]
    heat = delta.pivot_table(index=["metric", "task_name"], columns="subject", values="delta").reindex(pd.MultiIndex.from_product([metrics, TASKS]))
    mat = heat.reindex(columns=PAIRED).to_numpy(dtype=float)
    scale = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set_yticks(range(len(heat.index)), [f"{m.replace('_abs_rho','')} | {TASK_LABELS[t]}" for m, t in heat.index])
    ax.set_xticks(range(len(PAIRED)), [s[-4:] for s in PAIRED], rotation=45, ha="right")
    ax.set_title("Subject-level deltas (available n varies by task)")
    fig.colorbar(im, ax=ax, shrink=.8, label="delta")

    # A highlighted case illustrates that the descriptor is built from only
    # five blocks per task; it is not a high-dimensional population fit.
    for ax, metric, title, ycol in [(axs[1, 0], "envelope", "1403 touch-head: PC1 vs EMG envelope", "mean_envelope"), (axs[1, 1], "coactivation", "1403 touch-head: PC1 vs coactivation", "coactivation")]:
        g = joined[(joined.subject == "2026081403") & joined.task_name.eq("摸头")]
        for session, marker, color in [("base_line", "o", "#4575b4"), ("sec_test", "s", "#d73027")]:
            subset = g[g.session.eq(session)]
            ax.scatter(subset.pc1, subset[ycol], marker=marker, color=color, s=35, label=session)
            if len(subset) >= 3 and subset.pc1.nunique() > 1 and subset[ycol].nunique() > 1:
                rho = spearmanr(subset.pc1, subset[ycol]).statistic
                ax.text(.04, .92 if session == "base_line" else .84, f"{session}: rho={rho:+.2f}", transform=ax.transAxes, color=color, fontsize=8)
        ax.set_xlabel("session-local fNIRS event PC1")
        ax.set_ylabel(ycol)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(title)

    fig.suptitle("Latent fNIRS-event / EMG coupling | baseline -> sec_test | exploratory", fontsize=14)
    for suffix, kwargs in [("png", {"dpi": 600}), ("svg", {}), ("pdf", {})]:
        fig.savefig(OUT / f"latent_brain_muscle_coupling.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    joined = load_joined()
    coupling = load_coupling()
    delta, summary = paired_delta(coupling)
    joined.to_csv(OUT / "block_coupling_tidy.csv", index=False, encoding="utf-8-sig")
    coupling.to_csv(OUT / "block_coupling_summary.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUT / "coupling_delta_matrix.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "coupling_summary.csv", index=False, encoding="utf-8-sig")
    make_figure(coupling, delta, summary, joined)
    (OUT / "review.md").write_text(
        """# Latent fNIRS-event / EMG coupling review

The PC1 score is a session-local latent coordinate from the existing
exploratory fNIRS event-state decomposition. It is paired with the same
protocol marker's 5-second EMG window. The primary descriptor is absolute
within-task Spearman rho because PC1 signs may flip between independently fit
sessions. Each task has at most five blocks per subject/session; all coupling
values are reduced to subject-level descriptors before paired comparison.

This is a hypothesis-generating brain-muscle coordination marker. Lower or
higher absolute coupling cannot be called improvement, causality, or
automation without behavior, force, reaction-time and lag-calibrated models.
EMG coverage is incomplete, so task-level effective n is reported explicitly. Automatic movement
marker times are treated as accurate and task names follow the confirmed fixed
protocol order; canonical `sec` duplicate data are excluded. All participants
received the same fixed combination acupuncture/electroacupuncture protocol.
Baseline -> sec_test is approximately a two-week longitudinal comparison, not
an immediate effect.
""",
        encoding="utf-8",
    )
    print(f"rows={len(coupling)} delta_rows={len(delta)} output={OUT.resolve()}")


if __name__ == "__main__":
    main()
