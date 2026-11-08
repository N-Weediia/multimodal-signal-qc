"""Single-page leadership briefing figure for the frozen paired-seven queue.

The script reads only existing phase outputs and writes a new, auditable
figure sheet plus the exact subject-level values used in each panel.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(17)
SUBJECTS = list(PAIRED_SUBJECTS)
N = len(SUBJECTS)
SESSIONS = "base_line\u2192sec_test"
TASKS = ["touch-head", "touch-knee", "touch-waist"]
TASK_MAP = {"摸头": "touch-head", "摸膝": "touch-knee", "摸腰": "touch-waist"}
PROTOCOL = "fixed combination acupuncture/electroacupuncture"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def read_csv(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "subject" in frame.columns:
        frame["subject"] = frame["subject"].astype(str)
    return frame


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rest = pd.read_csv(phase_root(11) / "nof1_delta_matrix.csv")
    rest["subject"] = rest["subject"].astype(str)
    rest = rest[rest.subject.isin(SUBJECTS)].set_index("subject").loc[SUBJECTS]
    task = pd.read_csv(phase_root(14) / "task_spatial_redistribution_summary.csv")
    task["subject"] = task["subject"].astype(str)
    task = task[task.subject.isin(SUBJECTS)].copy()
    task["task_alias"] = task["task"].map(TASK_MAP)
    task = task[task.task_alias.isin(TASKS)]
    dyn = pd.read_csv(phase_root(16) / "dynamic_state_panel.csv")
    dyn["subject"] = dyn["subject"].astype(str)
    dyn = dyn[dyn.subject.isin(SUBJECTS)].copy()
    if rest.shape[0] != N or task.subject.nunique() != N or dyn.subject.nunique() != N:
        raise ValueError(f"Frozen n={N} cohort is incomplete in one or more source panels")
    return rest, task, dyn


def add_source(rows: list[dict], panel: str, subject: str, metric: str, value: float,
               unit: str, interpretation: str, source: str, task: str = "") -> None:
    rows.append(
        {
            "panel": panel,
            "subject": subject,
            "task": task,
            "metric": metric,
            "baseline_session": "base_line",
            "followup_session": "sec_test",
            "value": float(value),
            "unit": unit,
            "interpretation": interpretation,
            "source": source,
        }
    )


def make_figure(rest: pd.DataFrame, task: pd.DataFrame, dyn: pd.DataFrame, source_rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13, wspace=0.26, hspace=0.30)
    fig.patch.set_facecolor("#fbfaf7")
    for ax in axes.ravel():
        ax.set_facecolor("#fbfaf7")

    # A: signed resting-network change. This is a metric delta, not an
    # activation label.
    ax = axes[0, 0]
    vals = rest["rest_network"].to_numpy(float)
    x = np.arange(N)
    ax.axhline(0, color="#555555", lw=0.9)
    ax.vlines(x, 0, vals, color="#2f6690", lw=2.5, alpha=0.7)
    ax.scatter(x, vals, s=65, color="#2f6690", edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(vals.mean(), color="#d1495b", lw=1.2, ls="--", alpha=0.85)
    ax.text(0.02, 0.96, f"mean {vals.mean():+.3f}", transform=ax.transAxes, va="top", color="#d1495b", weight="bold")
    ax.set_xticks(x, [s[-4:] for s in SUBJECTS])
    ax.set_ylabel("resting network delta")
    ax.set_title("A  Resting fNIRS network change")
    ax.text(0.02, 0.03, "signed metric delta; not an activation claim", transform=ax.transAxes, fontsize=7.5, color="#555555")
    ax.grid(axis="y", alpha=0.2)
    for sid, value in zip(SUBJECTS, vals):
        add_source(source_rows, "A_resting_network", sid, "rest_network", value, "signed delta", "signed subject-level network metric delta", "paired7/phase11/nof1_delta_matrix.csv")

    # B: fixed protocol task-space ranking, preserving magnitude-only wording.
    ax = axes[0, 1]
    means = task.groupby("task_alias")["mean_abs_channel_delta"].mean().reindex(TASKS)
    grouped = task.groupby("task_alias")["mean_abs_channel_delta"]
    sem = grouped.agg(lambda z: z.dropna().sem()).reindex(TASKS)
    effective_n = grouped.count().reindex(TASKS).astype(int)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    xpos = np.arange(len(TASKS))
    ax.bar(xpos, means, yerr=sem, color=colors, edgecolor="#333333", linewidth=0.5, capsize=3, alpha=0.88)
    for i, task_name in enumerate(TASKS):
        y = task.loc[task.task_alias.eq(task_name)].set_index("subject").reindex(SUBJECTS)["mean_abs_channel_delta"].to_numpy(float)
        ax.scatter(np.full(len(y), i), y, color="#1f2933", s=22, zorder=3)
        for sid, value in zip(SUBJECTS, y):
            add_source(source_rows, "B_task_space", sid, "mean_abs_channel_delta", value, "absolute fNIRS channel delta", "magnitude only; does not encode activation direction", "paired7/phase14/task_spatial_redistribution_summary.csv", task_name)
    ax.set_xticks(xpos, ["head", "knee", "waist"])
    ax.set_ylabel("mean |fNIRS channel delta|")
    n_text = ", ".join(f"{name.split('-')[-1]}={effective_n[name]}" for name in TASKS)
    ax.set_title(f"B  Task-space change (magnitude only; n: {n_text})")
    ax.text(0.02, -0.14, "fixed protocol tasks; probe-coordinate summary", transform=ax.transAxes, fontsize=7.5, color="#555555", clip_on=False)
    ax.grid(axis="y", alpha=0.2)

    # C: label-invariant dynamic-state deltas, shown as a compact matrix.
    ax = axes[1, 0]
    focus = ["occupancy_entropy_norm", "switch_rate", "persistence_index", "transition_entropy_norm"]
    labels = ["occupancy entropy", "switch rate", "persistence", "transition entropy"]
    mat = dyn[dyn.metric.isin(focus)].pivot(index="metric", columns="subject", values="delta").reindex(index=focus, columns=SUBJECTS).to_numpy(float)
    lim = max(float(np.nanmax(np.abs(mat))), 1e-6)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(np.arange(len(focus)), labels)
    ax.set_xticks(np.arange(N), [s[-4:] for s in SUBJECTS])
    ax.set_title("C  Dynamic-state descriptors (exploratory)")
    ax.set_ylabel("session-local metric")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="sec_test - base_line")
    for metric, row in zip(focus, mat):
        for sid, value in zip(SUBJECTS, row):
            add_source(source_rows, "C_dynamic_state", sid, metric, value, "session-local delta", "label-invariant exploratory descriptor", "paired7/phase16/dynamic_state_panel.csv")

    # D: one open exploration candidate: rank bridge between two signed deltas.
    ax = axes[1, 1]
    x = rest["rest_network"].to_numpy(float)
    y = rest["movement_fnirs"].to_numpy(float)
    ax.axhline(0, color="#aaaaaa", lw=0.7)
    ax.axvline(0, color="#aaaaaa", lw=0.7)
    ax.scatter(x, y, s=70, color="#7b2cbf", edgecolor="white", linewidth=0.8, zorder=3)
    for sid, xv, yv in zip(SUBJECTS, x, y):
        ax.annotate(sid[-4:], (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=7)
        add_source(source_rows, "D_open_candidate", sid, "rest_network_signed_delta", xv, "signed rest network delta", "open exploratory bridge; signed metric delta", "paired7/phase11/nof1_delta_matrix.csv")
        add_source(source_rows, "D_open_candidate", sid, "rest_to_movement_rank_bridge", yv, "signed movement fNIRS delta", "open exploratory bridge; signed proxy delta, not activation direction", "paired7/phase11/nof1_delta_matrix.csv")
    if np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, slope * grid + intercept, color="#7b2cbf", lw=1.3, alpha=0.7)
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    ax.text(0.03, 0.95, f"Spearman rho = {rho:+.2f}", transform=ax.transAxes, va="top", weight="bold", color="#7b2cbf")
    ax.set_ylabel("movement fNIRS signed delta")
    ax.set_title("D  Open candidate: rest-to-movement bridge")
    ax.text(0.03, 0.03, f"hypothesis-generating only; n={N}", transform=ax.transAxes, fontsize=7.5, color="#555555")
    ax.grid(alpha=0.18)

    fig.suptitle(f"n={N} longitudinal briefing | base_line\u2192sec_test | exploratory", fontsize=15, weight="bold", x=0.03, ha="left")
    fig.text(0.03, 0.012, f"Cohort: {N} paired subjects | {PROTOCOL} | subject-level summaries | no causal or inferential claims", fontsize=8, color="#3f3f46")
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rest, task, dyn = load_sources()
    source_rows: list[dict] = []
    fig = make_figure(rest, task, dyn, source_rows)
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"briefing_figure.{ext}", bbox_inches="tight", facecolor=fig.get_facecolor(), **kwargs)
    plt.close(fig)
    pd.DataFrame(source_rows).to_csv(OUT / "source_data.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT}; n={N}; source_rows={len(source_rows)}")


if __name__ == "__main__":
    main()
