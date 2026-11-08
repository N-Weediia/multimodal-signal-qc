"""Formal dashboard for the frozen paired-seven work queue.

This stage is read-only with respect to raw data and previous outputs.  It
reuses subject-level summaries from the paired analysis phases, applies the frozen
seven-subject cohort, and writes a reproducible analysis dictionary alongside
the three primary result panels.
"""
from __future__ import annotations

import json
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
OUT = phase_root(16)
SUBJECTS = list(PAIRED_SUBJECTS)
N = len(SUBJECTS)
SESSIONS = ["base_line", "sec_test"]
PROTOCOL = f"n={N} | base_line\u2192sec_test | approximately two weeks | fixed combination acupuncture/electroacupuncture"
TASKS = ["touch-head", "touch-knee", "touch-waist"]
TASK_MAP = {"摸头": "touch-head", "摸膝": "touch-knee", "摸腰": "touch-waist"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def read_csv(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "subject" in frame:
        frame["subject"] = frame["subject"].astype(str)
    return frame


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def rest_panel() -> pd.DataFrame:
    source = pd.read_csv(phase_root(11) / "nof1_delta_matrix.csv")
    source["subject"] = source["subject"].astype(str)
    required = {"subject", "rest_network"}
    if not required.issubset(source.columns):
        raise ValueError(f"rest source missing columns: {sorted(required - set(source.columns))}")
    out = source[source.subject.isin(SUBJECTS)][["subject", "rest_network"]].copy()
    if out.subject.nunique() != N:
        raise ValueError(f"rest panel expected {N} subjects, got {out.subject.nunique()}")
    out["baseline_session"] = "base_line"
    out["followup_session"] = "sec_test"
    out["metric"] = "rest_network_integration_delta"
    out["interpretation"] = "signed subject-level change in resting fNIRS network integration"
    return out[["subject", "baseline_session", "followup_session", "metric", "rest_network", "interpretation"]]


def task_panel() -> pd.DataFrame:
    source = pd.read_csv(phase_root(14) / "task_spatial_redistribution_summary.csv")
    source["subject"] = source["subject"].astype(str)
    source = source[source.subject.isin(SUBJECTS)].copy()
    source["task"] = source["task"].map(TASK_MAP)
    source = source[source.task.isin(TASKS)]
    if source.subject.nunique() != N or source.task.nunique() != len(TASKS):
        raise ValueError(f"task panel does not cover the frozen n={N} x 3 task grid")
    out = source[["subject", "task", "mean_abs_channel_delta", "left_right_difference", "n_valid_channels", "coordinate_match_baseline_sec_test"]].copy()
    out["baseline_session"] = "base_line"
    out["followup_session"] = "sec_test"
    out["task_definition"] = "fixed protocol task; subject-level paired fNIRS channel summary"
    return out[["subject", "baseline_session", "followup_session", "task", "mean_abs_channel_delta", "left_right_difference", "n_valid_channels", "coordinate_match_baseline_sec_test", "task_definition"]]


def dynamic_panel() -> pd.DataFrame:
    source = pd.read_csv(phase_root(8) / "state_transition_metrics.csv")
    source["subject"] = source["subject"].astype(str)
    source = source[source.subject.isin(SUBJECTS) & source.session.isin(SESSIONS)].copy()
    # Formal panel uses the common first-19-event window; all_observed is
    # retained only in the upstream exploratory output and is excluded here.
    source = source[source.scope.eq("core19")]
    metrics = ["occupancy_entropy_norm", "occupancy_imbalance_abs", "switch_rate", "persistence_index", "transition_entropy_norm", "mean_dwell_events"]
    missing = set(metrics) - set(source.columns)
    if missing:
        raise ValueError(f"dynamic source missing columns: {sorted(missing)}")
    if source.subject.nunique() != N or source.session.nunique() != 2:
        raise ValueError(f"dynamic panel does not cover n={N} paired sessions")
    rows = []
    for sid in SUBJECTS:
        pair = source[source.subject.eq(sid)].set_index("session")
        for metric in metrics:
            b = float(pair.loc["base_line", metric])
            f = float(pair.loc["sec_test", metric])
            rows.append({"subject": sid, "scope": "core19", "metric": metric, "baseline": b, "sec_test": f, "delta": f - b, "comparison": "label-invariant exploratory descriptor; common first 19 events"})
    return pd.DataFrame(rows)


def make_dashboard(rest: pd.DataFrame, task: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    # Panel A: resting fNIRS network integration change.
    ax = axes[0]
    vals = rest.set_index("subject").loc[SUBJECTS, "rest_network"].to_numpy(float)
    x = np.arange(N)
    ax.axhline(0, color="0.45", lw=0.8)
    ax.scatter(x, vals, s=58, color="#2166ac", edgecolor="white", linewidth=0.7, zorder=3)
    ax.plot([x.min(), x.max()], [vals.mean(), vals.mean()], color="#16324f", lw=1.4, label=f"mean={vals.mean():.3f}")
    ax.set_xticks(x, [s[-4:] for s in SUBJECTS])
    ax.set_ylabel("integration delta")
    ax.set_title(f"A  Resting fNIRS network | n={N} | base_line\u2192sec_test")
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.2)

    # Panel B: task-space ranking using magnitude-preserving spatial summary.
    ax = axes[1]
    means = task.groupby("task")["mean_abs_channel_delta"].mean().reindex(TASKS)
    sem = task.groupby("task")["mean_abs_channel_delta"].agg(lambda z: np.std(z, ddof=1) / np.sqrt(len(z))).reindex(TASKS)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    ax.bar(np.arange(3), means, yerr=sem, color=colors, edgecolor="0.25", linewidth=0.5, capsize=3)
    for i, t in enumerate(TASKS):
        y = task.loc[task.task.eq(t), "mean_abs_channel_delta"].to_numpy(float)
        ax.scatter(np.full(len(y), i), y, color="0.2", s=18, zorder=3)
    ax.set_xticks(np.arange(3), ["head", "knee", "waist"])
    ax.set_ylabel("mean |fNIRS channel delta|")
    task_n = task.groupby("task")["mean_abs_channel_delta"].count().reindex(TASKS)
    task_n_label = ", ".join(f"{task_name.replace('touch-', '')}={int(task_n[task_name])}" for task_name in TASKS)
    ax.set_title(f"B  Task-space ranking | cohort n={N}; effective n: {task_n_label}")
    ax.grid(axis="y", alpha=0.2)

    # Panel C: compact dynamic-state delta matrix.
    ax = axes[2]
    focus = ["occupancy_entropy_norm", "switch_rate", "persistence_index", "transition_entropy_norm"]
    labels = ["occupancy entropy", "switch rate", "persistence", "transition entropy"]
    mat = dynamic[dynamic.metric.isin(focus)].pivot(index="metric", columns="subject", values="delta").reindex(index=focus, columns=SUBJECTS).to_numpy(float)
    lim = max(float(np.nanmax(np.abs(mat))), 1e-6)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(np.arange(len(focus)), labels)
    ax.set_xticks(np.arange(N), [s[-4:] for s in SUBJECTS], rotation=45, ha="right")
    ax.set_title(f"C  Dynamic state delta (exploratory) | n={N} | base_line\u2192sec_test")
    fig.colorbar(im, ax=ax, shrink=0.82, label="sec_test - base_line")
    fig.suptitle(f"Formal dashboard | {PROTOCOL}", fontsize=12)
    save_figure(fig, "formal_dashboard")


def write_dictionary(rest: pd.DataFrame, task: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    dictionary = {
        "dataset": {"subjects": SUBJECTS, "n": N, "sessions": SESSIONS, "contrast": "sec_test minus base_line", "interval": "approximately two weeks", "intervention": "fixed combination acupuncture/electroacupuncture", "unit_of_analysis": "participant"},
        "panels": {
            "resting_fnirs_network": {"role": "primary", "source": "paired7 phase11/nof1_delta_matrix.csv", "metric": "rest_network", "definition": "subject-level signed change in resting fNIRS network integration", "rows": int(len(rest))},
            "task_space": {"role": "primary", "source": "paired7 phase14/task_spatial_redistribution_summary.csv", "metric": "mean_abs_channel_delta", "definition": "mean absolute paired channel delta, summarized by fixed touch-head/touch-knee/touch-waist task", "rows": int(len(task)), "effective_n_by_task": {task_name: int(task.loc[task.task.eq(task_name), "mean_abs_channel_delta"].notna().sum()) for task_name in TASKS}, "directional_proxy": "left_right_difference is probe-coordinate only"},
            "dynamic_state": {"role": "primary", "source": "paired7 phase8/state_transition_metrics.csv", "scope": "core19", "scope_definition": "common first 19 observed events per paired session", "excluded_scope": "all_observed", "metrics": ["occupancy_entropy_norm", "occupancy_imbalance_abs", "switch_rate", "persistence_index", "transition_entropy_norm", "mean_dwell_events"], "rows": int(len(dynamic)), "validity": "label-invariant exploratory descriptors; state labels are session-local"},
        },
        "guardrails": ["No channel/event/block is treated as an independent sample.", "No inferential p-values or multiplicity claims are made in the formal panels.", "Previous outputs and raw data are read-only inputs.", "All plot text is ASCII/DejaVu Sans to avoid missing-glyph boxes."],
    }
    (OUT / "formal_analysis_dictionary.json").write_text(json.dumps(dictionary, ensure_ascii=True, indent=2), encoding="utf-8")
    pd.DataFrame([{"panel": "resting_fnirs_network", "n": N, "metric": "rest_network", "source": "paired7/phase11"}, {"panel": "task_space", "n": N, "metric": "mean_abs_channel_delta", "source": "paired7/phase14"}, {"panel": "dynamic_state", "n": N, "metric": "delta descriptors", "source": "paired7/phase8"}]).to_csv(OUT / "dashboard_panel_index.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rest = rest_panel()
    task = task_panel()
    dynamic = dynamic_panel()
    rest.to_csv(OUT / "resting_fnirs_network_panel.csv", index=False, encoding="utf-8-sig")
    task.to_csv(OUT / "task_space_panel.csv", index=False, encoding="utf-8-sig")
    dynamic.to_csv(OUT / "dynamic_state_panel.csv", index=False, encoding="utf-8-sig")
    make_dashboard(rest, task, dynamic)
    write_dictionary(rest, task, dynamic)
    print(f"Wrote {OUT}; n={N}; rest_rows={len(rest)}; task_rows={len(task)}; dynamic_rows={len(dynamic)}")


if __name__ == "__main__":
    main()
