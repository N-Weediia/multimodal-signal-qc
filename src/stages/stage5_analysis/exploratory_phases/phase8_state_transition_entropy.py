"""Phase 8: dynamic state occupancy and transition-structure exploration.

The input state sequence is an exploratory PCA+k-means representation of
fNIRS event vectors. Events are retained only to form a subject/session
sequence; they are never treated as independent participants. This script
adds occupancy, dwell-time and transition-entropy descriptors to the existing
switch-rate/silhouette summaries.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, phase_root


ROOT = Path(__file__).resolve().parents[1]
INPUT = CANONICAL_ROOT / "fnirs_event_states.csv"
OUT = phase_root(8)
PAIRED = list(PAIRED_SUBJECTS)
SESSIONS = {"base_line", "sec_test"}


def entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def sequence_metrics(states: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    states = np.asarray(states, dtype=int)
    states = states[np.isin(states, [1, 2])]
    n = len(states)
    transition = np.zeros((2, 2), dtype=int)
    if n > 1:
        for a, b in zip(states[:-1], states[1:]):
            transition[a - 1, b - 1] += 1
    transition_probs = transition / transition.sum() if transition.sum() else np.zeros((2, 2), dtype=float)
    switch_count = int(transition[0, 1] + transition[1, 0])
    occupancy = np.array([(states == 1).mean() if n else np.nan, (states == 2).mean() if n else np.nan])
    runs: list[int] = []
    if n:
        start = 0
        for i in range(1, n):
            if states[i] != states[i - 1]:
                runs.append(i - start)
                start = i
        runs.append(n - start)
    # Conditional transition uncertainty, averaged over occupied source states.
    conditional = []
    for row in transition:
        total = row.sum()
        conditional.append(entropy(row / total) if total else np.nan)
    row_weights = transition.sum(axis=1) / transition.sum() if transition.sum() else np.zeros(2)
    finite_cond = np.nan_to_num(np.asarray(conditional), nan=0.0)
    conditional_entropy = float((row_weights * finite_cond).sum()) if transition.sum() else 0.0
    metrics = {
        "n_events": float(n),
        # State identities are session-local; only symmetric occupancy
        # descriptors are used for across-session comparisons below.
        "state1_occupancy_session_local": float(occupancy[0]),
        "state2_occupancy_session_local": float(occupancy[1]),
        "occupancy_entropy_norm": float(entropy(occupancy)),
        "occupancy_imbalance_abs": float(abs(occupancy[0] - occupancy[1])),
        "switch_rate": float(switch_count / (n - 1)) if n > 1 else np.nan,
        "persistence_index": float(1 - switch_count / (n - 1)) if n > 1 else np.nan,
        "mean_dwell_events": float(np.mean(runs)) if runs else np.nan,
        "max_dwell_events": float(np.max(runs)) if runs else np.nan,
        "dwell_cv": float(np.std(runs, ddof=1) / np.mean(runs)) if len(runs) > 1 and np.mean(runs) else np.nan,
        "transition_entropy_norm": float(entropy(transition_probs) / 2.0),
        "conditional_transition_entropy_norm": float(conditional_entropy),
        "stay_fraction": float(np.trace(transition) / transition.sum()) if transition.sum() else np.nan,
        "switch_fraction": float(switch_count / transition.sum()) if transition.sum() else np.nan,
        "within_state_stay_imbalance_abs": float(abs(transition[0, 0] - transition[1, 1]) / transition.sum()) if transition.sum() else np.nan,
        "switch_reciprocity_imbalance_abs": float(abs(transition[0, 1] - transition[1, 0]) / switch_count) if switch_count else 0.0,
    }
    return metrics, transition


def load_metrics() -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    d = pd.read_csv(INPUT)
    d["subject"] = d["subject"].astype(str)
    d = d[d.session.isin(SESSIONS)].copy()
    rows: list[dict[str, object]] = []
    matrices: dict[tuple[str, str, str], np.ndarray] = {}
    for (subject, session), group in d.groupby(["subject", "session"], sort=True):
        group = group.sort_values("event_index")
        states = group["state"].to_numpy()
        scopes = {"all_observed": states, "core19": states[:19] if len(states) >= 19 else states}
        for scope, scoped_states in scopes.items():
            metrics, transition = sequence_metrics(scoped_states)
            rows.append({"subject": subject, "session": session, "scope": scope, **metrics})
            matrices[(subject, session, scope)] = transition
    return pd.DataFrame(rows), matrices


def exact_signflip_mean_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    observed = abs(float(values.mean()))
    null = [abs(float((values * np.asarray(signs)).mean())) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def paired_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = metrics[metrics.subject.isin(PAIRED)].pivot(index="subject", columns="session")
    rows: list[dict[str, object]] = []
    metric_cols = [c for c in metrics.columns if c not in {"subject", "session", "scope"}]
    session_local = {"state1_occupancy_session_local", "state2_occupancy_session_local"}
    for metric in metric_cols:
        base = wide.get((metric, "base_line"), pd.Series(dtype=float))
        follow = wide.get((metric, "sec_test"), pd.Series(dtype=float))
        delta = (follow - base).reindex(PAIRED).to_numpy(dtype=float)
        finite = delta[np.isfinite(delta)]
        rows.append({
            "metric": metric,
            "comparison_validity": "session_local_raw_only" if metric in session_local else "label_invariant_exploratory",
            "n_paired": int(len(finite)),
            "mean_delta": float(np.mean(finite)) if len(finite) else np.nan,
            "median_delta": float(np.median(finite)) if len(finite) else np.nan,
            "sd_delta": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
            "positive_count": int((finite > 0).sum()) if len(finite) else 0,
            "negative_count": int((finite < 0).sum()) if len(finite) else 0,
            "signflip_mean_p": exact_signflip_mean_p(finite),
        })
    summary = pd.DataFrame(rows)
    keep = ["subject"]
    delta_rows: list[dict[str, object]] = []
    for subject in PAIRED:
        row: dict[str, object] = {"subject": subject}
        for metric in metric_cols:
            try:
                row[f"{metric}_baseline"] = float(wide.loc[subject, (metric, "base_line")])
                row[f"{metric}_sec_test"] = float(wide.loc[subject, (metric, "sec_test")])
                row[f"{metric}_delta"] = row[f"{metric}_sec_test"] - row[f"{metric}_baseline"]
            except (KeyError, TypeError, ValueError):
                row[f"{metric}_baseline"] = np.nan
                row[f"{metric}_sec_test"] = np.nan
                row[f"{metric}_delta"] = np.nan
        delta_rows.append(row)
    return pd.DataFrame(delta_rows), summary


def make_figure(metrics: pd.DataFrame, delta: pd.DataFrame, matrices: dict[tuple[str, str, str], np.ndarray]) -> None:
    focus = ["occupancy_entropy_norm", "mean_dwell_events", "switch_rate", "transition_entropy_norm"]
    labels = ["Occupancy entropy", "Mean dwell (events)", "Switch rate", "Transition entropy"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axs[0, 0]
    x = np.arange(len(PAIRED))
    for metric, label, color in zip(focus, labels, ["#1976a2", "#2a9d8f", "#d95f02", "#6a51a3"]):
        vals = []
        for subject in PAIRED:
            b = metrics[(metrics.subject == subject) & (metrics.session == "base_line")][metric]
            f = metrics[(metrics.subject == subject) & (metrics.session == "sec_test")][metric]
            vals.append([b.iloc[0] if len(b) else np.nan, f.iloc[0] if len(f) else np.nan])
        vals = np.asarray(vals, dtype=float)
        # Standardize within metric only for the compact multi-metric overview.
        z = (vals - np.nanmean(vals)) / (np.nanstd(vals) or 1.0)
        for i in range(len(PAIRED)):
            ax.plot([i - 0.11, i + 0.11], z[i], color=color, alpha=.65, lw=1)
            ax.scatter([i - 0.11, i + 0.11], z[i], color=color, s=20, zorder=3)
    ax.axhline(0, color="#999999", lw=.8)
    ax.set_xticks(x, [s[-4:] for s in PAIRED])
    ax.set_ylabel("within-metric standardized value")
    ax.set_title(f"Dynamic state descriptors: paired slopes (n={len(PAIRED)})")
    handles = [plt.Line2D([0], [0], color=c, marker="o", lw=1, label=l) for c, l in zip(["#1976a2", "#2a9d8f", "#d95f02", "#6a51a3"], labels)]
    ax.legend(handles=handles, fontsize=7, loc="best")

    ax = axs[0, 1]
    heat = delta[[f"{m}_delta" for m in focus]].to_numpy(dtype=float).T
    scale = np.nanmax(np.abs(heat)) or 1.0
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set_yticks(range(len(focus)), labels)
    ax.set_xticks(range(len(PAIRED)), [s[-4:] for s in PAIRED], rotation=45, ha="right")
    ax.set_title("Follow-up minus baseline (raw delta)")
    fig.colorbar(im, ax=ax, shrink=.8, label="delta")

    ax = axs[1, 0]
    profile = metrics[metrics.subject.isin(PAIRED)].groupby("session")[["stay_fraction", "switch_fraction"]].mean().reindex(["base_line", "sec_test"])
    bottom = np.zeros(2)
    for col, label, color in zip(profile.columns, ["stay", "switch"], ["#2a9d8f", "#e76f51"]):
        vals = profile[col].to_numpy(dtype=float)
        ax.bar([0, 1], vals, bottom=bottom, color=color, width=.58, label=label)
        bottom += vals
    ax.set_xticks([0, 1], ["baseline", "sec_test"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("pooled transition proportion")
    ax.set_title("Label-invariant transition profile")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 1]
    occ = metrics[metrics.subject.isin(PAIRED)].groupby("session")[["occupancy_entropy_norm", "occupancy_imbalance_abs"]].mean().reindex(["base_line", "sec_test"])
    x2 = np.arange(2)
    width = .34
    ax.bar(x2 - width / 2, occ["occupancy_entropy_norm"], width, color="#2a9d8f", label="occupancy entropy")
    ax.bar(x2 + width / 2, occ["occupancy_imbalance_abs"], width, color="#e9c46a", label="occupancy imbalance")
    ax.set_xticks([0, 1], ["baseline", "sec_test"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean normalized descriptor")
    ax.set_title("Label-invariant occupancy structure")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Dynamic fNIRS state structure | baseline -> sec_test | exploratory paired n={len(PAIRED)}", fontsize=14)
    for suffix, kwargs in [("png", {"dpi": 600}), ("svg", {}), ("pdf", {})]:
        fig.savefig(OUT / f"state_transition_entropy.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, matrices = load_metrics()
    # Compare the common first 19 events so optional later grip blocks do not
    # alter the task composition between baseline and sec_test.
    core = metrics[metrics.scope.eq("core19")].copy()
    delta, summary = paired_tables(core)
    metrics.to_csv(OUT / "state_transition_metrics.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUT / "state_transition_delta_matrix.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "state_transition_summary.csv", index=False, encoding="utf-8-sig")
    make_figure(core, delta, matrices)
    review = """# Phase 8 dynamic state-structure review

Independent unit: seven paired subjects (baseline -> sec_test). Event states
were only used as an ordered sequence within each recording. The state labels
are mathematical PCA+k-means states and are not named as meridian, cognitive,
or treatment states.

This phase expands switch rate with label-invariant state occupancy entropy,
occupancy imbalance, mean/max dwell length, dwell CV, transition entropy,
conditional transition entropy, and stay/switch proportions. Session-local
state 1/2 occupancy and directed transition counts are retained only as raw
descriptors and are not compared across sessions because independent PCA and
k-means fits can exchange labels. Exact sign-flip p values in
`state_transition_summary.csv` are small-sample diagnostics, not confirmatory
tests. The pooled transition panel is a visualization aid, not a population
transition matrix. `sec` duplicate data for 2026081403 were excluded by using
only canonical baseline and sec_test sessions. The main paired comparison and
figure use the common first 19 events (`scope=core19`) so optional later grip
blocks do not change task composition. `scope=all_observed` remains in the raw
metrics file as a supplementary sensitivity view.

The figure's intended claim is limited to candidate reconfiguration of dynamic
state structure across the approximately two-week follow-up under the fixed
combination acupuncture/electroacupuncture protocol. It does not establish causality,
clinical efficacy, or a point-specific acupuncture mechanism. Replication with
more sessions and behavior-anchored state labels is required.
"""
    (OUT / "review.md").write_text(review, encoding="utf-8")
    print(f"rows={len(metrics)} paired={len(delta)} output={OUT.resolve()}")


if __name__ == "__main__":
    main()
