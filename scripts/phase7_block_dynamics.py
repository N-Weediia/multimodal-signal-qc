"""Phase 7: repeated-block dynamics and action adaptation.

All block-level observations are reduced to subject x session x task features
before computing baseline -> sec_test effects. Movement marker times are
automatic and treated as accurate; task labels follow the confirmed fixed
protocol in the upstream pipeline. A block is never treated as an independent
subject.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, phase_root

ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(7)
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["摸头", "摸膝", "摸腰", "握拳"]
TASK_CODES = {"摸头": "touch-head", "摸膝": "touch-knee", "摸腰": "touch-waist", "握拳": "grip"}


def load(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def slope(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce").to_numpy(float)
    ok = np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(np.polyfit(np.arange(1, len(y) + 1)[ok], y[ok], 1)[0])


def half_delta(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(y) < 2:
        return np.nan
    k = len(y) // 2
    return float(np.mean(y[k:]) - np.mean(y[:k]))


def cv(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(y) < 2 or np.nanmean(np.abs(y)) == 0:
        return np.nan
    return float(np.nanstd(y, ddof=1) / np.nanmean(np.abs(y)))


def state_features(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame(columns=["subject", "session", "task_name", "state_entropy", "state_switch_rate", "state_pc1_slope"])
    d = states.copy()
    d["subject"] = d["subject"].astype(str)
    # Fixed protocol mapping: events 3-7, 9-13, 15-19 are the three 5-block tasks.
    ranges = {"摸头": range(3, 8), "摸膝": range(9, 14), "摸腰": range(15, 20)}
    rows = []
    for (sid, ses), g in d.groupby(["subject", "session"]):
        for task, inds in ranges.items():
            z = g[g.event_index.isin(inds)].sort_values("event_index")
            st = z["state"].dropna().astype(int).to_numpy()
            if len(st) == 0:
                rows.append({"subject": sid, "session": ses, "task_name": task, "state_entropy": np.nan, "state_switch_rate": np.nan, "state_pc1_slope": np.nan})
                continue
            p = np.bincount(st) / len(st)
            ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
            sw = float(np.mean(st[1:] != st[:-1])) if len(st) > 1 else np.nan
            pc = z["pc1"].to_numpy(float)
            pcsl = slope(pd.Series(pc))
            rows.append({"subject": sid, "session": ses, "task_name": task, "state_entropy": ent, "state_switch_rate": sw, "state_pc1_slope": pcsl})
    return pd.DataFrame(rows)


def block_features(emg: pd.DataFrame, fnirs: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not emg.empty:
        emg = emg.copy()
        emg["subject"] = emg["subject"].astype(str)
        emg = emg[emg.session.isin(["base_line", "sec_test"])]
        for (sid, ses, task), g in emg.groupby(["subject", "session", "task_name"]):
            g = g.sort_values("block_index")
            rows.append({"subject": sid, "session": ses, "task_name": task, "n_blocks": int(g.block_index.nunique()), "coactivation_slope": slope(g.coactivation), "coactivation_half_delta": half_delta(g.coactivation), "coactivation_cv": cv(g.coactivation), "envelope_slope": slope(g.mean_envelope), "envelope_half_delta": half_delta(g.mean_envelope), "envelope_cv": cv(g.mean_envelope), "reported_envelope_cv_mean": pd.to_numeric(g.mean_envelope_cv, errors="coerce").mean()})
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["subject", "session", "task_name"])
    if not fnirs.empty:
        f = fnirs.copy(); f["subject"] = f["subject"].astype(str)
        f = f[f.session.isin(["base_line", "sec_test"])]
        f = f[f.task_name.isin(TASKS)]
        keep = ["subject", "session", "task_name", "mean_response_delta", "mean_left_right_asymmetry", "event_count"]
        f = f[[c for c in keep if c in f.columns]].drop_duplicates(["subject", "session", "task_name"])
        out = out.merge(f, on=["subject", "session", "task_name"], how="outer")
    sf = state_features(states)
    out = out.merge(sf, on=["subject", "session", "task_name"], how="outer")
    return out


def paired_deltas(features: pd.DataFrame) -> pd.DataFrame:
    d = features[features.task_name.isin(TASKS)].copy()
    ids = []
    for sid, g in d.groupby("subject"):
        if {"base_line", "sec_test"}.issubset(set(g.session)):
            ids.append(sid)
    d = d[d.subject.isin(ids)]
    numeric = [c for c in d.select_dtypes(include=[np.number]).columns if c not in {"n_blocks", "event_count"}]
    rows = []
    for sid in ids:
        for task in TASKS:
            b = d[(d.subject == sid) & (d.session == "base_line") & (d.task_name == task)]
            f = d[(d.subject == sid) & (d.session == "sec_test") & (d.task_name == task)]
            if b.empty and f.empty:
                continue
            row = {"subject": sid, "task_name": task}
            for c in numeric:
                bv = b[c].iloc[0] if not b.empty and c in b else np.nan
                fv = f[c].iloc[0] if not f.empty and c in f else np.nan
                row[f"baseline_{c}"] = bv; row[f"followup_{c}"] = fv; row[f"delta_{c}"] = fv - bv if pd.notna(bv) and pd.notna(fv) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def signflip_p(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    obs = abs(np.mean(x)); vals = [abs(np.mean(x * np.asarray(s))) for s in itertools.product([-1, 1], repeat=len(x))]
    return float(np.mean(np.asarray(vals) >= obs - 1e-12))


def summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [x for x in delta.columns if x.startswith("delta_")]:
        for task, g in delta.groupby("task_name"):
            x = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(float)
            if len(x) == 0: continue
            loo = [np.mean(np.delete(x, i)) for i in range(len(x))] if len(x) >= 3 else []
            rows.append({"task_name": task, "feature": c[6:], "n_subjects": len(x), "mean_delta": np.mean(x), "median_delta": np.median(x), "sd_delta": np.std(x, ddof=1) if len(x)>1 else np.nan, "positive_count": int((x>0).sum()), "negative_count": int((x<0).sum()), "exact_signflip_p_mean": signflip_p(x), "loo_mean_min": min(loo) if loo else np.nan, "loo_mean_max": max(loo) if loo else np.nan})
    return pd.DataFrame(rows)


def figure(features: pd.DataFrame, delta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"base_line": "#777777", "sec_test": "#1f4e79"}
    for task in TASKS[:3]:
        gd = delta[delta.task_name == task]
        x = pd.to_numeric(gd.get("delta_coactivation_slope"), errors="coerce")
        ax[0, 0].scatter([TASK_CODES[task]] * x.notna().sum(), x.dropna(), label=TASK_CODES[task], s=35)
    ax[0, 0].axhline(0, color="0.5", lw=.7); ax[0, 0].set_title("Coactivation slope delta by task"); ax[0, 0].set_ylabel("follow-up minus baseline"); ax[0, 0].tick_params(axis="x", rotation=25)
    ax[0, 0].text(.02, .97, "EMG coverage varies by task; see summary n", transform=ax[0, 0].transAxes, va="top", fontsize=8)
    # CV and half-delta paired distributions.
    for j, feat in enumerate(["delta_coactivation_cv", "delta_coactivation_half_delta"]):
        a = ax[0, 1] if j == 0 else ax[1, 0]
        for i, task in enumerate(TASKS[:3]):
            x = pd.to_numeric(delta.loc[delta.task_name == task, feat], errors="coerce").dropna().to_numpy(float)
            a.scatter(np.full(len(x), i), x, s=30, label=TASK_CODES[task] if j == 0 else None)
        a.axhline(0, color="0.5", lw=.7); a.set_xticks(range(3), [TASK_CODES[t] for t in TASKS[:3]], rotation=25, ha="right"); a.set_ylabel("delta"); a.set_title("Block CV delta" if j == 0 else "Early-to-late coactivation delta")
    # State switch rate delta.
    a = ax[1, 1]
    for i, task in enumerate(TASKS[:3]):
        x = pd.to_numeric(delta.loc[delta.task_name == task, "delta_state_switch_rate"], errors="coerce").dropna().to_numpy(float)
        a.scatter(np.full(len(x), i), x, s=30)
    a.axhline(0, color="0.5", lw=.7); a.set_xticks(range(3), [TASK_CODES[t] for t in TASKS[:3]], rotation=25, ha="right"); a.set_ylabel("delta"); a.set_title("Within-task state switching delta")
    fig.suptitle(f"Repeated-block dynamics and action adaptation | baseline -> sec_test | n={delta.subject.nunique()} exploratory")
    for ext, dpi in [("png", 600), ("svg", None), ("pdf", None)]: fig.savefig(OUT / f"block_dynamics.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    emg = pd.read_csv(CANONICAL_ROOT / "movement_subtask_emg_summary.csv")
    fnirs = pd.read_csv(CANONICAL_ROOT / "movement_subtask_summary.csv")
    states = pd.read_csv(CANONICAL_ROOT / "fnirs_event_states.csv")
    features = block_features(emg, fnirs, states)
    delta = paired_deltas(features)
    summary = summary_table(delta)
    features.to_csv(OUT / "block_features.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUT / "block_dynamics_matrix.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    figure(features, delta)
    ids = sorted(delta.subject.unique()) if not delta.empty else []
    report = f"""# Phase 7 repeated-block dynamics\n\nIndependent units are subject-level baseline -> sec_test deltas. Paired subjects represented: **{len(ids)}** ({', '.join(ids)}).\n\nFeatures include block-wise slope, early-to-late half difference, coefficient of variation, fNIRS task response summaries, and within-task state entropy/switching. Automatic movement marker times are treated as accurate; the confirmed protocol mapping uses events 3-7 (touch-head), 9-13 (touch-knee), and 15-19 (touch-waist). A grip task is retained if present; absent columns/blocks are left missing.\n\n`summary.csv` reports exact sign-flip tests and leave-one-out mean ranges by task and feature. These are exploratory sensitivity diagnostics; blocks are not independent observations and no confirmatory p-values are claimed. Task labels follow the confirmed fixed upstream protocol and do not require video re-labeling.\n\nThe follow-up session is approximately two weeks after baseline; deltas are longitudinal changes under the fixed combination acupuncture/electroacupuncture protocol, not immediate effects.\n\nEMG coverage is incomplete for some baseline sessions (for example, subjects with only touch-head blocks); fNIRS/state-derived rows are retained through an outer merge and missing EMG dynamics are not imputed.\n\nOutputs: `block_features.csv`, `block_dynamics_matrix.csv`, `summary.csv`, and `block_dynamics.{{png,svg,pdf}}`.\n"""
    (OUT / "review.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}; paired_subjects={len(ids)}; feature_rows={len(features)}; delta_rows={len(delta)}")


if __name__ == "__main__": main()
