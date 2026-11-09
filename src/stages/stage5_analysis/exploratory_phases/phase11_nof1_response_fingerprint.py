"""Phase 11: N-of-1 multimodal response fingerprints with robust uncertainty.

The configured paired subjects are the only independent units. Event/channel/task
rows are first reduced to subject-level deltas; they are never treated as n.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import phase_root

ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(11)
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(1102)
NON_DIRECTIONAL_MARKERS = {"spatial_channel_change"}


def read(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def signflip_p(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan
    obs = abs(np.mean(x)); vals = [abs(np.mean(x * np.asarray(s))) for s in itertools.product([-1, 1], repeat=len(x))]
    return float(np.mean(np.asarray(vals) >= obs - 1e-12))


def bootstrap_ci(x: np.ndarray, B: int = 10000, seed: int = 1102) -> tuple[float, float]:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan, np.nan
    r = np.random.default_rng(seed).choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return tuple(np.nanpercentile(r, [2.5, 97.5]))


def robust_zframe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        med = x.median(); mad = (x - med).abs().median(); scale = 1.4826 * mad
        if not np.isfinite(scale) or scale == 0: scale = x.std(ddof=1)
        z[c] = (x - med) / scale if np.isfinite(scale) and scale > 0 else 0.0
    return z


def aggregate_task_delta(df: pd.DataFrame, metrics: list[str], prefix: str) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["subject"] + [f"{prefix}_{m}" for m in metrics])
    d = df.copy(); d["subject"] = d["subject"].astype(str)
    if "delta" not in d: return pd.DataFrame(columns=["subject"] + [f"{prefix}_{m}" for m in metrics])
    d = d[d.metric.isin(metrics)]
    out = d.groupby(["subject", "metric"], as_index=False).delta.mean().pivot(index="subject", columns="metric", values="delta").reset_index()
    out = out.rename(columns={m: f"{prefix}_{m}" for m in metrics})
    return out


def build_matrix() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    c6 = pd.read_csv(phase_root(6) / "crossmodal_delta_matrix.csv")
    c6["subject"] = c6["subject"].astype(str)
    # Exclude report-only fields and precomputed z/composites to avoid circularity.
    base_map = {
        "rest_network_integration_delta": "rest_network",
        "movement_response_delta": "movement_fnirs",
        "rsa_task_separation_delta": "rsa_separation",
        "spatial_task_dispersion_delta": "spatial_dispersion",
        "spatial_abs_channel_delta": "spatial_channel_change",
        "state_silhouette_delta": "state_silhouette",
        "state_switch_delta": "state_switch_rate",
        "emg_delta": "emg_global",
        "emg_bridge_delta": "emg_bridge",
        "eeg_delta": "eeg_exploratory",
    }
    keep = ["subject"] + [c for c in base_map if c in c6]
    m = c6[keep].rename(columns=base_map).copy()
    phase8 = pd.read_csv(phase_root(8) / "state_transition_delta_matrix.csv")
    phase8["subject"] = phase8["subject"].astype(str)
    trans = ["occupancy_entropy_norm_delta", "persistence_index_delta", "transition_entropy_norm_delta", "dwell_cv_delta", "stay_fraction_delta", "switch_fraction_delta"]
    if not phase8.empty:
        m = m.merge(phase8[["subject"] + [x for x in trans if x in phase8]], on="subject", how="left")
    c10 = pd.read_csv(phase_root(10) / "coupling_delta_matrix.csv")
    coup = aggregate_task_delta(c10, ["envelope_abs_rho", "coactivation_abs_rho"], "coupling_mean")
    m = m.merge(coup, on="subject", how="left")
    p7 = pd.read_csv(phase_root(7) / "block_dynamics_matrix.csv")
    p7["subject"] = p7["subject"].astype(str)
    p7cols = ["delta_coactivation_slope", "delta_coactivation_half_delta", "delta_coactivation_cv", "delta_envelope_cv", "delta_state_switch_rate"]
    if not p7.empty:
        p7a = p7.groupby("subject")[[x for x in p7cols if x in p7]].mean().reset_index().rename(columns={x: f"block_{x[6:]}" for x in p7cols if x in p7})
        m = m.merge(p7a, on="subject", how="left")
    # Restrict to subjects with baseline and sec_test in the canonical phase6 table.
    subjects = sorted(m.subject.dropna().astype(str).unique())
    markers = [c for c in m.columns if c != "subject" and pd.to_numeric(m[c], errors="coerce").notna().sum() >= 3]
    return m.sort_values("subject").reset_index(drop=True), markers, base_map


def marker_summary(m: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    rows = []
    for c in markers:
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(float); x = x[np.isfinite(x)]
        if len(x) < 2: continue
        med = float(np.median(x)); signs = np.sign(x[np.sign(x) != 0]); concord = np.mean(signs == np.sign(med)) if len(signs) else np.nan
        lo, hi = bootstrap_ci(x)
        loo = [np.mean(np.delete(x, i)) for i in range(len(x))] if len(x) >= 3 else []
        zz = robust_zframe(pd.DataFrame({c: x}), [c])[c]
        magnitude_only = c in NON_DIRECTIONAL_MARKERS
        rows.append({"marker": c, "n_subjects": len(x), "mean_delta": np.mean(x), "median_delta": med, "sd_delta": np.std(x, ddof=1) if len(x)>1 else np.nan, "positive_count": int((x>0).sum()) if not magnitude_only else np.nan, "negative_count": int((x<0).sum()) if not magnitude_only else np.nan, "directionality": "magnitude_only" if magnitude_only else "signed", "direction_concordance": np.nan if magnitude_only else concord, "robust_z_mean": np.mean(zz), "bootstrap_mean_ci95_low": lo, "bootstrap_mean_ci95_high": hi, "exact_signflip_p_mean": np.nan if magnitude_only else signflip_p(x), "loo_mean_min": min(loo) if loo else np.nan, "loo_mean_max": max(loo) if loo else np.nan})
    return pd.DataFrame(rows).sort_values("marker")


def plot(m: pd.DataFrame, markers: list[str], summary: pd.DataFrame) -> None:
    order = [x for x in ["rest_network", "movement_fnirs", "rsa_separation", "state_silhouette", "state_switch_rate", "spatial_dispersion", "spatial_channel_change", "coupling_mean_envelope_abs_rho", "coupling_mean_coactivation_abs_rho", "emg_global", "emg_bridge", "eeg_exploratory", "occupancy_entropy_norm_delta", "persistence_index_delta", "transition_entropy_norm_delta"] if x in markers]
    z = robust_zframe(m, order)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    lim = np.nanmax(np.abs(z[order].to_numpy())) if order else 1
    im = ax[0].imshow(z[order].T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0].set_yticks(range(len(order)), order, fontsize=8); ax[0].set_xticks(range(len(m)), m.subject, rotation=45, ha="right", fontsize=8); ax[0].set_title(f"Robust-z response fingerprint\n(subject units; n={len(m)})"); fig.colorbar(im, ax=ax[0], shrink=.7, label="within-marker robust z")
    # Paired dot/interval panel for selected axis markers.
    sel = [x for x in ["rest_network", "rsa_separation", "state_silhouette", "state_switch_rate", "coupling_mean_envelope_abs_rho"] if x in order]
    for j, c in enumerate(sel):
        x = pd.to_numeric(m[c], errors="coerce").dropna().to_numpy(float); ax[1].scatter(np.full(len(x), j), x, s=36, color="#1f4e79")
        if len(x) > 1:
            lo, hi = bootstrap_ci(x); ax[1].plot([j, j], [lo, hi], color="#e76f51", lw=3, alpha=.8)
    ax[1].axhline(0, color="0.5", lw=.7); ax[1].set_xticks(range(len(sel)), sel, rotation=30, ha="right", fontsize=8); ax[1].set_ylabel("raw baseline -> sec_test delta"); ax[1].set_title("Candidate markers with subject bootstrap CI")
    fig.suptitle("Phase 11 N-of-1 multimodal response fingerprint | exploratory", fontsize=14)
    for ext, dpi in [("png", 600), ("svg", None), ("pdf", None)]: fig.savefig(OUT / f"nof1_response_fingerprint.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    m, markers, _ = build_matrix()
    s = marker_summary(m, markers)
    z = robust_zframe(m, markers)
    for c in markers: m[f"z_{c}"] = z[c]
    # A missing-aware, equal-marker response burden: sign is preserved per marker.
    m["fingerprint_mean_robust_z"] = z[markers].mean(axis=1, skipna=True)
    m.to_csv(OUT / "nof1_delta_matrix.csv", index=False, encoding="utf-8-sig")
    s.to_csv(OUT / "marker_summary.csv", index=False, encoding="utf-8-sig")
    plot(m, markers, s)
    eye_coverage = read(str(phase_root(9).relative_to(ROOT) / "paired_eye_process_coverage.csv"))
    if "paired_eye_process_available" in eye_coverage:
        eye_pairs = int(eye_coverage["paired_eye_process_available"].astype(str).str.lower().eq("true").sum())
    else:
        # Backward-compatible fallback for runs before Phase 9 emitted
        # task-aligned process coverage.
        eye = read(str(phase_root(9).relative_to(ROOT) / "tcm_eye_raw_quantitative_tidy.csv"))
        eye_pairs = int(eye.groupby("subject").session.nunique().ge(2).sum()) if not eye.empty else 0
    report = f"""# Phase 11 N-of-1 multimodal response fingerprint

Independent units: **{len(m)} paired subjects** ({', '.join(m.subject.astype(str))}). All source event, channel, task and block rows were reduced before inference.

The matrix combines canonical phase6 multimodal deltas, phase8 state-transition deltas, phase10 task-averaged brain-muscle coupling deltas, and phase7 subject-averaged repeated-block dynamics. No acupuncture-point subgrouping or prediction model is used.

`marker_summary.csv` reports direction concordance, robust-z mean, exact sign-flip p for the mean, subject bootstrap 95% intervals and leave-one-subject-out mean ranges. These intervals quantify subject-level uncertainty; they do not turn n={len(m)} into a confirmatory study.

The largest screening-level patterns should be read from the heatmap as response fingerprints, not as validated biomarkers.

Eye/TCM quantitative pilot table exists, but only **{eye_pairs} subject(s)**
have observed, exact-task paired eye-process values; it is therefore not
included in the fingerprint correlations.

Design and interpretation: this is an approximately two-week longitudinal comparison under a fixed combination acupuncture/electroacupuncture protocol. Movement marker timestamps are automatic/accurate; task names are protocol-labeled; mathematical state labels remain session-local exploratory labels. `spatial_channel_change` is a magnitude-only descriptor because it uses absolute channel deltas; its positive values are not a signed treatment direction. EEG quality is low; fNIRS is an optical-density proxy without short-separation regression; missing EMG/brain-muscle coupling is not imputed; no single-point stratification or meridian-to-cortex claim is made.
"""
    (OUT / "review.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}; n={len(m)}; markers={len(markers)}")


if __name__ == "__main__": main()
