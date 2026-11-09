"""Phase 6: paired, cross-modal marker exploration.

This script treats the configured paired baseline -> sec_test subjects as the only
independent units. It consumes audited phase 3-5 CSV outputs and produces a
small, reproducible marker table, correlation diagnostics, and a leadership-
friendly exploratory figure. No event/block is treated as an observation and
no acupuncture point is attributed to a brain location.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, phase_root


ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(6)
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def paired_subjects(qd: pd.DataFrame) -> list[str]:
    ids = qd["subject"].astype(str).tolist()
    return sorted(ids)


def delta_from_rsa(rsa: pd.DataFrame) -> pd.DataFrame:
    # One task_separation_index is repeated for each pair; keep one value.
    rsa = rsa.copy()
    rsa["subject"] = rsa["subject"].astype(str)
    v = rsa.groupby(["subject", "session"], as_index=False)["task_separation_index"].first()
    wide = v.pivot(index="subject", columns="session", values="task_separation_index")
    wide["rsa_task_separation_delta"] = wide.get("sec_test") - wide.get("base_line")
    return wide[["rsa_task_separation_delta"]].reset_index()


def state_deltas(state: pd.DataFrame) -> pd.DataFrame:
    s = state.copy()
    s["subject"] = s["subject"].astype(str)
    # State output is already paired; follow-up minus baseline is explicit.
    s["state_silhouette_delta"] = s["followup_silhouette"] - s["baseline_silhouette"]
    # Lower switching is interpreted only as a candidate stability marker.
    s["state_switch_delta"] = s["switch_rate_change"]
    return s[["subject", "state_silhouette_delta", "state_switch_delta"]]


def state_effects_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical session rows to the historical paired state schema."""
    s = summary.copy()
    s["subject"] = s["subject"].astype(str)
    s = s[s.session.isin(["base_line", "sec_test"])]
    rows: list[dict[str, object]] = []
    for subject, group in s.groupby("subject", sort=True):
        pair = group.drop_duplicates("session").set_index("session")
        if not {"base_line", "sec_test"}.issubset(pair.index):
            continue
        baseline_events = float(pair.loc["base_line", "events"])
        followup_events = float(pair.loc["sec_test", "events"])
        rows.append(
            {
                "subject": subject,
                "baseline_silhouette": float(pair.loc["base_line", "silhouette"]),
                "followup_silhouette": float(pair.loc["sec_test", "silhouette"]),
                "switch_rate_change": (
                    float(pair.loc["sec_test", "state_switches"]) / max(followup_events - 1.0, 1.0)
                    - float(pair.loc["base_line", "state_switches"]) / max(baseline_events - 1.0, 1.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def quality_weights(phen: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    p = phen[phen.subject.astype(str).isin(subjects)].copy()
    p["subject"] = p["subject"].astype(str)
    cols = {
        "fnirs": "fnirs_quality_weight",
        "emg": "emg_quality_weight",
        "eeg": "eeg_quality_weight",
        "bridge": "emg_bridge_quality_weight",
    }
    out = pd.DataFrame({"subject": subjects})
    for name, col in cols.items():
        q = p.groupby("subject")[col].mean().rename(f"{name}_quality_weight")
        out = out.merge(q, on="subject", how="left")
    return out


def spatial_deltas(spatial: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    """Derive compact channel-level spatial response descriptors per subject."""
    d = spatial[(spatial.modality == "fNIRS") & spatial.subject.astype(str).isin(subjects)].copy()
    d["subject"] = d["subject"].astype(str)
    d = d[d.feature.eq("event_delta") & d.session.isin(["base_line", "sec_test"])]
    # Per subject/session/task mean response and cross-task dispersion.
    task = d.groupby(["subject", "session", "task"], as_index=False).value.mean()
    wide = task.pivot_table(index=["subject", "session"], columns="task", values="value", aggfunc="mean")
    wide["spatial_task_dispersion"] = wide.std(axis=1, ddof=1)
    # Channel-level magnitude of the baseline -> follow-up spatial change.
    ch = d.groupby(["subject", "session", "channel"], as_index=False).value.mean()
    cw = ch.pivot_table(index=["subject", "channel"], columns="session", values="value", aggfunc="mean")
    cw["abs_channel_delta"] = (cw.get("sec_test") - cw.get("base_line")).abs()
    mag = cw.groupby("subject")["abs_channel_delta"].mean().rename("spatial_abs_channel_delta")
    out = wide["spatial_task_dispersion"].unstack("session")
    out["spatial_task_dispersion_delta"] = out.get("sec_test") - out.get("base_line")
    out = out[["spatial_task_dispersion_delta"]].reset_index()
    out = out.merge(mag.reset_index(), on="subject", how="left")
    return out


def exact_signflip_p(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    """Two-sided exact sign-flip p for paired-delta sensitivity screening."""
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3:
        return np.nan
    obs = abs(corr(x, y, method)[0])
    vals = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(y)):
        vals.append(abs(corr(x, y * np.asarray(signs), method)[0]))
    return float(np.mean(np.asarray(vals) >= obs - 1e-12))


def corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> tuple[float, float]:
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    if method == "spearman":
        r, p = spearmanr(x, y)
    else:
        # Winsorized Pearson limits leverage without deleting a subject.
        lo_x, hi_x = np.nanpercentile(x, [10, 90])
        lo_y, hi_y = np.nanpercentile(y, [10, 90])
        r, p = pearsonr(np.clip(x, lo_x, hi_x), np.clip(y, lo_y, hi_y))
    return float(r), float(p)


def loo_range(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float, str]:
    vals = []
    for i in range(len(x)):
        xx, yy = np.delete(x, i), np.delete(y, i)
        vals.append(corr(xx, yy, method)[0])
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if not len(finite):
        return np.nan, np.nan, ""
    return float(np.min(finite)), float(np.max(finite)), ",".join(f"{v:.3f}" for v in vals)


def robust_z(v: pd.Series) -> pd.Series:
    med = v.median()
    mad = (v - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        scale = v.std(ddof=1)
    else:
        scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - med) / scale


def make_figure(m: pd.DataFrame, c: pd.DataFrame) -> None:
    markers = [
        "rest_network_integration_delta",
        "movement_response_delta",
        "rsa_task_separation_delta",
        "state_silhouette_delta",
        "state_switch_delta",
        "emg_delta",
        "emg_bridge_delta",
        "eeg_delta",
    ]
    labels = ["Rest fNIRS network", "Movement fNIRS", "RSA task separation", "State silhouette", "State switch rate", "EMG", "EMG bridge", "EEG"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axs[0, 0]
    mat = m.set_index("subject")[markers].T
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(abs(mat.values)), vmax=np.nanmax(abs(mat.values)))
    ax.set_yticks(range(len(markers)), labels, fontsize=8)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Paired delta heatmap (subject units; n={len(m)})")
    fig.colorbar(im, ax=ax, shrink=.75, label="delta")

    ax = axs[0, 1]
    axis_cols = ["rest_network_integration_delta", "rsa_task_separation_delta", "emg_bridge_delta"]
    axis_labels = ["Rest integration", "Action fingerprint", "Brain-muscle bridge"]
    for j, (col, lab) in enumerate(zip(axis_cols, axis_labels)):
        vals = m[col].values
        for i, val in enumerate(vals):
            ax.plot([j - .16, j + .16], [0, val], color="#8c8c8c", lw=1)
            ax.scatter(j + .16, val, s=32, color="#1f4e79")
        ax.axhline(0, color="black", lw=.6)
    ax.set_xticks(range(3), axis_labels, rotation=20, ha="right")
    ax.set_ylabel("raw paired delta")
    ax.set_title("Candidate axis components (paired slopes)")

    ax = axs[1, 0]
    x, y = m["rest_network_integration_delta"].to_numpy(), m["rsa_task_separation_delta"].to_numpy()
    ax.scatter(x, y, c="#2a9d8f", s=55, edgecolor="white", linewidth=.7)
    for sid, xx, yy in zip(m.subject, x, y):
        ax.annotate(str(sid)[-4:], (xx, yy), xytext=(4, 3), textcoords="offset points", fontsize=8)
    cc = c[(c.x == "rest_network_integration_delta") & (c.y == "rsa_task_separation_delta")]
    if len(cc):
        row = cc.iloc[0]
        ax.text(.03, .96, f"Spearman={row.spearman_r:.2f}; LOO=[{row.spearman_loo_min:.2f},{row.spearman_loo_max:.2f}]", transform=ax.transAxes, va="top", fontsize=8)
    ax.axhline(0, color="0.7", lw=.6); ax.axvline(0, color="0.7", lw=.6)
    ax.set_xlabel("Rest fNIRS network delta"); ax.set_ylabel("RSA separation delta")
    ax.set_title("Rest integration -> action fingerprint (exploratory)")

    ax = axs[1, 1]
    x, y = m["rsa_task_separation_delta"].to_numpy(), m["emg_bridge_delta"].to_numpy()
    ax.scatter(x, y, c="#e76f51", s=55, edgecolor="white", linewidth=.7)
    for sid, xx, yy in zip(m.subject, x, y):
        ax.annotate(str(sid)[-4:], (xx, yy), xytext=(4, 3), textcoords="offset points", fontsize=8)
    cc = c[(c.x == "rsa_task_separation_delta") & (c.y == "emg_bridge_delta")]
    if len(cc):
        row = cc.iloc[0]
        ax.text(.03, .96, f"Spearman={row.spearman_r:.2f}; perm p={row.spearman_perm_p:.3f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.axhline(0, color="0.7", lw=.6); ax.axvline(0, color="0.7", lw=.6)
    ax.set_xlabel("RSA separation delta"); ax.set_ylabel("EMG bridge delta")
    ax.set_title("Action fingerprint -> brain-muscle bridge (exploratory)")
    fig.suptitle(f"Phase 6 cross-modal candidate markers | baseline -> sec_test | n={len(m)}", fontsize=14)
    for ext, dpi in [("png", 600), ("svg", None), ("pdf", None)]:
        fig.savefig(OUT / f"crossmodal_markers.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    qd = pd.read_csv(phase_root(4) / "quality_gated_delta.csv")
    subjects = paired_subjects(qd)
    phen = pd.read_csv(phase_root(4) / "quality_gated_phenotype.csv")
    state = state_effects_from_summary(pd.read_csv(CANONICAL_ROOT / "fnirs_event_state_summary.csv"))
    rsa = pd.read_csv(phase_root(5) / "fnirs_task_fingerprint_rsa.csv")
    # Read spatial source tables as an audit dependency; the group values are
    # already summarized in qd, while this confirms channel-level provenance.
    spatial = pd.read_csv(phase_root(5) / "figure_spatial_source_data.csv")
    pair_ch = pd.read_csv(phase_root(5) / "fnirs_pair_channel_delta.csv")
    _ = (spatial.shape, pair_ch.shape)

    m = qd.copy()
    m["subject"] = m["subject"].astype(str)
    m = m[m.subject.isin(subjects)].copy()
    m = m.rename(columns={"fnirs_delta": "movement_response_delta"})
    m = m.merge(delta_from_rsa(rsa), on="subject", how="left")
    m = m.merge(state_deltas(state), on="subject", how="left")
    m = m.merge(spatial_deltas(spatial, subjects), on="subject", how="left")
    m = m.rename(columns={"fnirs_delta": "movement_response_delta"})
    # Rest network is available in phase3 case summary; use exact paired deltas.
    case = pd.read_csv(CANONICAL_ROOT / "expanded_case_delta_summary.csv")
    case["subject"] = case["subject"].astype(str)
    rest = case[case.feature.eq("rest_fnirs_network_mean_abs_fisher_z")][["subject", "absolute_change"]].rename(columns={"absolute_change": "rest_network_integration_delta"})
    m = m.drop(columns=["rest_network_integration_delta"], errors="ignore").merge(rest, on="subject", how="left")
    m = m.merge(quality_weights(phen, subjects), on="subject", how="left")
    m = m.sort_values("subject").reset_index(drop=True)

    marker_cols = ["rest_network_integration_delta", "movement_response_delta", "rsa_task_separation_delta", "spatial_task_dispersion_delta", "spatial_abs_channel_delta", "state_silhouette_delta", "state_switch_delta", "emg_delta", "emg_bridge_delta", "eeg_delta"]
    # Candidate composite: standardized components, reported alongside raw data.
    for col in marker_cols:
        m[f"z_{col}"] = robust_z(m[col])
    m["axis_rest_to_action"] = m[["z_rest_network_integration_delta", "z_rsa_task_separation_delta"]].mean(axis=1)
    m["axis_brain_muscle_state"] = m[["z_emg_bridge_delta", "z_state_silhouette_delta", "z_state_switch_delta"]].assign(z_state_switch_delta=lambda x: -x.z_state_switch_delta).mean(axis=1)
    m.to_csv(OUT / "crossmodal_delta_matrix.csv", index=False, encoding="utf-8-sig")

    pairs = [(a, b) for i, a in enumerate(marker_cols) for b in marker_cols[i + 1 :]]
    rows = []
    for a, b in pairs:
        x, y = m[a].to_numpy(float), m[b].to_numpy(float)
        sr, sp = corr(x, y, "spearman")
        rr, rp = corr(x, y, "robust")
        smin, smax, svals = loo_range(x, y, "spearman")
        rmin, rmax, rvals = loo_range(x, y, "robust")
        rows.append({"x": a, "y": b, "n": int(np.isfinite(x * y).sum()), "spearman_r": sr, "spearman_p_asymptotic": sp, "spearman_perm_p": exact_signflip_p(x, y, "spearman"), "spearman_loo_min": smin, "spearman_loo_max": smax, "spearman_loo_values": svals, "winsor_pearson_r": rr, "winsor_pearson_p_asymptotic": rp, "winsor_pearson_loo_min": rmin, "winsor_pearson_loo_max": rmax, "winsor_pearson_loo_values": rvals})
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "marker_summary.csv", index=False, encoding="utf-8-sig")
    make_figure(m, c)

    strongest = c.loc[c["spearman_r"].abs().idxmax()] if c["spearman_r"].notna().any() else None
    strongest_text = (
        f"The largest absolute screening correlation was {strongest.x} vs {strongest.y} "
        f"(Spearman rho={strongest.spearman_r:+.2f}, exact sign-flip p={strongest.spearman_perm_p:.3f}, n={int(strongest.n)})."
        if strongest is not None
        else "No cross-marker correlation had sufficient finite paired data."
    )
    report = f"""# Phase 6 cross-modal marker exploration

Independent units: **{len(subjects)} paired subjects** ({', '.join(subjects)}).

## Candidate findings

The candidate axis is descriptive: rest fNIRS network integration -> movement/RSA task fingerprint -> EMG bridge and state stability. Values are baseline-to-sec_test deltas; quality weights are retained in the matrix and are not used to create extra samples.

Two additional spatial descriptors come from the phase5 channel table: `spatial_task_dispersion_delta` (change in between-task response dispersion) and `spatial_abs_channel_delta` (mean absolute channel-wise change). {strongest_text} This is hypothesis-generating only and requires replication.

Use `marker_summary.csv` for Spearman, winsorized Pearson, exact sign-flip permutation (Spearman), and leave-one-out ranges. With n={len(subjects)} these are sensitivity diagnostics, not confirmatory p-values.

## Provenance and limits

- fNIRS movement delta is the phase4 quality-gated fNIRS composite (event-response-derived); rest integration is the phase3 paired Fisher-z network metric.
- RSA task separation is the phase5 three-subtask fingerprint index; task blocks are not independent observations.
- State switch/silhouette is rebuilt from the current canonical event-state summary.
- EMG/EEG are exploratory bridge channels; EEG quality is low and no causal interpretation is made.
- Eye-process and TCM raw quantitative fields are not correlated here: coverage and paired completeness are insufficient.
- All participants received the same fixed combination acupuncture/electroacupuncture protocol; no point-wise stratification is performed. Baseline -> sec_test is a longitudinal change approximately two weeks apart, not an immediate post-stimulation effect.
- Current fNIRS is a wavelength-balanced optical-density proxy without short-separation regression; EEG/fNIRS coordinates are not individually co-registered.

Outputs: `crossmodal_delta_matrix.csv`, `marker_summary.csv`, and `crossmodal_markers.{{png,svg,pdf}}`.
"""
    (OUT / "review.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}; subjects={len(subjects)}, spatial_rows={len(spatial)}, channel_delta_rows={len(pair_ch)}")


if __name__ == "__main__":
    main()
