"""Phase 19: quality-gated, interpretable cross-modal response graph.

This is a deliberately small subject-level model for the fixed paired-seven
queue.  It does not train a classifier.  Each node is a pre-specified
baseline-to-follow-up descriptor and each edge is a theory-led bridge between
layers of the proposed cognition-to-movement story.  Edge weights are estimated
with Spearman rank association and are stress-tested by leave-one-subject-out
fits.  A subject's edge contribution uses a leave-one-out robust z score and a
quality gate, so the subject does not determine its own explanatory weight.

The graph is hypothesis-generating.  It is not a treatment score, causal graph,
source-localized brain network, or acupuncture-point classifier.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import PAIRED_SUBJECTS, phase_root


ROOT = Path(__file__).resolve().parents[1]
OUT = phase_root(19)
SUBJECTS = list(PAIRED_SUBJECTS)
SESSIONS = ("base_line", "sec_test")

NODE_SPECS = {
    "rest_network": {"source": "phase11/nof1_delta_matrix.csv", "column": "rest_network", "modality": "fnirs", "label": "Rest fNIRS network"},
    "cognitive_state": {"source": "phase11/nof1_delta_matrix.csv", "column": "occupancy_entropy_norm_delta", "modality": "fnirs", "label": "Dynamic state entropy"},
    "task_space": {"source": "phase14/task_spatial_redistribution_summary.csv", "column": "head_minus_knee", "modality": "fnirs", "label": "Task-space magnitude"},
    "eeg_control": {"source": "phase11/nof1_delta_matrix.csv", "column": "eeg_exploratory", "modality": "eeg", "label": "EEG control proxy"},
    "emg_execution": {"source": "phase11/nof1_delta_matrix.csv", "column": "emg_global", "modality": "emg", "label": "EMG execution"},
    "brain_muscle_bridge": {"source": "phase11/nof1_delta_matrix.csv", "column": "emg_bridge", "modality": "bridge", "label": "Brain-muscle bridge"},
}

NODE_QUALITY_COLUMNS = {
    "rest_network": "quality_rest_fnirs",
    "cognitive_state": "quality_motion_fnirs",
    "task_space": "quality_motion_fnirs",
    "eeg_control": "quality_motion_eeg",
    "emg_execution": "quality_motion_emg",
    "brain_muscle_bridge": "quality_bridge",
}

# These are the proposed story's bridges, not data-driven edge selection.
EDGE_SPECS = [
    ("rest_network", "cognitive_state", "resting network -> dynamic state"),
    ("cognitive_state", "task_space", "dynamic state -> task space"),
    ("rest_network", "task_space", "resting network -> task response"),
    ("task_space", "eeg_control", "task response -> cognitive control"),
    ("eeg_control", "emg_execution", "cognitive control -> peripheral execution"),
    ("task_space", "emg_execution", "task response -> peripheral execution"),
    ("eeg_control", "brain_muscle_bridge", "EEG control -> brain-muscle bridge"),
    ("emg_execution", "brain_muscle_bridge", "execution -> brain-muscle bridge"),
]

QUALITY_COLUMNS = {
    "fnirs": "fnirs_quality_weight",
    "eeg": "eeg_quality_weight",
    "emg": "emg_quality_weight",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index, dtype=float)


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        q25, q75 = np.nanpercentile(values, [25, 75])
        scale = float((q75 - q25) / 1.349) if np.isfinite(q75 - q25) else np.nan
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan
    return center, scale if np.isfinite(scale) and scale > 0 else 1.0


def loo_robust_z(values: pd.Series, index: int) -> float:
    target = float(values.iloc[index]) if np.isfinite(values.iloc[index]) else np.nan
    if not np.isfinite(target):
        return np.nan
    reference = values.drop(values.index[index]).to_numpy(dtype=float)
    center, scale = robust_location_scale(reference)
    return float((target - center) / scale) if np.isfinite(center) else np.nan


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, int(len(x))
    rho, p = spearmanr(x, y)
    return float(rho), float(p), int(len(x))


def exact_permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided exact permutation p for the small subject-level edge."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    observed = abs(float(spearmanr(x, y).statistic))
    values = []
    for permutation in itertools.permutations(y):
        values.append(abs(float(spearmanr(x, permutation).statistic)))
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def load_task_space() -> pd.Series:
    path = phase_root(14) / "task_spatial_redistribution_summary.csv"
    table = pd.read_csv(path)
    table["subject"] = table["subject"].astype(str)
    table = table[table.subject.isin(SUBJECTS)]
    wide = table.pivot_table(index="subject", columns="task", values="mean_abs_channel_delta", aggfunc="mean")
    out = wide.get("摸头", pd.Series(dtype=float)) - wide.get("摸膝", pd.Series(dtype=float))
    out.name = "task_space"
    return out


def load_sync_quality() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "outputs" / "analysis" / "20260903_all_cases_canonical" / "movement_emg_markers.csv"
    if not path.exists():
        return pd.DataFrame(columns=["subject", "sync_pair_weight"]), pd.DataFrame()
    table = pd.read_csv(path)
    table["subject"] = table["subject"].astype(str)
    table = table[table.subject.isin(SUBJECTS)].copy()
    disagreement = pd.to_numeric(table.get("secondary_marker_disagreement_ms"), errors="coerce")
    table["disagreement_ms"] = disagreement
    rows = []
    for subject, group in table.groupby("subject", sort=True):
        finite = group.disagreement_ms[np.isfinite(group.disagreement_ms)]
        if len(finite) == 0:
            weight = 0.5
            status = "unknown_secondary_stream"
        else:
            # The 250 ms audit threshold is fixed upstream; no arbitrary
            # continuous attenuation is introduced for a few large outliers.
            weight = float(np.mean(finite <= 250.0))
            status = "all_within_250ms" if weight == 1.0 else "secondary_disagreement_present"
        rows.append({"subject": subject, "sync_pair_weight": weight, "sync_status": status, "sync_rows": int(len(finite)), "sync_median_ms": float(np.nanmedian(finite)) if len(finite) else np.nan, "sync_over_1s_fraction": float(np.mean(finite > 1000.0)) if len(finite) else np.nan})
    return pd.DataFrame(rows), table


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nof1_path = phase_root(11) / "nof1_delta_matrix.csv"
    nof1 = pd.read_csv(nof1_path)
    nof1["subject"] = nof1["subject"].astype(str)
    frame = pd.DataFrame({"subject": SUBJECTS})
    for node, spec in NODE_SPECS.items():
        if spec["source"].startswith("phase11"):
            frame[node] = frame.subject.map(nof1.set_index("subject")[spec["column"]]).astype(float)
    frame["task_space"] = frame.subject.map(load_task_space()).astype(float)

    quality_path = phase_root(4) / "quality_gated_phenotype.csv"
    quality = pd.read_csv(quality_path)
    quality["subject"] = quality["subject"].astype(str)
    paired_quality = pd.DataFrame({"subject": SUBJECTS})
    for modality, column in QUALITY_COLUMNS.items():
        q = quality.pivot_table(index="subject", columns="session", values=column, aggfunc="mean")
        values = np.minimum(q.get("base_line", np.nan), q.get("sec_test", np.nan))
        paired_quality[f"quality_{modality}"] = paired_quality.subject.map(values)

    # Prefer task-specific inventory scores. The phase-4 phenotype table is a
    # modality/session fallback and intentionally remains in the output for
    # comparison; it mixes rest, eye and movement recordings by design.
    inventory_path = ROOT / "outputs" / "analysis" / "20260903_all_cases_canonical" / "expanded_availability_inventory.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    if not inventory.empty:
        inventory["subject"] = inventory["subject"].astype(str)
        inventory["quality_score"] = pd.to_numeric(inventory["quality_score"], errors="coerce") / 100.0
        inventory["modality"] = inventory["modality"].astype(str).str.upper()
        inventory["task"] = inventory["task"].astype(str)

        def paired_task_quality(modality: str, token: str) -> pd.Series:
            sub = inventory[(inventory.modality == modality) & inventory.task.str.contains(token, case=False, na=False)]
            q = sub.pivot_table(index="subject", columns="session", values="quality_score", aggfunc="mean")
            return np.minimum(q.get("base_line", np.nan), q.get("sec_test", np.nan))

        for column, modality, token in [
            ("quality_rest_fnirs", "FNIRS", "静息"),
            ("quality_motion_fnirs", "FNIRS", "运动"),
            ("quality_motion_eeg", "EEG", "运动"),
            ("quality_motion_emg", "EMG", "运动"),
        ]:
            values = paired_task_quality(modality, token)
            fallback = paired_quality[f"quality_{'fnirs' if modality == 'FNIRS' else 'eeg' if modality == 'EEG' else 'emg'}"]
            mapped = paired_quality.subject.map(values)
            paired_quality[column] = mapped.where(mapped.notna(), fallback)
        paired_quality["quality_scope"] = "task_specific_inventory_with_modality_fallback"
    else:
        for column, fallback in [("quality_rest_fnirs", "quality_fnirs"), ("quality_motion_fnirs", "quality_fnirs"), ("quality_motion_eeg", "quality_eeg"), ("quality_motion_emg", "quality_emg")]:
            paired_quality[column] = paired_quality[fallback]
        paired_quality["quality_scope"] = "session_modality_aggregate_fallback_not_task_specific"

    visit_path = phase_root("17_eeg_quality") / "eeg_quality_gated_subject_summary.csv"
    visits = pd.read_csv(visit_path) if visit_path.exists() else pd.DataFrame()
    if not visits.empty:
        visits["subject"] = visits["subject"].astype(str)
        visits["visit_protocol_complete"] = visits["visit_protocol_complete"].astype(str).str.lower().eq("true")
        complete = visits.pivot_table(index="subject", columns="session", values="visit_protocol_complete", aggfunc="all")
        eeg_complete = complete.get("base_line", False) & complete.get("sec_test", False)
    else:
        eeg_complete = pd.Series(True, index=SUBJECTS)
    paired_quality["eeg_protocol_complete"] = paired_quality.subject.map(eeg_complete).fillna(False).astype(bool)

    sync, sync_rows = load_sync_quality()
    paired_quality = paired_quality.merge(sync[["subject", "sync_pair_weight", "sync_status", "sync_rows", "sync_median_ms", "sync_over_1s_fraction"]], on="subject", how="left")
    paired_quality["sync_pair_weight"] = paired_quality.sync_pair_weight.fillna(0.5)
    paired_quality["quality_bridge"] = np.minimum.reduce([paired_quality.quality_motion_fnirs.fillna(0.0), paired_quality.quality_motion_emg.fillna(0.0), paired_quality.quality_motion_eeg.fillna(0.0)]) * paired_quality.sync_pair_weight * paired_quality.eeg_protocol_complete.astype(float)
    paired_quality["quality_motion_eeg"] = paired_quality.quality_motion_eeg.fillna(0.0)
    paired_quality.loc[~paired_quality.eeg_protocol_complete, "quality_motion_eeg"] = 0.0
    paired_quality.loc[~paired_quality.eeg_protocol_complete, "quality_eeg"] = 0.0
    # Do not let an incomplete EEG visit masquerade as a valid EEG node.
    frame.loc[~paired_quality.eeg_protocol_complete.to_numpy(), "eeg_control"] = np.nan
    frame.loc[~paired_quality.eeg_protocol_complete.to_numpy(), "brain_muscle_bridge"] = np.nan
    return frame, paired_quality, sync_rows


def edge_statistics(frame: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q_by_subject = quality.set_index("subject")
    for left, right, rationale in EDGE_SPECS:
        x, y = frame[left].to_numpy(float), frame[right].to_numpy(float)
        rho, p_nominal, n = correlation(x, y)
        p_exact = exact_permutation_p(x, y)
        loo = []
        for i in range(len(frame)):
            mask = np.ones(len(frame), dtype=bool)
            mask[i] = False
            loo.append(correlation(x[mask], y[mask])[0])
        finite_loo = np.asarray(loo, dtype=float)
        finite_loo = finite_loo[np.isfinite(finite_loo)]
        if len(finite_loo):
            signs = np.sign(finite_loo[finite_loo != 0])
            sign_stable = bool(len(signs) > 0 and (np.all(signs > 0) or np.all(signs < 0)))
            lo, hi = float(np.min(finite_loo)), float(np.max(finite_loo))
        else:
            sign_stable, lo, hi = False, np.nan, np.nan
        q_left = q_by_subject.reindex(frame.subject)[NODE_QUALITY_COLUMNS[left]].to_numpy(float)
        q_right = q_by_subject.reindex(frame.subject)[NODE_QUALITY_COLUMNS[right]].to_numpy(float)
        edge_quality = float(np.nanmean(np.sqrt(np.clip(q_left, 0, 1) * np.clip(q_right, 0, 1)))) if np.isfinite(np.sqrt(np.clip(q_left, 0, 1) * np.clip(q_right, 0, 1))).any() else np.nan
        rows.append({"left_node": left, "right_node": right, "rationale": rationale, "rho_full": rho, "p_nominal": p_nominal, "p_exact_permutation": p_exact, "n_subjects": n, "loo_rho_min": lo, "loo_rho_max": hi, "loo_sign_stable": sign_stable, "mean_edge_quality": edge_quality})
    return pd.DataFrame(rows)


def subject_contributions(frame: pd.DataFrame, quality: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_by_subject = quality.set_index("subject")
    rows = []
    for i, subject in enumerate(frame.subject):
        for edge in edges.itertuples(index=False):
            left, right = edge.left_node, edge.right_node
            x, y = frame[left].to_numpy(float), frame[right].to_numpy(float)
            mask = np.ones(len(frame), dtype=bool); mask[i] = False
            rho_loo = correlation(x[mask], y[mask])[0]
            z_left, z_right = loo_robust_z(frame[left], i), loo_robust_z(frame[right], i)
            q_left = float(q_by_subject.loc[subject, NODE_QUALITY_COLUMNS[left]])
            q_right = float(q_by_subject.loc[subject, NODE_QUALITY_COLUMNS[right]])
            q_edge = float(np.sqrt(np.clip(q_left, 0, 1) * np.clip(q_right, 0, 1)))
            value = rho_loo * q_edge * z_left * z_right if np.isfinite(rho_loo) and np.isfinite(z_left) and np.isfinite(z_right) else np.nan
            rows.append({"subject": subject, "left_node": left, "right_node": right, "rationale": edge.rationale, "rho_loo_excluding_subject": rho_loo, "left_loo_robust_z": z_left, "right_loo_robust_z": z_right, "edge_quality": q_edge, "edge_contribution": value, "valid": bool(np.isfinite(value))})
    long = pd.DataFrame(rows)
    summary_rows = []
    for subject, group in long.groupby("subject", sort=True):
        valid = pd.to_numeric(group.edge_contribution, errors="coerce").dropna()
        node_signed = {node: 0.0 for node in NODE_SPECS}
        node_abs = {node: 0.0 for node in NODE_SPECS}
        for row in group.itertuples(index=False):
            if not np.isfinite(row.edge_contribution):
                continue
            half = float(row.edge_contribution) / 2.0
            for node in (row.left_node, row.right_node):
                node_signed[node] += half
                node_abs[node] += abs(half)
        top = group.loc[group.edge_contribution.abs().idxmax()] if len(valid) else None
        summary = {"subject": subject, "n_valid_edges": int(len(valid)), "graph_alignment_score": float(valid.mean()) if len(valid) else np.nan, "graph_abs_activity": float(valid.abs().mean()) if len(valid) else np.nan, "top_edge": f"{top.left_node}->{top.right_node}" if top is not None else "", "top_edge_contribution": float(top.edge_contribution) if top is not None else np.nan}
        for node in NODE_SPECS:
            summary[f"node_signed_{node}"] = node_signed[node] if node_signed[node] != 0 else np.nan
            summary[f"node_importance_{node}"] = node_abs[node] if node_abs[node] != 0 else np.nan
        summary_rows.append(summary)
    return long, pd.DataFrame(summary_rows)


def figures(frame: pd.DataFrame, quality: pd.DataFrame, edges: pd.DataFrame, contributions: pd.DataFrame) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(NODE_SPECS)
    for edge in edges.itertuples(index=False):
        if np.isfinite(edge.rho_full):
            graph.add_edge(edge.left_node, edge.right_node, weight=abs(edge.rho_full), sign=np.sign(edge.rho_full))
    positions = {"rest_network": (-1.0, 1.0), "cognitive_state": (0.0, 1.2), "task_space": (1.0, 0.6), "eeg_control": (0.0, 0.0), "emg_execution": (1.0, -0.7), "brain_muscle_bridge": (-0.8, -0.8)}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    ax = axes[0]
    nx.draw_networkx_nodes(graph, positions, node_color="#d9edf7", edgecolors="#1f4e79", node_size=1900, ax=ax)
    widths = [1.0 + 4.0 * graph[u][v]["weight"] for u, v in graph.edges]
    colors = ["#b2182b" if graph[u][v]["sign"] > 0 else "#2166ac" for u, v in graph.edges]
    nx.draw_networkx_edges(graph, positions, width=widths, edge_color=colors, alpha=.8, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels={n: NODE_SPECS[n]["label"] for n in graph.nodes}, font_size=8, ax=ax)
    ax.set_title("Theory-led response graph\nred=positive rank bridge; blue=negative")
    ax.axis("off")

    pivot = contributions.pivot_table(index="subject", columns=["left_node", "right_node"], values="edge_contribution", aggfunc="first").reindex(SUBJECTS)
    pivot.columns = [f"{a}->{b}" for a, b in pivot.columns]
    arr = pivot.to_numpy(float)
    limit = max(float(np.nanpercentile(np.abs(arr), 95)) if np.isfinite(arr).any() else 1.0, 1e-6)
    im = axes[1].imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=55, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(SUBJECTS)), [s[-4:] for s in SUBJECTS])
    axes[1].set_title("Leave-one-subject-out edge contributions")
    fig.colorbar(im, ax=axes[1], shrink=.8, label="quality-weighted contribution")
    fig.suptitle("Phase 19 | interpretable quality-gated multimodal graph | exploratory", fontsize=13)
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"quality_gated_multimodal_graph.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    q = quality.set_index("subject").reindex(SUBJECTS)
    fig, ax = plt.subplots(figsize=(8.3, 3.8), constrained_layout=True)
    cols = ["quality_rest_fnirs", "quality_motion_fnirs", "quality_motion_eeg", "quality_motion_emg", "quality_bridge"]
    matrix = q[cols].to_numpy(float)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace("quality_", "") for c in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(SUBJECTS)), [s[-4:] for s in SUBJECTS])
    ax.set_title("Paired quality gates used by graph nodes/edges")
    fig.colorbar(im, ax=ax, label="pair quality weight")
    for ext, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"quality_gate_matrix.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_review(frame: pd.DataFrame, quality: pd.DataFrame, edges: pd.DataFrame) -> None:
    excluded_eeg = quality.loc[~quality.eeg_protocol_complete, "subject"].astype(str).tolist()
    severe_sync = quality.loc[quality.sync_over_1s_fraction.fillna(0) > 0, "subject"].astype(str).tolist()
    stable = edges.loc[edges.loo_sign_stable, "left_node"].astype(str) + " -> " + edges.loc[edges.loo_sign_stable, "right_node"].astype(str)
    lines = [
        "# Phase 19 quality-gated multimodal graph",
        "",
        f"Independent unit: participant-level baseline -> sec_test delta; fixed paired queue n={len(SUBJECTS)}.",
        "",
        "The graph is a transparent algorithmic descriptor, not a trained classifier. Nodes are pre-specified from existing phase outputs: resting fNIRS network integration, fNIRS dynamic-state occupancy entropy, task-space head-minus-knee magnitude, EEG exploratory control proxy, EMG global execution, and the existing brain-muscle bridge.",
        "",
        "Edges encode the intended cognition -> movement story. Full-cohort Spearman weights, exact permutation diagnostics and leave-one-subject-out ranges are written to edge_statistics.csv. Each participant contribution is computed from an edge fit excluding that participant, leave-one-out robust z scores, and the participant's paired quality gate.",
        "",
        f"EEG protocol-incomplete participants were excluded from EEG and brain-muscle nodes: {', '.join(excluded_eeg) if excluded_eeg else 'none'}. This is intentional because an inferred marker stream cannot upgrade a truncated EEG visit.",
        f"",
        f"Secondary marker disagreement exceeded 1 s for: {', '.join(severe_sync) if severe_sync else 'none'}. EMG-related edges are downweighted by the fixed <=250 ms agreement fraction; no continuous attenuation or imputation is used.",
        "",
        f"LOO sign-stable edges in this small exploratory run: {', '.join(stable.tolist()) if len(stable) else 'none'}. A stable edge is a robustness flag, not a validated biomarker.",
        "",
        "Interpretation guardrails: fNIRS values are optical-density proxies without short-separation regression or source localization; EEG/fNIRS sensors are not co-registered; task-space values are magnitude-only probe-coordinate summaries; all acupuncture points remain an intervention context rather than single-point causal labels; n=7 does not support efficacy or clinical prediction claims.",
        "",
        "Outputs: node_matrix.csv, paired_quality_gates.csv, edge_statistics.csv, edge_contributions.csv, subject_graph_explanations.csv, quality_gated_multimodal_graph.{png,svg,pdf}, quality_gate_matrix.{png,svg,pdf}, metadata.json.",
    ]
    (OUT / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, quality, sync_rows = load_frame()
    edges = edge_statistics(frame, quality)
    contributions, summaries = subject_contributions(frame, quality, edges)
    frame.to_csv(OUT / "node_matrix.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT / "paired_quality_gates.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUT / "edge_statistics.csv", index=False, encoding="utf-8-sig")
    contributions.to_csv(OUT / "edge_contributions.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(OUT / "subject_graph_explanations.csv", index=False, encoding="utf-8-sig")
    figures(frame, quality, edges, contributions)
    review = {
        "subjects": SUBJECTS,
        "nodes": NODE_SPECS,
        "edges": [{"left": a, "right": b, "rationale": r} for a, b, r in EDGE_SPECS],
        "inputs": {
            "nof1": str(phase_root(11) / "nof1_delta_matrix.csv"),
            "task_space": str(phase_root(14) / "task_spatial_redistribution_summary.csv"),
            "quality": str(phase_root(4) / "quality_gated_phenotype.csv"),
            "task_specific_quality_inventory": str(ROOT / "outputs" / "analysis" / "20260903_all_cases_canonical" / "expanded_availability_inventory.csv"),
            "marker_audit": str(ROOT / "outputs" / "analysis" / "20260903_all_cases_canonical" / "movement_emg_markers.csv"),
        },
        "sha256": {"script": sha256_file(Path(__file__))},
        "rules": {"min_edge_n": 4, "sync_agreement_threshold_ms": 250, "eeg_incomplete_excluded": True, "subject_self_excluded_from_edge_and_z": True, "no_imputation": True},
        "sync_rows": int(len(sync_rows)),
        "quality_scope": str(quality.quality_scope.iloc[0]) if "quality_scope" in quality and len(quality) else "unknown",
    }
    (OUT / "metadata.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(frame, quality, edges)
    print(json.dumps({"output": str(OUT), "subjects": len(frame), "nodes": len(NODE_SPECS), "edges": len(edges), "valid_edge_contributions": int(contributions.valid.sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
