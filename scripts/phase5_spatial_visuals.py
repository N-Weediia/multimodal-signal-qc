"""Phase 5 exploratory spatial and multimodal visualisations.

This script is deliberately descriptive.  It uses the canonical local subset
(``subject/base_line`` and ``subject/sec_test``) and never treats movement
blocks as independent observations. Movement marker times are automatic and
treated as accurate; task names follow the confirmed fixed protocol order.
EEG positions are approximate
10-10 locations; fNIRS positions are source-detector midpoints and are not
registered to EEG.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import main
from emg_qc import read_emg_csv
from explore_202608_cases import (
    EventSeries, butter_filter, eeg_features, emg_features, find_recording,
    fnirs_features, infer_movement_event_labels, read_fnirs_events,
    set_data_root,
)
from fnirs_qc import assess_fnirs_channel, read_fnirs
try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, RAW_DATA_ROOT, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, RAW_DATA_ROOT, phase_root


IDENTITY_PATH = CANONICAL_ROOT / "subject_identity_from_xlsx.json"
SUBJECT_NAMES = json.loads(IDENTITY_PATH.read_text(encoding="utf-8")) if IDENTITY_PATH.exists() else {}
HIGHLIGHT = ["2026081403", "2026081501"]
TASKS = ["摸头", "摸膝", "摸腰"]
COLORS = {"摸头": "#2a9d8f", "摸膝": "#e9c46a", "摸腰": "#e76f51"}
KEY_EEG_LABELS = {"Fz", "FCz", "Cz", "C3", "C4", "CP3", "CP4", "Pz"}


def _first(paths):
    paths = sorted(paths)
    return paths[0] if paths else None


def recording(data_root: Path, subject: str, session: str, modality: str, task: str):
    root = data_root / subject / session
    if modality == "EEG":
        candidates = [
            path
            for folder in (root / "EEG").rglob("*")
            if folder.is_dir() and task in folder.name
            for path in folder.glob("*.bdf")
            if path.name.lower() != "evt.bdf"
        ]
        return max(candidates, key=lambda path: (path.stat().st_size, str(path))) if candidates else None
    if modality == "FNIRS":
        candidates = [
            path
            for folder in (root / "FNIRS").rglob("*")
            if folder.is_dir() and task in folder.name
            for path in folder.glob("*.nirs")
        ]
        return max(candidates, key=lambda path: (path.stat().st_size, str(path))) if candidates else None
    if modality == "EMG":
        return _first((root / "EMG").glob("*.csv"))
    return None


def savefig(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _head(ax):
    ax.add_patch(Circle((0, 0), 1.18, fill=False, lw=1.0, color="#444"))
    ax.plot([-0.10, 0, 0.10], [1.16, 1.27, 1.16], color="#444", lw=.8)
    ax.plot([-1.17, -1.27, -1.17], [0.10, 0, -0.10], color="#444", lw=.8)
    ax.plot([1.17, 1.27, 1.17], [0.10, 0, -0.10], color="#444", lw=.8)
    ax.set_xlim(-1.38, 1.38); ax.set_ylim(-1.35, 1.38)
    ax.set_aspect("equal"); ax.axis("off")


def _color_limits(arrays, symmetric=True):
    vals = np.concatenate([np.ravel(np.asarray(a, float)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return (-1, 1)
    lim = float(np.nanpercentile(np.abs(vals), 95)) if symmetric else float(np.nanpercentile(vals, 95))
    lim = max(lim, 1e-9)
    return (-lim, lim) if symmetric else (float(np.nanpercentile(vals, 5)), lim)


def eeg_case(subject: str, name: str, base: Path, follow: Path, out: Path, source_rows: list[dict]):
    records = {}
    for state, path in [("baseline", base), ("sec_test", follow)]:
        if path is None: continue
        try:
            features, extra = eeg_features(path, None)
            header, _, _ = main.read_bdf(path, None)
            labels = [str(x) for x in header.labels]
            wanted = {main.normalize_label(x) for x in main.FULL_EEG_CHANNELS}
            scalp = [i for i, x in enumerate(labels) if main.normalize_label(x) in wanted]
            records[state] = (np.asarray([main.POSITION_MAP.get(labels[i], (np.nan, np.nan)) for i in scalp]),
                              [labels[i] for i in scalp], extra["band_matrix"], float(header.sample_rate))
            for band, vals in extra["band_matrix"].items():
                for label, value in zip(records[state][1], vals):
                    x, y = main.POSITION_MAP.get(label, (np.nan, np.nan))
                    source_rows.append({"subject": subject, "name": name, "session": state, "modality": "EEG", "feature": band, "channel": label, "x": x, "y": y, "value": float(value), "quality_note": "exploratory; EEG QC low and approximate 10-10 coordinates"})
        except Exception as exc:
            source_rows.append({"subject": subject, "name": name, "session": state, "modality": "EEG", "feature": "error", "value": "", "quality_note": str(exc)})
    if "baseline" not in records or "sec_test" not in records: return
    labels = records["baseline"][1]
    bands = ["theta", "alpha", "beta"]
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 9.2))
    for r, band in enumerate(bands):
        arrays = [records[s][2][band] for s in ("baseline", "sec_test")]
        delta = arrays[1] - arrays[0]
        lo, hi = _color_limits(arrays + [delta])
        for c, (title, values) in enumerate([("baseline", arrays[0]), ("sec_test", arrays[1]), ("delta", delta)]):
            ax = axes[r, c]; _head(ax)
            xy = records["baseline"][0]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="RdBu_r", vmin=lo, vmax=hi, s=70, edgecolor="k", lw=.25)
            # Full 59-channel labels make the scalp map unreadable. Keep only
            # a compact sensorimotor/frontal/parietal landmark set visible;
            # every channel remains traceable in the source CSV.
            for (x, y), label in zip(xy, labels):
                if label in KEY_EEG_LABELS:
                    ax.text(x, y + .055, label, fontsize=6, ha="center", va="bottom")
            if r == 0: ax.set_title(title, fontsize=10)
            if c == 0: ax.set_ylabel(band, fontsize=10)
            fig.colorbar(sc, ax=ax, fraction=.046, pad=.02, shrink=.8)
    fig.suptitle(f"{subject} {name} EEG scalp band power | exploratory only\nApproximate 10-10 coordinates; low-quality EEG, not a confirmatory result", fontsize=12)
    savefig(fig, out / f"eeg_topography_{subject}")


def event_task_maps(subject: str, name: str, state_paths: dict[str, Path], out: Path, source_rows: list[dict]):
    maps = {}
    quality = {}
    for state, path in state_paths.items():
        if path is None: continue
        try:
            ev = read_fnirs_events(path)
            if ev is None: continue
            labels = infer_movement_event_labels(ev.times, subject)
            _, extra = fnirs_features(path, ev)
            rec = read_fnirs(path, None)
            q = np.asarray([assess_fnirs_channel(ch, rec.sample_rate_hz).quality_score for ch in rec.channels], float)
            vec = np.asarray(extra["event_delta_vectors"], float)
            used = np.asarray(extra["event_indices_used"], int)
            by_task = {}
            for task in TASKS:
                rows = [j for j, idx in enumerate(used) if idx < len(labels) and labels[idx]["task_name"] == task]
                by_task[task] = np.nanmean(vec[rows], axis=0) if rows else np.full(vec.shape[1], np.nan)
                for ch, value in enumerate(by_task[task], 1):
                    pos = rec.channels[ch - 1].position
                    source_rows.append({"subject": subject, "name": name, "session": state, "modality": "fNIRS", "feature": "event_delta", "task": task, "channel": f"CH{ch:02d}", "x": pos[0], "y": pos[1], "value": float(value), "quality_score": float(q[ch - 1]), "quality_note": "wavelength-balanced OD proxy; automatic marker times; fixed-protocol task labels"})
            # Keep the external session name in source data, but normalize the
            # internal key so ``base_line`` matches the comparison logic below.
            map_key = "baseline" if state == "base_line" else state
            maps[map_key] = (by_task, np.asarray([c.position for c in rec.channels]), q)
        except Exception as exc:
            # Keep the batch alive, but expose a concise diagnostic for a
            # malformed recording rather than silently dropping its map.
            print(f"[fNIRS map skipped] {subject} {state}: {exc}")
            source_rows.append({"subject": subject, "name": name, "session": state, "modality": "fNIRS", "feature": "error", "value": "", "quality_note": str(exc)})
    if "baseline" not in maps or "sec_test" not in maps: return None
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0))
    for r, task in enumerate(TASKS):
        a, b = maps["baseline"][0][task], maps["sec_test"][0][task]
        d = b - a; lo, hi = _color_limits([a, b, d])
        for c, (title, values) in enumerate([("baseline", a), ("sec_test", b), ("delta", d)]):
            ax = axes[r, c]; xy = maps["baseline"][1]; q = maps["baseline"][2]
            valid = np.isfinite(values)
            sc = None
            if np.any(valid):
                sc = ax.scatter(xy[valid, 0], xy[valid, 1], c=values[valid], cmap="RdBu_r", vmin=lo, vmax=hi, s=35 + 55 * np.clip(q[valid], 0, 1), alpha=.35 + .65 * np.clip(q[valid], 0, 1), edgecolor="k", lw=.2)
            ax.scatter(xy[~valid, 0], xy[~valid, 1], marker="x", color="gray", s=22)
            ax.set_aspect("equal"); ax.set_title(title if r == 0 else ""); ax.set_ylabel(task if c == 0 else ""); ax.set_xticks([]); ax.set_yticks([])
            if sc is not None:
                fig.colorbar(sc, ax=ax, fraction=.046, pad=.02, shrink=.8)
            else:
                ax.text(.5, .5, "no valid channels", transform=ax.transAxes, ha="center", va="center", fontsize=7)
    fig.suptitle(f"{subject} {name} fNIRS movement response by subtask\nChannel midpoint coordinates; automatic marker times and fixed-protocol labels; transparency/size = channel QC", fontsize=12)
    savefig(fig, out / f"fnirs_subtask_maps_{subject}")
    return maps


def group_map(pair_maps: dict[str, dict], out: Path, source_rows: list[dict], delta_rows: list[dict] | None = None):
    paired = [s for s, m in pair_maps.items() if "baseline" in m and "sec_test" in m]
    if not paired: return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, task in enumerate(TASKS):
        deltas = [pair_maps[s]["sec_test"][0][task] - pair_maps[s]["baseline"][0][task] for s in paired]
        d = np.nanmean(np.vstack(deltas), axis=0); xy = pair_maps[paired[0]]["baseline"][1]; lo, hi = _color_limits([d])
        ax = axes[i]; sc = ax.scatter(xy[:, 0], xy[:, 1], c=d, cmap="RdBu_r", vmin=lo, vmax=hi, s=60, edgecolor="k", lw=.25)
        ax.set_aspect("equal"); ax.set_title(task); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc, ax=ax, fraction=.05, pad=.03)
        for ch, value in enumerate(d, 1):
            source_rows.append({"subject": f"GROUP_PAIRED_{len(paired)}", "name": f"{len(paired)} paired subjects", "session": "sec_test-minus-baseline", "modality": "fNIRS", "feature": "group_event_delta", "task": task, "channel": f"CH{ch:02d}", "x": xy[ch-1, 0], "y": xy[ch-1, 1], "value": float(value), "n": len(paired), "quality_note": "descriptive group mean; windows are not independent"})
            if delta_rows is not None:
                delta_rows.append({"subject": f"GROUP_PAIRED_{len(paired)}", "name": f"{len(paired)} paired subjects", "session": "sec_test-minus-baseline", "task": task, "channel": f"CH{ch:02d}", "x": float(xy[ch-1,0]), "y": float(xy[ch-1,1]), "delta": float(value), "n": len(paired)})
    if delta_rows is not None:
        for subject in paired:
            for task in TASKS:
                a = pair_maps[subject]["baseline"][0][task]; b = pair_maps[subject]["sec_test"][0][task]
                for ch, value in enumerate(b-a, 1):
                    delta_rows.append({"subject": subject, "name": SUBJECT_NAMES.get(subject, subject), "session": "sec_test-minus-baseline", "task": task, "channel": f"CH{ch:02d}", "x": float(pair_maps[subject]["baseline"][1][ch-1,0]), "y": float(pair_maps[subject]["baseline"][1][ch-1,1]), "delta": float(value), "n": 1})
    fig.suptitle(f"fNIRS subtask spatial change, paired group (n={len(paired)})\nFixed combination protocol: task-specific response fingerprints")
    savefig(fig, out / "fnirs_group_subtask_delta")
    return paired


def network_map(subject: str, name: str, base: Path, follow: Path, out: Path, source_rows: list[dict]):
    if base is None or follow is None: return
    try:
        _, a = fnirs_features(base, None); _, b = fnirs_features(follow, None)
        delta = b["corr"] - a["corr"]; xy = a["positions"]
        tri = np.triu_indices_from(delta, 1)
        finite_edges = np.flatnonzero(np.isfinite(delta[tri]))
        # A small, fixed edge budget is easier to audit than a dense 15%
        # threshold graph and keeps the direction of the strongest changes
        # visible in a presentation-sized panel.
        ranked = finite_edges[np.argsort(np.abs(delta[tri])[finite_edges])[::-1]]
        selected = ranked[:12]
        limit = max(float(np.nanpercentile(np.abs(delta[tri]), 95)), 1e-9)
        fig, ax = plt.subplots(figsize=(6.3, 5.4)); ax.scatter(xy[:, 0], xy[:, 1], s=28, color="#222")
        for i, (x, y) in enumerate(xy): ax.text(x, y, str(i+1), fontsize=5)
        for edge_index in selected:
            i, j = int(tri[0][edge_index]), int(tri[1][edge_index])
            ax.plot([xy[i,0],xy[j,0]], [xy[i,1],xy[j,1]], color=plt.cm.RdBu_r((delta[i,j]/limit+1)/2), lw=1 + 3*abs(delta[i,j])/limit, alpha=.82)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{subject} {name} fNIRS resting-network correlation delta\nsec_test - baseline; strongest 12 edges")
        savefig(fig, out / f"fnirs_rest_network_delta_{subject}")
        for i, j in zip(*tri): source_rows.append({"subject": subject, "name": name, "session": "sec_test-minus-baseline", "modality": "fNIRS", "feature": "rest_network_corr_delta", "channel": f"CH{i+1}-CH{j+1}", "value": float(delta[i,j]), "quality_note": "resting fNIRS proxy correlation; strongest-edge display"})
    except Exception as exc:
        source_rows.append({"subject": subject, "name": name, "modality": "fNIRS", "feature": "network_error", "value": "", "quality_note": str(exc)})


def joint_case(subject: str, name: str, eeg_base: Path, eeg_follow: Path, fnirs_base: Path, fnirs_follow: Path, out: Path):
    """A deliberately side-by-side, not co-registered, multimodal panel."""
    if any(x is None for x in (eeg_base, eeg_follow, fnirs_base, fnirs_follow)): return
    try:
        _, eb = eeg_features(eeg_base, None); _, ef = eeg_features(eeg_follow, None)
        hb, _, _ = main.read_bdf(eeg_base, None); labels = [str(x) for x in hb.labels]
        wanted = {main.normalize_label(x) for x in main.FULL_EEG_CHANNELS}
        inds = [i for i, x in enumerate(labels) if main.normalize_label(x) in wanted]
        eeg_labels = [labels[i] for i in inds]; xy = np.asarray([main.POSITION_MAP.get(x, (np.nan, np.nan)) for x in eeg_labels])
        eeg_delta = ef["band_matrix"]["alpha"] - eb["band_matrix"]["alpha"]
        evb = read_fnirs_events(fnirs_base); evf = read_fnirs_events(fnirs_follow)
        if evb is None or evf is None: return
        _, fb = fnirs_features(fnirs_base, evb); _, ff = fnirs_features(fnirs_follow, evf)
        fd = np.nanmean(np.vstack([ff["event_delta_vectors"].mean(0), fb["event_delta_vectors"].mean(0)]), axis=0)
        fd = ff["event_delta_vectors"].mean(0) - fb["event_delta_vectors"].mean(0)
        fp = fb["positions"]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        lim = max(float(np.nanpercentile(np.abs(eeg_delta),95)),1e-9); sc=axes[0].scatter(xy[:,0],xy[:,1],c=eeg_delta,cmap="RdBu_r",vmin=-lim,vmax=lim,s=62,edgecolor="k",lw=.25)
        axes[0].set_title("EEG alpha power delta\n(sec_test - baseline; approximate 10-10)"); axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([]); fig.colorbar(sc,ax=axes[0],fraction=.05,pad=.03)
        lim=max(float(np.nanpercentile(np.abs(fd),95)),1e-9); sc=axes[1].scatter(fp[:,0],fp[:,1],c=fd,cmap="RdBu_r",vmin=-lim,vmax=lim,s=48,edgecolor="k",lw=.25)
        axes[1].set_title("fNIRS mean movement response delta\n(channel midpoint coordinates)"); axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([]); fig.colorbar(sc,ax=axes[1],fraction=.05,pad=.03)
        fig.suptitle(f"{subject} {name} EEG-fNIRS multimodal contrast\nSide-by-side exploratory view; devices are not spatially co-registered", y=0.99, fontsize=11)
        fig.subplots_adjust(top=0.76, wspace=0.28)
        savefig(fig, out / f"joint_eeg_fnirs_{subject}")
    except Exception as exc:
        print(f"[joint skipped] {subject}: {exc}")


def state_transition(subject: str, name: str, path: Path, out: Path, session: str):
    """PCA + k-means state trajectory over protocol-labeled event blocks (AI pilot)."""
    if path is None: return
    try:
        ev = read_fnirs_events(path)
        if ev is None: return
        labels = infer_movement_event_labels(ev.times, subject); _, ex = fnirs_features(path, ev)
        x = np.asarray(ex["event_delta_vectors"], float); used = np.asarray(ex["event_indices_used"], int)
        good = np.all(np.isfinite(x), axis=1)
        if good.sum() < 5: return
        z = PCA(n_components=2, random_state=0).fit_transform(x[good]); k=min(3,len(z)); states=KMeans(n_clusters=k,n_init=20,random_state=0).fit_predict(z)
        transition=np.zeros((k,k),int)
        for a,b in zip(states[:-1],states[1:]): transition[a,b]+=1
        fig, axes=plt.subplots(1,2,figsize=(8.4,3.5),gridspec_kw={"width_ratios":[1.25,1]})
        for task,color in COLORS.items():
            idx=[j for j,u in enumerate(used) if u<len(labels) and labels[u]["task_name"]==task and good[j]]
            if idx: axes[0].scatter(z[idx,0],z[idx,1],s=30,c=color,label=task)
        axes[0].legend(frameon=False,fontsize=8); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].set_title("PCA event-response states")
        im=axes[1].imshow(transition,cmap="Blues",vmin=0); axes[1].set_xlabel("next state"); axes[1].set_ylabel("current state"); axes[1].set_title(f"Transition counts (events={len(states)})")
        for i in range(k):
            for j in range(k): axes[1].text(j,i,str(transition[i,j]),ha="center",va="center")
        fig.suptitle(f"{subject} {name} fNIRS unsupervised state pilot ({session})\nPCA + k-means; exploratory AI aid, event blocks are not independent", y=0.99, fontsize=11)
        fig.subplots_adjust(top=0.78, wspace=0.3)
        savefig(fig, out / f"fnirs_state_transition_{subject}_{session}")
    except Exception as exc:
        print(f"[state skipped] {subject}: {exc}")


def joint_rest_map(subject: str, name: str, base_eeg: Path, follow_eeg: Path, base_fnirs: Path, follow_fnirs: Path, out: Path, source_rows: list[dict]):
    """Parallel EEG/fNIRS resting-state maps; deliberately no cross-device registration."""
    if any(path is None for path in (base_eeg, follow_eeg, base_fnirs, follow_fnirs)):
        return
    try:
        _, eeg_a = eeg_features(base_eeg, None); _, eeg_b = eeg_features(follow_eeg, None)
        header, _, _ = main.read_bdf(base_eeg, None)
        labels = [str(x) for x in header.labels]
        wanted = {main.normalize_label(x) for x in main.FULL_EEG_CHANNELS}
        scalp = [i for i, x in enumerate(labels) if main.normalize_label(x) in wanted and labels[i] in main.POSITION_MAP]
        eeg_delta = np.asarray(eeg_b["band_matrix"]["alpha"]) - np.asarray(eeg_a["band_matrix"]["alpha"])
        eeg_xy = np.asarray([main.POSITION_MAP[labels[i]] for i in scalp])
        eeg_labels = [labels[i] for i in scalp]
        eeg_delta = eeg_delta[:len(eeg_labels)]

        _, fn_a = fnirs_features(base_fnirs, None); _, fn_b = fnirs_features(follow_fnirs, None)
        fn_xy = np.asarray(fn_a["positions"]); corr_delta = np.asarray(fn_b["corr"]) - np.asarray(fn_a["corr"])
        fn_node = np.nansum(np.abs(corr_delta), axis=1)
        fn_lim = max(float(np.nanpercentile(np.abs(fn_node), 95)), 1e-9)
        eeg_lim = max(float(np.nanpercentile(np.abs(eeg_delta), 95)), 1e-9)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": [1, 1.12]})
        ax = axes[0]; _head(ax)
        sc = ax.scatter(eeg_xy[:, 0], eeg_xy[:, 1], c=eeg_delta, cmap="RdBu_r", vmin=-eeg_lim, vmax=eeg_lim, s=55, edgecolor="k", lw=.25)
        for (x, y), label in zip(eeg_xy, eeg_labels):
            if label in KEY_EEG_LABELS: ax.text(x, y + .055, label, fontsize=6, ha="center")
        ax.set_title("EEG alpha power delta\n(approx. 10-10 scalp positions)", fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=.046, pad=.03, label="sec_test - baseline")
        ax = axes[1]
        sc = ax.scatter(fn_xy[:, 0], fn_xy[:, 1], c=fn_node, cmap="magma", vmin=0, vmax=fn_lim, s=60, edgecolor="k", lw=.25)
        for i, (x, y) in enumerate(fn_xy, 1): ax.text(x, y, str(i), fontsize=5, ha="center", va="center", color="white" if fn_node[i-1] > fn_lim*.6 else "black")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("fNIRS resting-network node change\n(source-detector midpoints)", fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=.046, pad=.03, label="sum |correlation delta|")
        fig.suptitle(f"{subject} {name} multimodal resting-state view\nParallel maps only; EEG-fNIRS spatial registration is not available", y=0.99, fontsize=11)
        fig.subplots_adjust(top=0.76, wspace=0.3)
        savefig(fig, out / f"joint_rest_multimodal_{subject}")
        for label, value in zip(eeg_labels, eeg_delta):
            x, y = main.POSITION_MAP[label]
            source_rows.append({"subject": subject, "name": name, "session": "sec_test-minus-baseline", "modality": "EEG-fNIRS", "feature": "EEG_alpha_delta_joint_panel", "channel": label, "x": x, "y": y, "value": float(value), "quality_note": "parallel exploratory panel; EEG approximate 10-10 coordinates"})
        for idx, value in enumerate(fn_node, 1):
            source_rows.append({"subject": subject, "name": name, "session": "sec_test-minus-baseline", "modality": "EEG-fNIRS", "feature": "fNIRS_rest_node_change_joint_panel", "channel": f"CH{idx:02d}", "x": fn_xy[idx-1, 0], "y": fn_xy[idx-1, 1], "value": float(value), "quality_note": "parallel exploratory panel; fNIRS source-detector midpoint coordinates"})
    except Exception as exc:
        source_rows.append({"subject": subject, "name": name, "modality": "EEG-fNIRS", "feature": "joint_error", "value": "", "quality_note": str(exc)})


def bridge_plot(subject: str, name: str, session: str, fnirs_path: Path, emg_path: Path, out: Path):
    if fnirs_path is None or emg_path is None: return
    try:
        ev = read_fnirs_events(fnirs_path)
        if ev is None: return
        labels = infer_movement_event_labels(ev.times, subject)
        _, fx = fnirs_features(fnirs_path, ev)
        _, ex = emg_features(emg_path)
        if not bool(np.asarray(ex.get("mapping_valid_for_primary_features", False))):
            return
        envelope = np.asarray(ex.get("envelope", []), dtype=float)
        if envelope.ndim != 2 or envelope.shape[0] == 0:
            return
        env = np.nanmean(envelope, axis=0); et = np.asarray(ex["time"], float)
        used = np.asarray(fx["event_indices_used"], int); fv = np.asarray(fx["event_delta_vectors"], float)
        rows = []
        for task in TASKS:
            fvals=[]; evals=[]
            for j, idx in enumerate(used):
                if idx >= len(labels) or labels[idx]["task_name"] != task: continue
                onset = ev.times[idx]; emask=(et >= onset+2) & (et <= onset+12)
                if np.any(emask): fvals.append(float(np.nanmean(fv[j]))); evals.append(float(np.nanmean(env[emask])))
            if fvals:
                rows.append((task, fvals, evals))
        if not rows: return
        fig, axes = plt.subplots(1, len(rows), figsize=(4.3*len(rows), 3.6), squeeze=False)
        for ax, (task, x, y) in zip(axes[0], rows):
            ax.scatter(x, y, color=COLORS[task], s=30); ax.set_title(task); ax.set_xlabel("fNIRS mean event response"); ax.set_ylabel("EMG envelope")
            if len(x) >= 2: ax.plot(np.polyfit(x, y, 1)[0]*np.asarray(x)+np.polyfit(x, y, 1)[1], color="#555", lw=.8)
        fig.suptitle(f"{subject} {name} {session} descriptive brain-muscle bridge\nSame protocol-labeled action blocks; no causal interpretation")
        savefig(fig, out / f"bridge_{subject}_{session}")
    except Exception:
        return


def write_csv(path: Path, rows: list[dict]):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main_run(data_root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True); set_data_root(data_root)
    rows=[]; pair_maps={}; paired=[]; delta_rows=[]
    # This refresh is a frozen paired-seven analysis; scanning unrelated
    # folders delays output creation and can leak non-cohort cases downstream.
    subjects = [subject for subject in PAIRED_SUBJECTS if (data_root / subject).is_dir()]
    for subject in subjects:
        name = SUBJECT_NAMES.get(subject, subject)
        state_paths={s: recording(data_root, subject, s, "FNIRS", "运动") for s in ("base_line", "sec_test")}
        maps = event_task_maps(subject, name, state_paths, out, rows)
        if maps is not None: pair_maps[subject] = maps
        network_map(subject, name, recording(data_root, subject, "base_line", "FNIRS", "静息态"), recording(data_root, subject, "sec_test", "FNIRS", "静息态"), out, rows)
        # The paired-seven refresh consumes the canonical state table in
        # Phase 8.  Avoid per-case KMeans plots here: on Windows/MKL they
        # create a large transient memory footprint without contributing to
        # the channel-level spatial source tables used by later phases.
    paired = group_map(pair_maps, out, rows, delta_rows) or []
    write_csv(out / "figure_spatial_source_data.csv", rows)
    write_csv(out / "fnirs_pair_channel_delta.csv", delta_rows)
    meta = {"canonical_rule": "outer subject folder ID; baseline + sec_test; sec excluded as duplicate", "subjects_scanned": len(subjects), "paired_subjects": paired, "highlight_subjects": HIGHLIGHT, "n_source_rows": len(rows)}
    (out / "figure_spatial_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    review = f"""# Phase 5 spatial visualisation review\n\n- Canonical sessions: `base_line` and `sec_test`; the duplicate `sec` session is excluded.\n- Subjects scanned: {len(subjects)}; paired fNIRS subjects with both sessions: {len(paired)} ({', '.join(paired)}).\n- EEG: theta/alpha/beta maps are exploratory only. Signals are low quality in most recordings and locations are approximate standard 10-10 coordinates, not individual digitisation.\n- fNIRS: values are wavelength-balanced optical-density proxies, not absolute HbO/HbR. Channel points are source-detector midpoints; no cross-device EEG-fNIRS registration is implied.\n- Movement marker times are automatic and treated as accurate; `摸头`/`摸膝`/`摸腰` labels follow the confirmed fixed protocol order. Blocks/windows are not independent samples.\n- Network maps: resting fNIRS correlation differences, strongest-edge display, descriptive only.\n- Task-fingerprint RSA: cosine similarity and task-separation index summarize the three protocol-labeled action vectors; this is an exploratory representational analysis, not a trained decoder.\n- State pilot: PCA + k-means event-response trajectories and transition counts are exploratory AI aids; state labels are session-specific and not clinical or TCM classes.\n- Brain-muscle bridge: same-block fNIRS response and EMG envelope are aligned descriptively; no causal or inferential claim.\n- All participants received the same fixed combination acupuncture/electroacupuncture protocol; baseline -> sec_test is approximately two weeks and represents longitudinal change, not an immediate effect.\n- Source traceability: all plotted channel-level values are in `figure_spatial_source_data.csv`; QC score is included where available.\n"""
    (out / "figure_spatial_review.md").write_text(review, encoding="utf-8")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root", type=Path, default=RAW_DATA_ROOT); ap.add_argument("--out", type=Path, default=phase_root(5))
    args=ap.parse_args(); main_run(args.data_root.resolve(), args.out.resolve())
