"""Phase 9: coverage-aware quantitative eye/TCM pilot anchors.

This stage is descriptive only. Eye report scores, diagnostic labels, and
missing values are excluded from process features. Observed eye-process pairs
are joined only within the same reported task; no missing task/session values
are imputed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import PAIRED_SUBJECTS, phase_root
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import PAIRED_SUBJECTS, phase_root


ROOT = Path(__file__).resolve().parents[1]
TCM_ROOT = phase_root("tcm")
EYE = TCM_ROOT / "eye_report_pilot_features.csv"
FOUR = TCM_ROOT / "four_diagnosis_pilot_features.csv"
TIDY_FALLBACKS = [
    ROOT / "outputs" / "release_20260903" / "data" / "tcm_eye_raw_quantitative_tidy.csv",
    ROOT / "outputs" / "phase9_20260902" / "tcm_eye_raw_quantitative_tidy.csv",
]
DELTA = phase_root(6) / "crossmodal_delta_matrix.csv"
OUT = phase_root(9)
PAIR = list(PAIRED_SUBJECTS)
EXPECTED_EYE_PROCESS_PAIRS = {"2026081401", "2026081502", "2026081701"}
TASK_ALIASES = {
    "执行功能与注意力": "executive/attention",
    "记忆力与注意力": "memory/attention",
}
EYE_PROCESS_METRICS = [
    "fixation_saccades", "pursuit_saccades", "prosaccade_latency_ms",
    "antisaccade_latency_ms", "antisaccade_error_percent",
    "target_fixation_time_ms", "target_fixation_count",
    "distractor_fixation_time_ms", "distractor_fixation_count",
]
TCM_METRICS = [
    "coating_thin_score", "coating_greasy_score", "prickles_count",
    "crack_score", "tooth_marks_count", "left_pulse_rate_per_min",
    "right_pulse_rate_per_min",
]


def read_report_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    if EYE.exists() and FOUR.exists():
        eye = pd.read_csv(EYE)
        four = pd.read_csv(FOUR)
    else:
        tidy_path = next((path for path in TIDY_FALLBACKS if path.exists()), None)
        if tidy_path is None:
            raise FileNotFoundError(f"Missing eye/TCM features and tidy fallback: {TIDY_FALLBACKS}")
        tidy = pd.read_csv(tidy_path)
        tidy["subject"] = tidy["subject"].astype(str)
        eye = tidy[tidy.domain.eq("eye_process")].pivot_table(
            index=["subject", "session"], columns="metric", values="value", aggfunc="mean"
        ).reset_index()
        four = tidy[tidy.domain.eq("tcm_raw")].pivot_table(
            index=["subject", "session"], columns="metric", values="value", aggfunc="mean"
        ).reset_index()
    for frame in (eye, four):
        frame["subject"] = frame["subject"].astype(str)
    return eye, four


def metric_unit(metric: str) -> str:
    if metric.endswith("_ms") or "time" in metric:
        return "ms"
    if "percent" in metric:
        return "percent"
    if "rate_per_min" in metric:
        return "per_min"
    if "score" in metric:
        return "score"
    return "count"


def task_text(values: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in values.dropna() if str(value).strip()}))


def source_text(values: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in values.dropna() if str(value).strip()}))


def build_eye_pair_delta(eye: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [metric for metric in EYE_PROCESS_METRICS if metric in eye.columns]
    observed = eye[eye.subject.isin(PAIR) & eye.session.isin(["base_line", "sec_test"])].copy()
    for metric in metrics:
        observed[metric] = pd.to_numeric(observed[metric], errors="coerce")

    # A single report/task row is expected. Aggregation handles duplicate source
    # rows without inventing a session or task match.
    task_rows = []
    for (subject, session, task), group in observed.groupby(["subject", "session", "task"], dropna=False):
        row: dict[str, object] = {
            "subject": subject,
            "session": session,
            "task": task,
            "source_reports": source_text(group["report"]) if "report" in group else "",
        }
        for metric in metrics:
            values = group[metric].dropna()
            row[metric] = float(values.median()) if not values.empty else np.nan
        task_rows.append(row)
    task_level = pd.DataFrame(task_rows)

    pair_rows = []
    for (subject, task), group in task_level.groupby(["subject", "task"], dropna=False):
        base = group[group.session.eq("base_line")]
        follow = group[group.session.eq("sec_test")]
        if base.empty or follow.empty:
            continue
        for metric in metrics:
            base_value = pd.to_numeric(base.iloc[0][metric], errors="coerce")
            follow_value = pd.to_numeric(follow.iloc[0][metric], errors="coerce")
            if not np.isfinite(base_value) or not np.isfinite(follow_value):
                continue
            pair_rows.append({
                "subject": subject, "task": task, "metric": metric,
                "unit": metric_unit(metric), "baseline_value": float(base_value),
                "followup_value": float(follow_value),
                "delta_followup_minus_baseline": float(follow_value - base_value),
                "event_alignment": "exact_task_label",
                "baseline_source_reports": base.iloc[0]["source_reports"],
                "followup_source_reports": follow.iloc[0]["source_reports"],
                "no_imputation": True,
            })
    paired = pd.DataFrame(pair_rows)
    if paired.empty:
        paired = pd.DataFrame(columns=[
            "subject", "task", "metric", "unit", "baseline_value", "followup_value",
            "delta_followup_minus_baseline", "event_alignment", "baseline_source_reports",
            "followup_source_reports", "no_imputation",
        ])

    coverage_rows = []
    for subject in PAIR:
        subject_rows = task_level[task_level.subject.eq(subject)]
        base = subject_rows[subject_rows.session.eq("base_line")]
        follow = subject_rows[subject_rows.session.eq("sec_test")]
        subject_pairs = paired[paired.subject.eq(subject)]
        observed_pair = not subject_pairs.empty
        coverage_rows.append({
            "subject": subject,
            "baseline_eye_available": not base.empty,
            "followup_eye_available": not follow.empty,
            "paired_eye_process_available": observed_pair,
            "expected_usable_paired_subject": subject in EXPECTED_EYE_PROCESS_PAIRS,
            "observed_matches_expected": observed_pair == (subject in EXPECTED_EYE_PROCESS_PAIRS),
            "baseline_task_coverage": task_text(base["task"]),
            "followup_task_coverage": task_text(follow["task"]),
            "paired_task_coverage": task_text(subject_pairs["task"]),
            "baseline_source_reports": source_text(base["source_reports"]),
            "followup_source_reports": source_text(follow["source_reports"]),
            "paired_metric_task_rows": len(subject_pairs),
            "coverage_note": "Task coverage is uneven; deltas use only exact task-label matches and observed values.",
        })
    return paired, pd.DataFrame(coverage_rows)


def save_tidy(eye: pd.DataFrame, four: pd.DataFrame, delta: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    eye_metrics = [metric for metric in EYE_PROCESS_METRICS if metric in eye.columns]
    four_metrics = [metric for metric in TCM_METRICS if metric in four.columns]
    for _, row in eye.iterrows():
        for metric in eye_metrics:
            value = pd.to_numeric(row.get(metric), errors="coerce")
            if np.isfinite(value):
                rows.append({"subject": row["subject"], "session": row["session"], "domain": "eye_process", "metric": metric, "value": float(value), "unit": metric_unit(metric)})
    for _, row in four.iterrows():
        for metric in four_metrics:
            value = pd.to_numeric(row.get(metric), errors="coerce")
            if np.isfinite(value):
                rows.append({"subject": row["subject"], "session": row["session"], "domain": "tcm_raw", "metric": metric, "value": float(value), "unit": metric_unit(metric)})
    pd.DataFrame(rows).to_csv(OUT / "tcm_eye_raw_quantitative_tidy.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUT / "paired_neuro_motor_delta_for_anchor.csv", index=False, encoding="utf-8-sig")


def plot_coverage(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    cols = ["baseline_eye_available", "followup_eye_available", "paired_eye_process_available"]
    labels = ["baseline\neye", "follow-up\neye", "observed\nprocess pair"]
    matrix = coverage[cols].astype(int).to_numpy()
    image = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(coverage)), [subject[-4:] for subject in coverage.subject])
    ax.set_xticks(range(len(cols)), labels)
    for i in range(len(coverage)):
        for j in range(len(cols)):
            ax.text(j, i, "available" if matrix[i, j] else "missing", ha="center", va="center", fontsize=7)
    ax.set_title("Eye-process report coverage (paired seven)")
    ax.set_xlabel("Observed availability; no missing values are imputed")
    plt.colorbar(image, ax=ax, shrink=.8, label="1 = available")


def plot_eye_pairs(ax: plt.Axes, paired: pd.DataFrame) -> None:
    executive = paired[paired.task.eq("执行功能与注意力")].copy()
    metrics = [metric for metric in EYE_PROCESS_METRICS if metric in set(executive.metric)]
    if executive.empty or not metrics:
        ax.text(.5, .5, "No exact-task paired eye-process values", ha="center", va="center")
        ax.set_axis_off()
        return
    for i, metric in enumerate(metrics):
        subset = executive[executive.metric.eq(metric)].reset_index(drop=True)
        values = pd.concat([subset.baseline_value, subset.followup_value]).to_numpy(float)
        lo, hi = np.nanmin(values), np.nanmax(values)
        scaled = (values - lo) / (hi - lo) if hi > lo else np.full_like(values, .5)
        n = len(subset)
        for j in range(n):
            ax.plot([i - .16, i + .16], scaled[[j, j + n]], color="#4c78a8", alpha=.65, lw=1.2)
            ax.scatter([i - .16, i + .16], scaled[[j, j + n]], s=22, color=["#4c78a8", "#e76f51"], zorder=3)
    ax.set_xticks(range(len(metrics)), [metric.replace("_", "\n") for metric in metrics], fontsize=7)
    ax.set_yticks([0, .5, 1], ["within-metric\nlow", "display\nmid", "within-metric\nhigh"])
    ax.set_title("Observed executive/attention eye-process pairs (n=3)")
    ax.text(.01, -.26, "Blue = baseline; orange = follow-up. Scaling is display-only; raw values and deltas are in CSV.", transform=ax.transAxes, fontsize=8)


def plot_delta_heatmap(ax: plt.Axes, paired: pd.DataFrame) -> None:
    if paired.empty:
        ax.text(.5, .5, "No paired eye-process deltas", ha="center", va="center")
        ax.set_axis_off()
        return
    matrix = paired.pivot_table(index=["subject", "task"], columns="metric", values="delta_followup_minus_baseline", aggfunc="mean")
    scaled = matrix.copy()
    for metric in scaled.columns:
        max_abs = scaled[metric].abs().max()
        scaled[metric] = scaled[metric] / max_abs if np.isfinite(max_abs) and max_abs > 0 else 0.0
    image = ax.imshow(scaled.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    row_labels = [f"{subject[-4:]} | {TASK_ALIASES.get(task, 'other task')}" for subject, task in matrix.index]
    ax.set_yticks(range(len(matrix)), row_labels, fontsize=7)
    ax.set_xticks(range(len(matrix.columns)), [metric.replace("_", "\n") for metric in matrix.columns], fontsize=7)
    ax.set_title("Task-aligned eye-process deltas")
    ax.set_xlabel("Colour is metric-normalized delta; raw delta values are in CSV")
    plt.colorbar(image, ax=ax, shrink=.8, label="within-metric signed display scale")


def plot_tcm(ax: plt.Axes, four: pd.DataFrame) -> None:
    metrics = [metric for metric in TCM_METRICS if metric in four.columns]
    baseline = four[four.subject.isin(PAIR) & four.session.eq("base_line")].set_index("subject").reindex(PAIR)
    if baseline.empty or not metrics:
        ax.text(.5, .5, "No baseline TCM quantitative values", ha="center", va="center")
        ax.set_axis_off()
        return
    values = baseline[metrics].apply(pd.to_numeric, errors="coerce")
    display = values.copy()
    for metric in display:
        lo, hi = display[metric].min(), display[metric].max()
        display[metric] = (display[metric] - lo) / (hi - lo) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else .5
    image = ax.imshow(display.to_numpy(float), aspect="auto", cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_yticks(range(len(baseline)), [subject[-4:] for subject in baseline.index])
    ax.set_xticks(range(len(metrics)), [metric.replace("_", "\n") for metric in metrics], fontsize=7)
    ax.set_title("Raw four-diagnosis quantitative anchors (baseline)")
    ax.set_xlabel("Display-scaled only; no syndrome or efficacy inference")
    plt.colorbar(image, ax=ax, shrink=.8, label="within-metric display scale")


def make_figure(eye_pairs: pd.DataFrame, coverage: pd.DataFrame, four: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plot_coverage(axes[0, 0], coverage)
    plot_eye_pairs(axes[0, 1], eye_pairs)
    plot_delta_heatmap(axes[1, 0], eye_pairs)
    plot_tcm(axes[1, 1], four)
    fig.suptitle("Quantitative eye/TCM anchors | exploratory coverage-aware pilot", fontsize=14)
    for suffix, kwargs in [("png", {"dpi": 600}), ("svg", {}), ("pdf", {})]:
        fig.savefig(OUT / f"tcm_eye_anchor.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_review(eye_pairs: pd.DataFrame) -> None:
    paired_subjects = sorted(eye_pairs.subject.astype(str).unique())
    by_metric = eye_pairs.groupby(["task", "metric"]).subject.nunique().reset_index(name="n_paired_subjects")
    metric_lines = "\n".join(f"- `{row.task}` / `{row.metric}`: n={row.n_paired_subjects}" for row in by_metric.itertuples(index=False)) or "- No observed exact-task process pairs."
    expected = ", ".join(sorted(EXPECTED_EYE_PROCESS_PAIRS))
    observed = ", ".join(paired_subjects) or "none"
    review = f"""# Quantitative eye/TCM anchor review

This is an exploratory TCM/cognitive anchor only. It contains no p-values,
efficacy estimate, treatment claim, report-score analysis, diagnostic label, or
syndrome inference.

The expected usable paired eye-process participants are `{expected}`. The
observed exact-task pairs are `{observed}`. `paired_eye_process_coverage.csv`
records every paired-seven participant, source-report coverage, and the exact
task labels used for alignment. Task coverage is uneven: only values observed
at both sessions with the same task label enter `paired_eye_process_delta.csv`;
missing values are not imputed.

Paired subject-by-metric-task rows: **{len(eye_pairs)}** across
**{len(paired_subjects)}** participants.

## Effective paired n by task and metric

{metric_lines}

Raw process values remain in `tcm_eye_raw_quantitative_tidy.csv`. The figure
uses metric-specific display scaling only; it must not be read as a common
effect-size scale. Four-diagnosis values are raw quantitative anchors pending
source-document verification and are not linked to treatment response.
"""
    (OUT / "review.md").write_text(review, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eye, four = read_report_features()
    delta = pd.read_csv(DELTA)
    delta["subject"] = delta["subject"].astype(str)
    delta = delta[delta.subject.isin(PAIR)].copy()
    eye_pairs, coverage = build_eye_pair_delta(eye)
    save_tidy(eye, four, delta)
    eye_pairs.to_csv(OUT / "paired_eye_process_delta.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT / "paired_eye_process_coverage.csv", index=False, encoding="utf-8-sig")
    make_figure(eye_pairs, coverage, four)
    write_review(eye_pairs)
    observed = sorted(eye_pairs.subject.astype(str).unique())
    print(f"eye_pair_rows={len(eye_pairs)} eye_pair_subjects={observed} output={OUT.resolve()}")


if __name__ == "__main__":
    main()
