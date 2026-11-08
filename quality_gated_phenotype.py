"""Interpretable quality-gated multimodal digital-phenotype prototype."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


GROUP_PREFIXES = {
    "fnirs": ("fnirs_", "rest_fnirs_"),
    "emg": ("emg_",),
    "eeg": ("eeg_", "eye_eeg_"),
    "emg_bridge": ("bridge_",),
}
EXCLUDED_FEATURE_TOKENS = {"report_score_archive_only", "diagnosis", "syndrome"}
EXCLUDED_NUMERIC_SUFFIXES = (
    "_count_used",
    "_pairs",
    "_p_nominal",
    "_status",
    "_error",
    "_missing_fraction",
    "_quality",
    "_excluded_sensor_count",
    "_sensor_count",
)


def _is_signal_feature(field: str) -> bool:
    lowered = field.lower()
    return not any(lowered.endswith(suffix) or suffix in lowered for suffix in EXCLUDED_NUMERIC_SUFFIXES)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonical_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = {(row.get("subject", ""), row.get("session", "")) for row in rows if row.get("session") == "sec_test"}
    return [row for row in rows if not (row.get("session") == "sec" and (row.get("subject", ""), "sec_test") in keys)]


def feature_groups(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    fields = set().union(*(row.keys() for row in rows)) if rows else set()
    groups: dict[str, list[str]] = {}
    for group, prefixes in GROUP_PREFIXES.items():
        groups[group] = sorted(
            field for field in fields
            if field not in {"subject", "name", "session"}
            and not any(token in field.lower() for token in EXCLUDED_FEATURE_TOKENS)
            and _is_signal_feature(field)
            and field.startswith(prefixes)
            and any(_number(row.get(field)) is not None for row in rows)
        )
    return groups


def robust_reference(rows: list[dict[str, str]], fields: list[str]) -> dict[str, tuple[float, float]]:
    baseline = [row for row in rows if row.get("session") == "base_line"]
    reference: dict[str, tuple[float, float]] = {}
    for field in fields:
        values = sorted(value for row in baseline if (value := _number(row.get(field))) is not None)
        if not values:
            continue
        center = values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
        deviations = sorted(abs(value - center) for value in values)
        mad = deviations[len(deviations) // 2] if len(deviations) % 2 else (deviations[len(deviations) // 2 - 1] + deviations[len(deviations) // 2]) / 2
        reference[field] = (center, max(1.4826 * mad, 1e-9))
    return reference


def quality_weights(availability: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in availability:
        key = (row.get("subject", ""), row.get("session", ""))
        modality = (row.get("modality", "") or "").lower()
        if modality not in {"fnirs", "emg", "eeg"} or row.get("available") not in {"1", "True", "true"}:
            continue
        score = _number(row.get("quality_score"))
        if score is None:
            continue
        grouped.setdefault(key, {}).setdefault(modality, []).append(max(0.0, min(1.0, score / 100.0)))
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, modalities in grouped.items():
        result[key] = {modality: sum(values) / len(values) for modality, values in modalities.items()}
    return result


def build_outputs(motor: list[dict[str, str]], availability: list[dict[str, str]], tcm: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    motor = canonical_rows(motor)
    groups = feature_groups(motor)
    tcm_by_key = {(row.get("subject", ""), row.get("session", "")): row for row in canonical_rows(tcm)}
    eye_meta_excluded = {"eye_process_metric_rows", "eye_total_score_model_feature", "eye_report_present"}
    eye_fields = sorted({
        field for row in tcm for field in row
        if field.startswith("eye_")
        and not field.startswith("eye_eeg_")
        and field not in eye_meta_excluded
        and "report_score" not in field
        and _number(row.get(field)) is not None
    })
    raw_tokens = ("coating_", "prickles_", "crack_", "tooth_marks_", "pulse_rate")
    four_fields = sorted({field for row in tcm for field in row if field.startswith("four_") and any(token in field for token in raw_tokens) and _number(row.get(field)) is not None})
    groups["eye_process"] = eye_fields
    groups["tcm_raw"] = four_fields
    enriched = []
    for row in motor:
        merged = dict(row)
        merged.update(tcm_by_key.get((row.get("subject", ""), row.get("session", "")), {}))
        enriched.append(merged)
    all_fields = sorted(set().union(*(fields for fields in groups.values())))
    reference = robust_reference(enriched, all_fields)
    weights = quality_weights(availability)
    rows: list[dict[str, object]] = []
    for source in enriched:
        key = (source.get("subject", ""), source.get("session", ""))
        output: dict[str, object] = {"subject": key[0], "name": source.get("name", ""), "session": key[1]}
        joined_report = tcm_by_key.get(key, {})
        eye_present = bool(_number(joined_report.get("eye_process_metric_rows")) or any(_number(joined_report.get(field)) is not None for field in eye_fields))
        four_present = bool(_number(joined_report.get("four_quantitative_rows")) or any(_number(joined_report.get(field)) is not None for field in four_fields))
        output["eye_report_present"] = int(eye_present)
        output["four_report_present"] = int(four_present)
        output["report_row_present"] = int(eye_present or four_present)
        modal_weights = weights.get(key, {})
        for modality in ("fnirs", "emg", "eeg", "emg_bridge", "eye_process", "tcm_raw"):
            weight = (1.0 if modality in {"eye_process", "tcm_raw"} else modal_weights.get("emg" if modality == "emg_bridge" else modality, 0.0))
            values: list[float] = []
            clipped_count = 0
            for field in groups[modality]:
                number = _number(source.get(field))
                if number is None or field not in reference:
                    continue
                center, scale = reference[field]
                z = (number - center) / scale
                clipped = max(-5.0, min(5.0, z))
                clipped_count += int(clipped != z)
                values.append(clipped)
            output[f"{modality}_quality_weight"] = weight
            output[f"{modality}_feature_count"] = len(values)
            output[f"{modality}_missing_fraction"] = 1.0 - len(values) / max(len(groups[modality]), 1)
            output[f"{modality}_clipped_feature_count"] = clipped_count
            output[f"{modality}_contribution"] = (weight * sum(values) / len(values)) if values else ""
        output["report_quality_unknown"] = 1
        rows.append(output)
    by_key = {(row["subject"], row["session"]): row for row in rows}
    deltas: list[dict[str, object]] = []
    for subject in sorted({row["subject"] for row in rows}):
        baseline = by_key.get((subject, "base_line"))
        follow = by_key.get((subject, "sec_test"))
        if not baseline or not follow:
            continue
        delta: dict[str, object] = {"subject": subject, "baseline_session": "base_line", "followup_session": "sec_test"}
        for group in (*GROUP_PREFIXES, "eye_process", "tcm_raw"):
            before, after = baseline[f"{group}_contribution"], follow[f"{group}_contribution"]
            delta[f"{group}_delta"] = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else ""
        deltas.append(delta)
    return rows, deltas


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["subject", "session"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_heatmap(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    groups = [*GROUP_PREFIXES, "eye_process", "tcm_raw"]
    matrix = np.asarray([[float(row[f"{group}_contribution"]) if isinstance(row[f"{group}_contribution"], (int, float)) else np.nan for group in groups] for row in rows])
    labels = [f"{row['subject']}/{row['session']}" for row in rows]
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.28 * len(rows) + 1.8)), dpi=150)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(len(groups)), groups, rotation=25, ha="right")
    ax.set_title("Quality-gated phenotype contributions (descriptive)")
    fig.colorbar(image, ax=ax, label="weighted robust-z contribution")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(analysis_root: Path, tcm_root: Path, output_root: Path) -> None:
    motor = read_csv(analysis_root / "motor_multimodal_features.csv")
    availability = read_csv(analysis_root / "expanded_availability_inventory.csv")
    tcm = read_csv(tcm_root / "tcm_multimodal_pilot.csv") if (tcm_root / "tcm_multimodal_pilot.csv").exists() else []
    rows, deltas = build_outputs(motor, availability, tcm)
    write_csv(output_root / "quality_gated_phenotype.csv", rows)
    write_csv(output_root / "quality_gated_delta.csv", deltas)
    write_heatmap(output_root / "quality_gated_phenotype_heatmap.png", rows)
    (output_root / "method.md").write_text(
        "# Quality-gated digital phenotype prototype\n\n"
        "Rows are canonical subject/session units; `sec` is excluded when `sec_test` exists. QC scores are clipped to [0,1] weights per signal modality. Numeric feature groups use baseline median and MAD robust z-scores clipped to [-5,5], with explicit feature counts, clipped counts and missing fractions. fNIRS response/asymmetry, EMG bridge, eye process metrics and raw TCM quantitative fields are retained as separate descriptive groups; report groups use weight 1.0 with unknown report quality.\n\n"
        "Eye-report total scores, TCM syndrome labels and marker value 1 as a task code are not used. Report presence flags reflect actual quantitative process/raw fields (not merely a join row); missing reports/signals remain missing and lower contribution rather than being imputed. This is an interpretable prototype, not a treatment-response prediction model.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/analysis/20260901_all_cases"))
    parser.add_argument("--tcm-root", type=Path, default=Path("outputs/experiments/tcm_20260901"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase4_20260901"))
    args = parser.parse_args()
    run(args.analysis_root, args.tcm_root, args.output_root)


if __name__ == "__main__":
    main()
