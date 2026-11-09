"""Inventory and pilot integration of eye-movement and four-diagnosis reports.

The reports are clinical documents, not synchronized waveforms.  This script
therefore keeps three states explicit: file inventory, machine-readable text
extraction, and manually transcribed pilot features that still require a
second reviewer.  It never infers a TCM syndrome from a single tongue or pulse
finding.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

import expand_all_cases as expanded


REPORT_SCORE_FIELDS = {"report_score_archive_only"}
JOIN_ID_FIELDS = {"subject", "name", "session", "report", "task", "extraction_status", "confidence", "metric_use"}
# The two PDFs were visually reviewed as the same execution report. Their
# binary hashes differ because PDF export metadata differs, so this explicit
# reviewer decision is retained instead of byte-level deduplication.
EYE_DUPLICATES = {
    ("2026081502", "sec_test"): {"bgh2.pdf": "bgh1.pdf"},
}
MANUAL_EYE_REPORTS = {
    ("2026081201", "base_line", "1.pdf"),
    ("2026081201", "base_line", "2.pdf"),
    ("2026081401", "base_line", "zh1.pdf"),
    ("2026081401", "sec_test", "zh1.pdf"),
    ("2026081401", "sec_test", "zh2.pdf"),
    ("2026081403", "base_line", "cgz1.pdf"),
    ("2026081403", "base_line", "cgz2.pdf"),
    ("2026081501", "base_line", "xwz1.pdf"),
    ("2026081502", "base_line", "bgh1.pdf"),
    ("2026081502", "sec_test", "bgh1.pdf"),
    ("2026081701", "base_line", "djs1.pdf"),
    ("2026081701", "base_line", "djs2.pdf"),
    ("2026081701", "sec_test", "djs1.pdf"),
    ("2026081701", "sec_test", "djs2.pdf"),
}


def _numeric_fields(rows: list[dict[str, str]], excluded: set[str]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in excluded or key in fields or not value:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                fields.append(key)
    return fields


def _aggregate_numeric(rows: list[dict[str, str]], fields: list[str]) -> dict[str, object]:
    output: dict[str, object] = {}
    for field in fields:
        values: list[float] = []
        for row in rows:
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            output[field] = sum(values) / len(values)
    return output


def build_pilot_join(analysis_root: Path, output: Path) -> Path | None:
    """Join motor features with report pilot numeric metrics by subject/session."""
    motor_path = analysis_root / "motor_multimodal_features.csv"
    eye_path = output / "eye_report_pilot_features.csv"
    four_path = output / "four_diagnosis_pilot_features.csv"
    if not motor_path.exists() or not eye_path.exists() or not four_path.exists():
        return None
    with motor_path.open("r", encoding="utf-8-sig", newline="") as handle:
        motor_rows = list(csv.DictReader(handle))
    with eye_path.open("r", encoding="utf-8-sig", newline="") as handle:
        eye_rows = list(csv.DictReader(handle))
    with four_path.open("r", encoding="utf-8-sig", newline="") as handle:
        four_rows = list(csv.DictReader(handle))
    sec_test_keys = {(row.get("subject", ""), row.get("session", "")) for row in motor_rows if row.get("session") == "sec_test"}
    eye_fields = _numeric_fields(eye_rows, JOIN_ID_FIELDS | REPORT_SCORE_FIELDS)
    four_fields = _numeric_fields(four_rows, JOIN_ID_FIELDS | REPORT_SCORE_FIELDS)
    eye_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    four_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in eye_rows:
        eye_by_key.setdefault((row.get("subject", ""), row.get("session", "")), []).append(row)
    for row in four_rows:
        four_by_key.setdefault((row.get("subject", ""), row.get("session", "")), []).append(row)

    joined: list[dict[str, object]] = []
    for motor in motor_rows:
        key = (motor.get("subject", ""), motor.get("session", ""))
        if key[1] == "sec" and (key[0], "sec_test") in sec_test_keys:
            continue
        row: dict[str, object] = dict(motor)
        row.update({f"eye_{name}": value for name, value in _aggregate_numeric(eye_by_key.get(key, []), eye_fields).items()})
        row.update({f"four_{name}": value for name, value in _aggregate_numeric(four_by_key.get(key, []), four_fields).items()})
        row["eye_process_metric_rows"] = len(eye_by_key.get(key, []))
        row["four_quantitative_rows"] = len(four_by_key.get(key, []))
        row["eye_total_score_model_feature"] = "no"
        joined.append(row)
    result_path = output / "tcm_multimodal_pilot.csv"
    write_rows(result_path, joined)
    (output / "tcm_multimodal_pilot_method.md").write_text(
        "# TCM pilot join method\n\n"
        "以 `motor_multimodal_features.csv` 为左表，按外层 `subject+session` 合并眼动过程量化字段和四诊原始量化字段；同键多条报告取非缺失数值的均值。\n\n"
        "`report_score_archive_only`（眼动报告总分）明确不进入输出特征；报告缺失保留为空。`2026081502/sec_test` 的 `bgh2.pdf` 经人工视觉核对为 `bgh1.pdf` 的重复报告，执行功能记录只计一次。若同一受试者同时存在 `sec_test`，则按 canonical 规则排除 `sec` 行。该输出不推断证型。\n",
        encoding="utf-8",
    )
    return result_path


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pdf_text_status(path: Path) -> str:
    """Classify whether a PDF exposes a machine-readable text layer."""
    executable = shutil.which("pdftotext")
    if executable is None:
        return "not_checked_pdftotext_unavailable"
    try:
        result = subprocess.run(
            [executable, "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "parse_check_error"
    if result.returncode != 0:
        return "parse_error"
    return "text_layer_available" if result.stdout.strip() else "image_only_no_text_layer"


def _report_parseability(eye: list[Path], four: list[Path]) -> tuple[str, str, str, str]:
    eye_status = ";".join(f"{path.name}:{_pdf_text_status(path)}" for path in eye)
    four_status = ";".join(f"{path.name}:image_only_manual_or_ocr" for path in four)
    if not eye:
        eye_reason = "no_eye_report_file"
    elif any("text_layer_available" in value for value in eye_status.split(";")):
        eye_reason = "machine_text_available_or_mixed"
    else:
        eye_reason = "pdf_has_no_machine_text_layer; quantitative transcription unavailable"
    if not four:
        four_reason = "no_four_diagnosis_report_file"
    else:
        four_reason = "image_files_only; structured quantitative fields unavailable without OCR_or_manual_review"
    return eye_status, eye_reason, four_status, four_reason


def inventory(output: Path, data_root: Path | None = None, analysis_root: Path | None = None) -> None:
    if data_root is not None:
        expanded.set_data_root(data_root)
    identity = expanded.load_identity(expanded.IDENTITY_XLSX)
    rows: list[dict[str, object]] = []
    for subject, name in identity.items():
        root = expanded.RAW_ROOT / subject
        if not root.exists():
            continue
        for session in sorted(item.name for item in root.iterdir() if item.is_dir()):
            eye = sorted((root / session / "眼动报告").glob("*.pdf"))
            four = sorted((root / session / "四诊报告").glob("*"))
            if not eye and not four:
                continue
            duplicate_map = EYE_DUPLICATES.get((subject, session), {})
            unique_eye = [path for path in eye if path.name not in duplicate_map]
            eye_status, eye_reason, four_status, four_reason = _report_parseability(
                unique_eye, [path for path in four if path.is_file()]
            )
            duplicate_text = ";".join(
                f"{name}:duplicate_of:{target}" for name, target in duplicate_map.items()
            )
            if duplicate_text:
                eye_status = ";".join(value for value in (eye_status, duplicate_text) if value)
                eye_reason = f"{duplicate_text}; {eye_reason}"
            manually_transcribed = [
                path.name for path in unique_eye
                if (subject, session, path.name) in MANUAL_EYE_REPORTS
            ]
            if manually_transcribed:
                eye_reason = eye_reason.replace(
                    "quantitative transcription unavailable",
                    "machine-readable quantitative text layer unavailable",
                )
                eye_reason = (
                    f"{eye_reason}; manually_visually_transcribed="
                    f"{','.join(manually_transcribed)}; second_review_pending"
                )
            rows.append({
                "subject": subject,
                "name": name,
                "session": session,
                "eye_report_count": len(eye),
                "eye_report_unique_count": len(unique_eye),
                "eye_report_duplicate_files": duplicate_text,
                "eye_report_paths": ";".join(str(path) for path in eye),
                "four_diagnosis_count": len([path for path in four if path.is_file()]),
                "four_diagnosis_paths": ";".join(str(path) for path in four if path.is_file()),
                "eye_machine_text_status": "TEXT_AVAILABLE_ONLY_IF_PDF_HAS_TEXT_LAYER",
                "four_diagnosis_extraction_status": "IMAGE_NEEDS_OCR_OR_MANUAL_REVIEW",
                "eye_report_parseability": eye_status,
                "eye_report_missing_reason": eye_reason,
                "four_diagnosis_parseability": four_status,
                "four_diagnosis_missing_reason": four_reason,
            })
    write_rows(output / "tcm_report_inventory.csv", rows)

    # Pilot transcriptions are reviewer-checked quantitative process values.
    # Report scores remain archive-only and no syndrome labels are transcribed.
    eye_rows = [
        {"subject": "2026081403", "name": "", "session": "base_line", "report": "cgz1.pdf", "task": "执行功能与注意力", "report_score_archive_only": 0, "fixation_saccades": 0, "pursuit_saccades": 11, "prosaccade_latency_ms": 419, "antisaccade_latency_ms": 959, "antisaccade_error_percent": 87.5, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081403", "name": "", "session": "base_line", "report": "cgz2.pdf", "task": "记忆力与注意力", "report_score_archive_only": 20, "target_fixation_time_ms": 1775, "target_fixation_count": 2, "distractor_fixation_time_ms": 268, "distractor_fixation_count": 1, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081501", "name": "", "session": "base_line", "report": "xwz1.pdf", "task": "执行功能与注意力", "report_score_archive_only": 0, "fixation_saccades": 5, "pursuit_saccades": 10, "prosaccade_latency_ms": 391, "antisaccade_latency_ms": 637, "antisaccade_error_percent": 37.5, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081201", "name": "", "session": "base_line", "report": "1.pdf", "task": "执行功能与注意力", "report_score_archive_only": 0, "fixation_saccades": 2, "pursuit_saccades": 2, "prosaccade_latency_ms": 793, "antisaccade_latency_ms": 1000, "antisaccade_error_percent": 100, "metric_use": "quantitative_process_metrics_only", "extraction_status": "pdf_text_layer", "confidence": "high"},
        {"subject": "2026081201", "name": "", "session": "base_line", "report": "2.pdf", "task": "记忆力与注意力", "report_score_archive_only": 20, "target_fixation_time_ms": 218, "target_fixation_count": 1, "distractor_fixation_time_ms": 416, "distractor_fixation_count": 1, "metric_use": "quantitative_process_metrics_only", "extraction_status": "pdf_text_layer", "confidence": "high"},
        {"subject": "2026081401", "name": "", "session": "base_line", "report": "zh1.pdf", "task": "执行功能与注意力", "report_score_archive_only": "", "fixation_saccades": 0, "pursuit_saccades": 13, "prosaccade_latency_ms": 412, "antisaccade_latency_ms": 772, "antisaccade_error_percent": 37.5, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081401", "name": "", "session": "sec_test", "report": "zh1.pdf", "task": "执行功能与注意力", "report_score_archive_only": "", "fixation_saccades": 2, "pursuit_saccades": 9, "prosaccade_latency_ms": 452, "antisaccade_latency_ms": 564, "antisaccade_error_percent": 0, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081401", "name": "", "session": "sec_test", "report": "zh2.pdf", "task": "记忆力与注意力", "report_score_archive_only": "", "target_fixation_time_ms": 1676, "target_fixation_count": 2, "distractor_fixation_time_ms": 380, "distractor_fixation_count": 1, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081502", "name": "", "session": "base_line", "report": "bgh1.pdf", "task": "执行功能与注意力", "report_score_archive_only": "", "fixation_saccades": 1, "pursuit_saccades": 4, "prosaccade_latency_ms": 468, "antisaccade_latency_ms": 498, "antisaccade_error_percent": 25, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081502", "name": "", "session": "sec_test", "report": "bgh1.pdf", "task": "执行功能与注意力", "report_score_archive_only": "", "fixation_saccades": 1, "pursuit_saccades": 4, "prosaccade_latency_ms": 438, "antisaccade_latency_ms": 696, "antisaccade_error_percent": 12.5, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081701", "name": "", "session": "base_line", "report": "djs1.pdf", "task": "执行功能与注意力", "report_score_archive_only": "", "fixation_saccades": 0, "pursuit_saccades": 8, "prosaccade_latency_ms": 348, "antisaccade_latency_ms": 520, "antisaccade_error_percent": 25, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081701", "name": "", "session": "base_line", "report": "djs2.pdf", "task": "记忆力与注意力", "report_score_archive_only": "", "target_fixation_time_ms": 1116, "target_fixation_count": 1, "distractor_fixation_time_ms": 161, "distractor_fixation_count": 1, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081701", "name": "", "session": "sec_test", "report": "djs1.pdf", "task": "执行功能与注意力", "report_score_archive_only": "", "fixation_saccades": 0, "pursuit_saccades": 4, "prosaccade_latency_ms": 287, "antisaccade_latency_ms": 356, "antisaccade_error_percent": 0, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
        {"subject": "2026081701", "name": "", "session": "sec_test", "report": "djs2.pdf", "task": "记忆力与注意力", "report_score_archive_only": "", "target_fixation_time_ms": 1678, "target_fixation_count": 1, "distractor_fixation_time_ms": 178, "distractor_fixation_count": 1, "metric_use": "quantitative_process_metrics_only", "extraction_status": "manual_visual", "confidence": "high"},
    ]
    # The outer ID and the Excel cross-check table are authoritative for names.
    for row in eye_rows:
        row["name"] = identity.get(str(row["subject"]), "")
    write_rows(output / "eye_report_pilot_features.csv", eye_rows)

    four_rows = [
        {"subject": "2026080901", "name": "", "session": "base_line", "tongue_color": "淡红", "coating_color": "苔白", "coating_thin_score": 0.42, "coating_greasy_score": 0.67, "prickles_count": 2, "crack_score": 1.10, "tooth_marks_count": 2, "tongue_body": "正常", "peeling": "有剥苔", "left_pulse_position": "沉", "right_pulse_position": "中", "left_pulse_rate_per_min": 78, "right_pulse_rate_per_min": 76, "left_pulse_tension": "弦", "right_pulse_tension": "弦", "extraction_status": "manual_visual", "confidence": "medium"},
        {"subject": "2026080901", "name": "", "session": "sec_test", "tongue_color": "淡红", "coating_color": "苔白", "coating_thin_score": 0.61, "coating_greasy_score": 0.72, "prickles_count": 16, "crack_score": 1.68, "tooth_marks_count": 0, "tongue_body": "胖", "peeling": "无剥苔", "left_pulse_position": "沉", "right_pulse_position": "沉", "left_pulse_rate_per_min": 109, "right_pulse_rate_per_min": 107, "left_pulse_tension": "弦", "right_pulse_tension": "弦", "extraction_status": "manual_visual", "confidence": "medium"},
        {"subject": "2026081403", "name": "", "session": "base_line", "tongue_color": "淡红", "coating_color": "黄白相兼", "coating_thin_score": 0.58, "coating_greasy_score": 0.61, "prickles_count": 11, "crack_score": 0.71, "tooth_marks_count": 2, "tongue_body": "正常", "peeling": "无剥苔", "left_pulse_position": "沉", "right_pulse_position": "中", "left_pulse_rate_per_min": 65, "right_pulse_rate_per_min": 64, "left_pulse_tension": "弦", "right_pulse_tension": "正常", "extraction_status": "manual_visual", "confidence": "medium"},
        {"subject": "2026081501", "name": "", "session": "base_line", "tongue_color": "淡红", "coating_color": "苔白", "coating_thin_score": 0.44, "coating_greasy_score": 0.56, "prickles_count": 3, "crack_score": 0.59, "tooth_marks_count": 0, "tongue_body": "正常", "peeling": "无剥苔", "left_pulse_position": "浮", "right_pulse_position": "中", "left_pulse_rate_per_min": 76, "right_pulse_rate_per_min": 73, "left_pulse_tension": "弦", "right_pulse_tension": "弦", "extraction_status": "manual_visual", "confidence": "medium"},
    ]
    for row in four_rows:
        row["name"] = identity.get(str(row["subject"]), "")
    write_rows(output / "four_diagnosis_pilot_features.csv", four_rows)
    build_pilot_join(analysis_root or Path("outputs/analysis/20260901_all_cases"), output)
    paired7 = {"2026081401", "2026081502", "2026081701"}
    parse_rows = [
        row for row in rows
        if row["subject"] in paired7 and row["session"] == "sec_test"
    ]
    parse_lines = [
        "# n=7 新增 sec_test 报告解析性检查",
        "",
        "身份键来自外层编号，姓名来自 `编号姓名数据核对表.xlsx`。本检查只记录文件存在性和机器可读性，不从报告评分推导特征，也不从四诊字段推断证型。",
        "",
        "| subject | name | eye reports (raw/unique) | eye report parseability | eye review status / missing reason | four diagnosis parseability | four diagnosis missing reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in parse_rows:
        values = [
            str(row.get("subject", "")),
            str(row.get("name", "")),
            f"{row.get('eye_report_count', '')}/{row.get('eye_report_unique_count', '')}",
            str(row.get("eye_report_parseability", "")),
            str(row.get("eye_report_missing_reason", "")),
            str(row.get("four_diagnosis_parseability", "")),
            str(row.get("four_diagnosis_missing_reason", "")),
        ]
        parse_lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    (output / "paired7_sec_test_report_parseability.md").write_text(
        "\n".join(parse_lines) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 中医报告接入说明\n\n"
        "`tcm_report_inventory.csv` 是文件级覆盖清单；`paired7_sec_test_report_parseability.md` 记录新增 n=7 sec_test 报告的可解析性与缺失原因。`eye_report_pilot_features.csv` 的分析字段只保留人工视觉核对的量化过程值，眼动报告总分是 archive-only，不进入模型；`2026081502/sec_test` 的 bgh2.pdf 已按人工视觉核对标记为 bgh1.pdf 重复并只计一次。`four_diagnosis_pilot_features.csv` 的分析字段只保留原始量化字段，不自动映射为证型。所有示范性转录仍需第二位研究人员逐项核对原报告。\n",
        encoding="utf-8",
    )
    print(f"report_sessions={len(rows)} output={output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="清点并试点接入眼动/四诊报告")
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments/tcm_20260901"))
    parser.add_argument("--data-root", type=Path, default=None, help="raw-data root (default: historical J: path)")
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/analysis/20260901_all_cases"), help="motor feature analysis directory")
    args = parser.parse_args()
    inventory(args.out, data_root=args.data_root, analysis_root=args.analysis_root)


if __name__ == "__main__":
    main()
