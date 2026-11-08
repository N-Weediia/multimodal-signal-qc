"""Build a reusable analysis contract and task-level provenance ledger.

The contract is intentionally data-root agnostic: new subjects can be added to
the same directory and the command can be rerun without changing task labels or
primary-endpoint rules.  It only inventories source files and existing QC
outputs; it never edits raw recordings and never imputes missing observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Allow direct ``python scripts/...`` execution from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import discover_recordings, recording_modality, task_label_for
from emg_qc import (
    EXPECTED_SENSOR_IDS,
    MAPPING_PROFILE_SHIFTED_2026080901,
    MAPPING_PROFILE_STANDARD,
    SENSOR_MUSCLES,
    SENSOR_TO_MUSCLE_INDEX,
    SHIFTED_SENSOR_IDS,
)
from preprocess import PREPROCESS_PIPELINE_VERSION, _preprocess_parameter_hash


EXPECTED_BLOCKS = {"摸头": 5, "摸膝": 5, "摸腰": 5}
TASK_ORDER = ("摸头", "摸膝", "摸腰")
SESSIONS = ("base_line", "sec_test")
EMG_EXPECTED_SENSOR_IDS = list(EXPECTED_SENSOR_IDS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, UnicodeError):
        return pd.DataFrame()


def _path_key(path: str) -> str:
    return str(path).replace("/", "\\")


def _inventory_rows(data_root: Path, output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recording in discover_recordings(data_root, None):
        modality = recording_modality(recording)
        if modality is None:
            continue
        relative = recording.relative_to(data_root)
        parts = relative.parts
        subject = parts[0] if parts else "外部输入"
        session = parts[1] if len(parts) > 1 else "unknown_session"
        outer_task = task_label_for(recording)
        # A single movement file contains the three protocol-labeled actions.
        # Expand it here so the ledger has the requested subject/session/task
        # grain; the source path and hash remain identical across the actions.
        task_names = TASK_ORDER if outer_task == "运动任务" else (outer_task,)
        digest = sha256_file(recording)
        for task_name in task_names:
            rows.append({
                "subject": subject,
                "session": session,
                "task": outer_task,
                "task_name": task_name,
                "modality": modality,
                "source_relative_path": str(relative),
                "source_filename": recording.name,
                "source_size_bytes": recording.stat().st_size,
                "source_sha256": digest,
                "path_gate": "PASS",
                "signal_gate": "PENDING",
                "alignment_gate": "NOT_APPLICABLE" if modality != "EMG" else "PENDING",
                "expected_blocks": EXPECTED_BLOCKS.get(task_name, ""),
                "observed_blocks": "",
                "valid_blocks": "",
                "exclusion_reason": "",
                "preprocessed_directory": str(output_root / relative.parent / recording.stem),
            })
    return rows


def _merge_task_outputs(rows: list[dict[str, Any]], analysis_root: Path) -> None:
    """Attach already-produced task summaries without making them mandatory."""
    summary = _read_csv(analysis_root / "movement_subtask_summary.csv")
    emg = _read_csv(analysis_root / "movement_subtask_emg_summary.csv")
    # Current canonical summary uses modality/name columns.  Keep the merge
    # conservative: it only fills counts for matching subject/session/task.
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for table, modality in ((summary, "SUMMARY"), (emg, "EMG")):
        if table.empty:
            continue
        for _, item in table.iterrows():
            subject = str(item.get("subject", ""))
            session = str(item.get("session", ""))
            task = str(item.get("task_name", item.get("task", "")))
            if not subject or not session or not task:
                continue
            key = (subject, session, task)
            entry = summaries.setdefault(key, {})
            row_modality = str(item.get("modality", "")).upper()
            if modality == "SUMMARY":
                if row_modality not in {"FNIRS", "EEG"}:
                    continue
                metric = "observed_blocks_" + row_modality.lower()
            else:
                row_modality = "EMG"
                metric = "observed_blocks_emg"
            # One summary row can already carry event_count; the EMG table is
            # block-wise, so its group size is the observed block count.
            value = item.get("event_count", "")
            if modality == "EMG":
                value = entry.get(metric, 0) + 1
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                value = ""
            entry[metric] = value
    for row in rows:
        key = (str(row["subject"]), str(row["session"]), str(row["task_name"]))
        values = summaries.get(key, {})
        modality = str(row["modality"]).lower()
        observed = values.get("observed_blocks_" + modality, "")
        if observed != "":
            row["observed_blocks"] = observed
            # Existing summaries report observed/event markers, not a full
            # channel-and-window validity decision.  Do not turn that count
            # into a PASS until the task-level masks are available.
            row["signal_gate"] = "PENDING_TASK_LEVEL_QC"
        elif row["task"] == "运动任务":
            row["signal_gate"] = "PENDING_TASK_LEVEL_QC"
            row["exclusion_reason"] = "尚未有任务级摘要；不得把文件存在性当作有效窗口"
        else:
            row["signal_gate"] = "PENDING_TASK_LEVEL_QC"


def build_contract(data_root: Path, output_root: Path, analysis_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _inventory_rows(data_root, output_root)
    _merge_task_outputs(rows, analysis_root)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    parameter_payload = {
        "contract_version": "2026-09-04.v1",
        "task_order": TASK_ORDER,
        "expected_blocks": EXPECTED_BLOCKS,
        "emg_expected_sensor_ids": EMG_EXPECTED_SENSOR_IDS,
        "emg_sensor_to_muscle_index": {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
        "emg_mapping_profiles": {
            MAPPING_PROFILE_STANDARD: {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
            MAPPING_PROFILE_SHIFTED_2026080901: {str(k): index for index, k in enumerate(SHIFTED_SENSOR_IDS, start=1)},
        },
        "alignment_threshold_ms": 250,
        "severe_alignment_threshold_ms": 1000,
        "missing_policy": "no_imputation",
    }
    parameter_hash = hashlib.sha256(
        json.dumps(parameter_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    preprocess_parameter_hash = _preprocess_parameter_hash()
    for row in rows:
        row["pipeline_version"] = PREPROCESS_PIPELINE_VERSION
        row["analysis_parameter_hash"] = parameter_hash
        row["preprocess_parameter_hash"] = preprocess_parameter_hash
    contract = {
        "contract_version": "2026-09-04.v1",
        "created_at": now,
        "analysis_parameter_hash": parameter_hash,
        "preprocess_parameter_hash": preprocess_parameter_hash,
        "data_root": str(data_root.resolve()),
        "data_root_policy": "工程内 data/data_repro 目录；新增受试者自动发现，外层文件夹决定任务语义",
        "independent_unit": "subject",
        "sessions": list(SESSIONS),
        "tasks": list(TASK_ORDER),
        "task_order_authority": "nearest matching parent directory; filename ignored",
        "protocol": {
            "task_duration_seconds": 5,
            "inter_block_rest_seconds": 15,
            "expected_blocks_per_task": EXPECTED_BLOCKS,
            "delta_direction": "sec_test - base_line",
        },
        "gates": {
            "path_gate": "file exists and parses",
            "signal_gate": "channel/window/task quality passes predefined thresholds",
            "alignment_gate": "same block cross-modal residual <=250 ms; >1 s is sensitivity exclusion",
            "missing_policy": "retain NaN and exclusion reason; never impute zero",
        },
        "emg": {
            "role": "peripheral execution proxy, not kinematics",
            "expected_sensor_ids": EMG_EXPECTED_SENSOR_IDS,
            "sensor_to_muscle_index": {str(k): v for k, v in SENSOR_TO_MUSCLE_INDEX.items()},
            "sensor_to_muscle_label": {str(k): v for k, v in SENSOR_MUSCLES.items()},
            "logical_muscle_index": "confirmed table order: 2->1, 3->2, 4->3, 5->4, 9->5, 10->6, 11->7",
            "mapping_profiles": {
                MAPPING_PROFILE_STANDARD: "standard physical IDs 2,3,4,5,9,10,11",
                MAPPING_PROFILE_SHIFTED_2026080901: "2026080901/base_line physical IDs 1,2,3,4,5,9,10 mapped by logical order",
            },
            "missing_sensor_policy": "compare by logical muscle index; preserve canonical missing/unexpected IDs for audit; sensor 1 is unexpected by default, and only the explicitly confirmed 2026080901/base_line record uses shifted mapping",
            "amplitude_unit": "source CSV unit; no MVC normalization claim",
        },
        "fnirs": {
            "primary_branch": "optical-density proxy with channel and motion masks",
            "sensitivity_branch": "modified Beer-Lambert HbO/HbR with explicit DPF/geometry parameters",
            "short_separation": "not available; no short-separation regression claim",
        },
        "eeg": {
            "primary_role": "quality-gated exploratory control proxy",
            "incomplete_visit_policy": "exclude from EEG and brain-muscle nodes",
        },
        "primary_endpoint": {
            "name": "touch-head minus touch-knee task-space magnitude delta",
            "label": "摸头 - 摸膝 fNIRS task-space absolute-response contrast",
            "status": "exploratory candidate; not a validated biomarker",
        },
        "ai": {
            "framework": "transparent quality-gated multimodal graph",
            "required_checks": ["exact permutation diagnostic", "leave-one-subject-out range", "self-excluded attribution"],
            "prohibited_claims": ["causal edge", "closed-loop BCI", "clinical prediction", "SHAP causal importance"],
        },
        "rows": len(rows),
    }
    (output_root / "analysis_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "task_qc_ledger.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(rows[0]) if rows else ["subject", "session", "task", "modality"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the incremental preprocessing contract and task QC ledger.")
    parser.add_argument("--data-root", type=Path, default=Path("data_repro_20260903"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/preprocessed/current"))
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/analysis/20260903_all_cases_canonical"))
    args = parser.parse_args()
    contract = build_contract(args.data_root, args.output_root, args.analysis_root)
    print(json.dumps({"output_root": str(args.output_root), "records": contract["rows"], "contract_version": contract["contract_version"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
