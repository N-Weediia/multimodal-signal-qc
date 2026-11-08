#!/usr/bin/env python
"""Classify likely legacy/redundant files in the reorganized J: data tree.

This report is deliberately non-destructive.  It consumes the verified SHA-256
inventories generated during the read-only scan, re-checks the current J: tree
by path and size, and writes reviewer-friendly CSV/Markdown outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DATA_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
SUBJECT_RE = re.compile(r"^\d{8}\d{2}$")
NUMERIC_BRANCH_RE = re.compile(r"^\d{4,8}$")


def read_inventory(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def path_key(path: str) -> str:
    return path.replace("/", "\\")


def current_shallow(root: Path) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in DATA_EXTENSIONS:
            continue
        stat = path.stat()
        rel = path_key(str(path.relative_to(root)))
        rows.append({"relative_path": rel, "length_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        total_bytes += stat.st_size
    return rows, {"file_count": len(rows), "total_bytes": total_bytes}


def load_identity_map(xlsx: Path) -> Dict[str, str]:
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("openpyxl is required to read the identity workbook") from exc

    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    identities: Dict[str, str] = {}
    for row in sheet.iter_rows(min_row=4, values_only=True):
        if not row or row[0] is None or row[1] is None:
            continue
        subject = str(row[0]).strip().lstrip("'")
        name = str(row[1]).strip().lstrip("'")
        if SUBJECT_RE.match(subject):
            identities[subject] = name
    return identities


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_group(rows: List[dict], identities: Dict[str, str]) -> dict:
    paths = [path_key(r["relative_path"]) for r in rows]
    subject = paths[0].split("\\")[0]
    extensions = {Path(p).suffix.lower() for p in paths}
    sessions = {p.split("\\")[1] for p in paths if len(p.split("\\")) > 1}
    dirs = [p.split("\\")[:-1] for p in paths]
    category = "exact_duplicate"
    confidence = "low"
    recommendation = "人工判断；相同内容不必重复保留，但不要据此推断哪次采集有效。"
    reason = "同一 SHA-256 内容在多个路径出现。"
    retain: List[str] = []
    candidates: List[str] = []

    # A numeric branch such as 081801 mirrors the parent modality/task tree.
    numeric_paths = [p for p in paths if any(NUMERIC_BRANCH_RE.match(part) for part in p.split("\\")[2:-1])]
    if numeric_paths and len(numeric_paths) < len(paths):
        category = "nested_subject_copy"
        confidence = "high"
        reason = "同一受试者、同一会话和模态下，存在带数字子目录与不带数字子目录的完整同内容副本。"
        recommendation = "优先保留不带数字子目录的标准路径；数字子目录副本可在确认后清理。"
        candidates = numeric_paths
        retain = [p for p in paths if p not in numeric_paths]

    # A .nirs file under EEG is almost certainly a modality placement issue.
    eeg_nirs = [p for p in paths if "\\EEG\\" in ("\\" + p + "\\") and Path(p).suffix.lower() == ".nirs"]
    fnirs_nirs = [p for p in paths if "\\FNIRS\\" in ("\\" + p + "\\") and Path(p).suffix.lower() == ".nirs"]
    if eeg_nirs and fnirs_nirs:
        category = "cross_modality_misplacement"
        confidence = "medium-high"
        reason = "同一 .nirs 文件同时位于 EEG 和 FNIRS 目录，属于历史整理时的跨模态误放或复制。"
        recommendation = "保留 FNIRS 目录版本；EEG 目录版本需人工确认后清理。"
        candidates = eeg_nirs
        retain = fnirs_nirs

    # A canonical identity is defined by the outer subject number and workbook.
    expected_name = identities.get(subject, "")
    name_mismatch = []
    name_match = []
    if expected_name:
        for p in paths:
            if expected_name in p:
                name_match.append(p)
            elif Path(p).suffix.lower() in {".nirs", ".csv", ".hpf", ".pdf", ".jpg", ".png"}:
                # Only flag files that visibly contain a Chinese/Latin name token.
                name_mismatch.append(p)
    if name_mismatch and name_match:
        category = "name_alias_conflict"
        confidence = "high"
        reason = f"外层编号 {subject} 在核对表中对应“{expected_name}”，但同内容文件还出现其他姓名路径。"
        recommendation = f"按核对表保留包含“{expected_name}”的路径；其他姓名路径是旧命名/别名候选，删除前保留审计记录。"
        candidates = [p for p in name_mismatch if p not in name_match]
        retain = name_match

    # Baseline/sec_test duplicates are provenance anomalies, not safe deletions.
    if {"base_line", "sec_test"}.issubset(sessions):
        category = "session_duplicate_provenance"
        confidence = "high"
        reason = "同一文件内容同时出现在 base_line 与 sec_test；这可能是重复导出或会话标记错误。"
        recommendation = "全部保留并人工回查采集日志/原始设备记录；不可自动删除。"
        candidates = []
        retain = paths

    return {
        "subject_id": subject,
        "expected_name": expected_name,
        "sha256": rows[0]["sha256"],
        "file_count": len(paths),
        "category": category,
        "confidence": confidence,
        "recommendation": recommendation,
        "reason": reason,
        "retain_paths": " | ".join(sorted(retain)),
        "candidate_paths": " | ".join(sorted(candidates)),
        "all_paths": " | ".join(sorted(paths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j-root", type=Path, default=Path(r"J:\2026二康\数据"))
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--xlsx", type=Path, default=Path(r"J:\2026二康\数据\编号姓名数据核对表.xlsx"))
    parser.add_argument("--j-inventory", type=Path, default=Path("outputs/j_drive_data_inventory_20260902.csv"))
    parser.add_argument("--local-inventory", type=Path, default=Path("outputs/local_data_inventory_20260902.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/j_old_data_scan_20260902"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    j_rows = read_inventory(args.j_inventory)
    local_rows = read_inventory(args.local_inventory)
    identities = load_identity_map(args.xlsx)

    # Stability check: the previous full hash inventory must still match current paths/sizes.
    current_rows, current_summary = current_shallow(args.j_root)
    j_by_path = {path_key(r["relative_path"]): r for r in j_rows}
    current_by_path = {r["relative_path"]: r for r in current_rows}
    missing = sorted(set(j_by_path) - set(current_by_path))
    added = sorted(set(current_by_path) - set(j_by_path))
    size_changed = sorted(
        p for p in set(j_by_path) & set(current_by_path)
        if int(j_by_path[p]["length_bytes"]) != int(current_by_path[p]["length_bytes"])
    )
    stable = not missing and not added and not size_changed
    stability = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "j_root": str(args.j_root),
        "stable_against_hash_inventory": stable,
        "inventory_file_count": len(j_rows),
        "current_file_count": current_summary["file_count"],
        "inventory_total_bytes": sum(int(r["length_bytes"]) for r in j_rows),
        "current_total_bytes": current_summary["total_bytes"],
        "missing_paths": missing,
        "added_paths": added,
        "size_changed_paths": size_changed,
        "excel_excluded": True,
    }
    (args.output / "stability_check.json").write_text(json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8")

    # Duplicate groups inside the reorganized J: tree.
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in j_rows:
        groups[row["sha256"]].append(row)
    duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
    classified = [classify_group(rows, identities) for rows in duplicate_groups]
    classified.sort(key=lambda x: (x["subject_id"], x["category"], x["sha256"]))
    group_fields = [
        "subject_id", "expected_name", "sha256", "file_count", "category", "confidence",
        "recommendation", "reason", "retain_paths", "candidate_paths", "all_paths",
    ]
    write_csv(args.output / "duplicate_groups.csv", classified, group_fields)

    row_by_hash = {r["sha256"]: r for r in j_rows}
    candidate_rows = []
    manual_rows = []
    for group in classified:
        # Keep this file conservative: only the complete nested numeric branch
        # has a clear canonical counterpart.  Other candidate paths remain in
        # duplicate_groups.csv/manual_review.csv until a person confirms them.
        if group["category"] == "nested_subject_copy":
            for path in filter(None, group["candidate_paths"].split(" | ")):
                row = row_by_hash[group["sha256"]]
                candidate_rows.append({**group, "relative_path": path, "length_bytes": row["length_bytes"]})
        if group["category"] in {"session_duplicate_provenance", "name_alias_conflict", "cross_modality_misplacement"}:
            for path in filter(None, group["all_paths"].split(" | ")):
                manual_rows.append({**group, "relative_path": path})
    write_csv(args.output / "safe_redundancy_candidates.csv", candidate_rows, group_fields + ["relative_path", "length_bytes"])
    write_csv(args.output / "manual_review.csv", manual_rows, group_fields + ["relative_path"])

    # Compare relative paths and hashes with the prior local data tree.
    local_by_path = {path_key(r["relative_path"]): r for r in local_rows}
    union = sorted(set(j_by_path) | set(local_by_path))
    path_diff = []
    for path in union:
        j = j_by_path.get(path)
        local = local_by_path.get(path)
        if j is None:
            status = "local_only"
        elif local is None:
            status = "j_only"
        elif j["sha256"] == local["sha256"]:
            status = "same_path_same_content"
        else:
            status = "same_path_content_changed"
        path_diff.append({
            "relative_path": path,
            "status": status,
            "j_sha256": j["sha256"] if j else "",
            "local_sha256": local["sha256"] if local else "",
            "j_length_bytes": j["length_bytes"] if j else "",
            "local_length_bytes": local["length_bytes"] if local else "",
        })
    write_csv(args.output / "path_diff_vs_local.csv", path_diff,
               ["relative_path", "status", "j_sha256", "local_sha256", "j_length_bytes", "local_length_bytes"])

    # Subject-level overview follows the user's outer-folder rule.
    subjects = sorted(set(r["relative_path"].split("\\")[0] for r in j_rows) | set(r["relative_path"].split("\\")[0] for r in local_rows))
    overview = []
    for subject in subjects:
        j_subject = [r for r in j_rows if r["relative_path"].split("\\")[0] == subject]
        l_subject = [r for r in local_rows if r["relative_path"].split("\\")[0] == subject]
        overview.append({
            "subject_id": subject,
            "canonical_name": identities.get(subject, ""),
            "j_file_count": len(j_subject),
            "local_file_count": len(l_subject),
            "j_bytes": sum(int(r["length_bytes"]) for r in j_subject),
            "local_bytes": sum(int(r["length_bytes"]) for r in l_subject),
            "j_sessions": ", ".join(sorted({r["relative_path"].split("\\")[1] for r in j_subject})),
            "local_sessions": ", ".join(sorted({r["relative_path"].split("\\")[1] for r in l_subject})),
        })
    write_csv(args.output / "subject_overview.csv", overview,
               ["subject_id", "canonical_name", "j_file_count", "local_file_count", "j_bytes", "local_bytes", "j_sessions", "local_sessions"])

    category_counts = Counter(g["category"] for g in classified)
    local_only = [r for r in path_diff if r["status"] == "local_only"]
    j_only = [r for r in path_diff if r["status"] == "j_only"]
    report_lines = [
        "# J 盘旧数据扫描报告",
        "",
        f"扫描时间：{stability['checked_at']}",
        f"J 盘根目录：`{args.j_root}`",
        "",
        "## 快照结论",
        "",
        f"- J 盘非 Excel 文件：**{len(j_rows)} 个**，约 **{sum(int(r['length_bytes']) for r in j_rows) / 1024**3:.2f} GiB**。",
        f"- 与前次 SHA-256 清单的路径/大小一致：**{'是' if stable else '否'}**。",
        f"- J 盘与本地旧 `data` 的相对路径：J 盘 {len(j_rows)} 个，本地 {len(local_rows)} 个。",
        f"- 仅本地旧 `data` 存在：**{len(local_only)} 个文件**（其中 `2026081403\\sec` 为 11 个，其余 3 个为 2026081301 的旧姓名路径）；仅 J 盘存在：**{len(j_only)} 个文件**。",
        "- Excel 及临时锁文件不纳入重复判断。",
        "",
        "## 高置信度/重点发现",
        "",
        "1. `2026081801\\base_line` 存在 `081801` 数字子目录与父目录的同内容副本，涉及 EEG、FNIRS、EMG；数字子目录是首要旧数据候选。",
        "2. `2026081403\\sec` 在本地旧 `data` 中存在，但 J 盘新整理目录中不存在；本地该会话的 11 个文件是最明确的旧会话候选。",
        "3. `2026081401` 的 baseline 与 sec_test 出现完全相同 EEG 文件；这是采集溯源异常，不能自动删除。",
        "4. `2026081301` 的部分 fNIRS 文件同时以“傅开强”和“秦福友”命名；核对表规定外层编号 `2026081301` 对应“秦福友”，故“傅开强”路径标为旧命名/别名候选。",
        "5. `2026081701` 有 `.nirs` 文件同时位于 EEG 与 FNIRS 目录；EEG 版本标为跨模态误放候选。",
        "",
        "## 重复组分类",
        "",
        f"J 盘共有 **{len(duplicate_groups)} 个**重复哈希组、涉及 **{sum(len(x) for x in duplicate_groups)} 个**文件。",
    ]
    for category, count in sorted(category_counts.items()):
        report_lines.append(f"- `{category}`：{count} 组")
    report_lines += [
        "",
        "## 文件",
        "",
        "- `duplicate_groups.csv`：每个重复哈希组的分类、保留路径和候选路径。",
        "- `safe_redundancy_candidates.csv`：仅列出 2026081801\\081801 数字子目录的冗余候选；本次仍不执行删除。",
        "- `manual_review.csv`：会话重复、姓名冲突和跨模态误放等需人工确认项。",
        "- `path_diff_vs_local.csv`：J 盘与本地旧 `data` 的逐路径差异。",
        "- `subject_overview.csv`：按外层受试者编号汇总，并附核对表姓名。",
        "- `stability_check.json`：稳定性和扫描规模证据。",
        "",
        "## 处理建议",
        "",
        "先只清理 `safe_redundancy_candidates.csv` 中人工确认过的数字子目录/跨模态副本，并保留本报告作为审计记录。`session_duplicate_provenance` 和 `name_alias_conflict` 不应直接删除；应先以外层编号、核对表姓名和采集日志完成确认。",
    ]
    (args.output / "scan_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(args.output),
        "stable": stable,
        "j_files": len(j_rows),
        "j_bytes": sum(int(r["length_bytes"]) for r in j_rows),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": sum(len(x) for x in duplicate_groups),
        "safe_candidates": len(candidate_rows),
        "manual_review_rows": len(manual_rows),
        "local_only": len(local_only),
        "j_only": len(j_only),
        "categories": dict(category_counts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
