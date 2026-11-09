"""Safely copy selected numeric subject folders from the raw-data drive.

The default mode is a dry run.  ``--apply`` is required to write files under
the local ``data/`` directory; existing files with a different hash are never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_ROOT = Path(r"J:\2026二康\数据")
DEFAULT_DEST_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs") / "phase2_20260901"
ALLOWED_EXTENSIONS = {
    ".bdf",
    ".csv",
    ".nirs",
    ".hcx",
    ".hpf",
    ".pdf",
    ".jpg",
    ".png",
    ".xlsx",
    ".json",
}
SUBJECT_PATTERN = re.compile(r"^\d+$")


@dataclass
class CopyRecord:
    subject: str
    source_relative_path: str
    destination_relative_path: str
    extension: str
    size_bytes: int
    source_sha256: str
    destination_sha256: str
    status: str
    reason: str


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def parse_subjects(value: str, source_root: Path) -> list[str]:
    if value.strip().lower() == "all":
        if not source_root.is_dir():
            return []
        return sorted(path.name for path in source_root.iterdir() if path.is_dir() and SUBJECT_PATTERN.fullmatch(path.name))
    subjects = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [subject for subject in subjects if not SUBJECT_PATTERN.fullmatch(subject)]
    if invalid:
        raise ValueError(f"subjects must be pure numeric outer IDs; invalid: {invalid}")
    return list(dict.fromkeys(subjects))


def iter_source_files(source_root: Path, subject: str) -> Iterable[Path]:
    subject_root = (source_root / subject).resolve()
    try:
        subject_root.relative_to(source_root.resolve())
    except ValueError:
        return
    if not subject_root.is_dir():
        return
    for path in sorted(subject_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(subject_root)
        except ValueError:
            continue
        yield path


def plan_copy(source_root: Path, dest_root: Path, subjects: Iterable[str], apply: bool = False) -> list[CopyRecord]:
    records: list[CopyRecord] = []
    for subject in subjects:
        subject_root = source_root / subject
        if not subject_root.is_dir():
            records.append(CopyRecord(subject, "", f"{subject}/", "", 0, "", "", "source_missing", "subject directory not found"))
            continue
        for source_path in iter_source_files(source_root, subject):
            relative = source_path.relative_to(source_root).as_posix()
            destination_path = dest_root / Path(relative)
            source_hash, size = sha256_and_size(source_path)
            destination_hash = ""
            if destination_path.exists():
                if not destination_path.is_file():
                    records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, "", "conflict", "destination exists and is not a file"))
                    continue
                destination_hash, destination_size = sha256_and_size(destination_path)
                if destination_hash != source_hash or destination_size != size:
                    records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, destination_hash, "conflict", "destination hash or size differs; never overwritten"))
                    continue
                records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, destination_hash, "existing_same_hash", "destination already matches source"))
                continue
            if not apply:
                records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, "", "planned", "dry-run; copy not requested"))
                continue
            try:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                destination_hash, destination_size = sha256_and_size(destination_path)
                if destination_hash != source_hash or destination_size != size:
                    records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, destination_hash, "failed", "post-copy hash or size mismatch"))
                else:
                    records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, destination_hash, "copied", "copied and hash-verified"))
            except OSError as exc:
                records.append(CopyRecord(subject, relative, relative, source_path.suffix.lower(), size, source_hash, destination_hash, "failed", f"copy failed: {exc}"))
    return records


def write_outputs(records: list[CopyRecord], output_dir: Path, source_root: Path, dest_root: Path, subjects: Iterable[str], apply: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "raw_copy_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "source_relative_path", "destination_relative_path", "extension", "size_bytes", "source_sha256", "destination_sha256", "status", "reason"])
        writer.writerows((record.subject, record.source_relative_path, record.destination_relative_path, record.extension, record.size_bytes, record.source_sha256, record.destination_sha256, record.status, record.reason) for record in records)
    counts = {status: sum(record.status == status for record in records) for status in ("planned", "copied", "existing_same_hash", "conflict", "source_missing", "failed")}
    bytes_total = sum(record.size_bytes for record in records if record.status in {"planned", "copied", "existing_same_hash"})
    mode = "APPLY" if apply else "DRY-RUN"
    lines = [
        "# Raw Subset Copy Log",
        "",
        f"Mode: **{mode}**; source=`{source_root}`; destination=`{dest_root}`.",
        f"Subjects: {', '.join(subjects) or '(none found)'}",
        "",
        f"- Planned/copied/already matching: {counts['planned'] + counts['copied'] + counts['existing_same_hash']} files, {bytes_total} bytes",
        f"- Copied: {counts['copied']}",
        f"- Existing same hash: {counts['existing_same_hash']}",
        f"- Conflicts (not overwritten): {counts['conflict']}",
        f"- Missing subjects: {counts['source_missing']}",
        f"- Failed: {counts['failed']}",
        "- Allowed extensions: .bdf, .csv, .nirs, .hcx, .hpf, .pdf, .jpg, .png, .xlsx, .json",
        "- No non-numeric outer subject directory is selected.",
    ]
    (output_dir / "raw_copy_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--subjects", default="all", help="all or comma-separated numeric outer IDs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply", action="store_true", help="copy files; default is dry-run")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    output_dir = args.output_dir.resolve()
    try:
        subjects = parse_subjects(args.subjects, source_root)
    except ValueError as exc:
        raise SystemExit(str(exc))
    records = plan_copy(source_root, dest_root, subjects, apply=args.apply)
    write_outputs(records, output_dir, source_root, dest_root, subjects, args.apply)
    counts = {status: sum(record.status == status for record in records) for status in ("planned", "copied", "existing_same_hash", "conflict", "source_missing", "failed")}
    bytes_total = sum(record.size_bytes for record in records if record.status in {"planned", "copied", "existing_same_hash"})
    print(f"mode={'apply' if args.apply else 'dry-run'} subjects={len(subjects)} planned_or_copied_files={counts['planned'] + counts['copied']} bytes={bytes_total} conflicts={counts['conflict']} missing={counts['source_missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
