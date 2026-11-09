"""Conservative, candidate-list-driven cleanup for the phase-one audit.

The default mode is a dry run.  Deletion requires ``--apply`` and every file is
revalidated against the candidate manifest immediately before removal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_TYPES = {"exact_duplicate_archive_copy", "regenerable_cache"}
PROTECTED_OUTPUT = Path("outputs") / "phase1_20260901"
SOURCE_SUFFIXES = {".py", ".pyw", ".pyi"}


@dataclass
class CleanupResult:
    status: str
    candidate_type: str
    relative_path: str
    size_bytes: int
    reason: str


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def safe_path(root: Path, relative_path: str) -> Path | None:
    raw = Path(relative_path)
    if raw.is_absolute() or "J:" in relative_path.upper():
        return None
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    try:
        candidate.relative_to((root / PROTECTED_OUTPUT).resolve())
    except ValueError:
        pass
    else:
        return None
    if candidate.suffix.lower() in SOURCE_SUFFIXES:
        return None
    return candidate


def validate_row(row: dict[str, str], root: Path) -> tuple[Path | None, str]:
    candidate_type = (row.get("candidate_type") or "").strip()
    relative_path = (row.get("relative_path") or "").strip()
    path = safe_path(root, relative_path)
    if candidate_type not in ALLOWED_TYPES:
        return None, "candidate_type is not allowlisted"
    if path is None:
        return None, "path is outside permitted cleanup scope"
    if not path.is_file():
        return None, "candidate file does not exist"
    try:
        declared_size = int(row.get("size_bytes", ""))
    except ValueError:
        return None, "invalid size_bytes"
    declared_hash = (row.get("sha256") or "").strip().lower()
    actual_hash, actual_size = sha256_and_size(path)
    if actual_size != declared_size or actual_hash != declared_hash:
        return None, "candidate hash or size does not match manifest"

    if candidate_type == "exact_duplicate_archive_copy":
        try:
            path.relative_to(root / "archive")
        except ValueError:
            return None, "archive duplicate candidate is not under archive/"
        anchor_value = (row.get("retention_anchor") or "").strip()
        anchor = safe_path(root, anchor_value) if anchor_value else None
        if anchor is None or not anchor.is_file():
            return None, "retention_anchor is missing or outside permitted scope"
        anchor_hash, anchor_size = sha256_and_size(anchor)
        if anchor_size != actual_size or anchor_hash != actual_hash:
            return None, "retention_anchor hash or size does not match candidate"
    elif (row.get("retention_anchor") or "").strip():
        return None, "regenerable_cache must not specify retention_anchor"
    return path, "validated"


def iter_rows(manifest: Path) -> Iterable[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def cleanup(manifest: Path, root: Path, apply: bool = False) -> list[CleanupResult]:
    results: list[CleanupResult] = []
    touched_dirs: set[Path] = set()
    for row in iter_rows(manifest):
        candidate_type = (row.get("candidate_type") or "").strip()
        relative_path = (row.get("relative_path") or "").strip()
        try:
            size = int(row.get("size_bytes", "0"))
        except ValueError:
            size = 0
        path, reason = validate_row(row, root)
        if path is None:
            results.append(CleanupResult("skipped", candidate_type, relative_path, size, reason))
            continue
        if not apply:
            results.append(CleanupResult("dry-run", candidate_type, relative_path, size, "validated; no deletion requested"))
            continue
        try:
            path.unlink()
            touched_dirs.add(path.parent)
            results.append(CleanupResult("deleted", candidate_type, relative_path, size, "deleted after hash/size revalidation"))
        except OSError as exc:
            results.append(CleanupResult("failed", candidate_type, relative_path, size, f"delete failed: {exc}"))
    if apply:
        clean_empty_directories(root, touched_dirs)
    return results


def clean_empty_directories(root: Path, touched_dirs: set[Path]) -> None:
    archive_root = (root / "archive").resolve()
    protected = (root / PROTECTED_OUTPUT).resolve()
    root_resolved = root.resolve()
    candidates: set[Path] = set()
    for directory in touched_dirs:
        current = directory.resolve()
        while current != root_resolved:
            candidates.add(current)
            current = current.parent
    for directory in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        if not directory.is_dir() or directory == archive_root:
            continue
        if directory == protected:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def write_log(results: list[CleanupResult], output_dir: Path, apply: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cleanup_log.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["status", "candidate_type", "relative_path", "size_bytes", "reason"])
        writer.writerows((item.status, item.candidate_type, item.relative_path, item.size_bytes, item.reason) for item in results)
    counts = {status: sum(item.status == status for item in results) for status in ("deleted", "dry-run", "skipped", "failed")}
    bytes_total = sum(item.size_bytes for item in results if item.status in {"deleted", "dry-run"})
    mode = "APPLY" if apply else "DRY-RUN"
    (output_dir / "cleanup_log.md").write_text(
        "\n".join([
            "# Phase 1 Cleanup Log",
            "",
            f"Mode: **{mode}**. Deletion is allowlisted and hash/size revalidated per row.",
            "",
            f"- Planned/deleted files: {counts['deleted'] + counts['dry-run']}",
            f"- Planned/deleted bytes: {bytes_total}",
            f"- Skipped: {counts['skipped']}",
            f"- Failed: {counts['failed']}",
            "- Protected: `outputs/phase1_20260901/`, source files, non-candidates, and external paths.",
        ]) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="actually delete validated candidates")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = (args.manifest or root / "outputs" / "phase1_20260901" / "deletion_candidates.csv").resolve()
    output_dir = (args.output_dir or root / "outputs" / "phase1_20260901").resolve()
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}")
    results = cleanup(manifest, root, apply=args.apply)
    write_log(results, output_dir, args.apply)
    planned = sum(item.size_bytes for item in results if item.status in {"dry-run", "deleted"})
    print(f"mode={'apply' if args.apply else 'dry-run'} planned_bytes={planned} planned_files={sum(item.status in {'dry-run', 'deleted'} for item in results)} skipped={sum(item.status == 'skipped' for item in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
