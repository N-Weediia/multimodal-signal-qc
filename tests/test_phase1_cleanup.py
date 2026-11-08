from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase1_cleanup", ROOT / "scripts" / "phase1_cleanup.py")
assert SPEC and SPEC.loader
CLEANUP = importlib.util.module_from_spec(SPEC)
sys.modules["phase1_cleanup"] = CLEANUP
SPEC.loader.exec_module(CLEANUP)


def _manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_type", "relative_path", "size_bytes", "sha256", "retention_anchor"])
        writer.writeheader()
        writer.writerows(rows)


def test_dry_run_validates_archive_copy_without_deleting(tmp_path: Path) -> None:
    anchor = tmp_path / "outputs" / "active.bin"
    candidate = tmp_path / "archive" / "outputs" / "old.bin"
    anchor.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    anchor.write_bytes(b"identical")
    candidate.write_bytes(b"identical")
    digest = hashlib.sha256(b"identical").hexdigest()
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [{"candidate_type": "exact_duplicate_archive_copy", "relative_path": "archive/outputs/old.bin", "size_bytes": "9", "sha256": digest, "retention_anchor": "outputs/active.bin"}])

    result = CLEANUP.cleanup(manifest, tmp_path, apply=False)

    assert result[0].status == "dry-run"
    assert candidate.exists()


def test_apply_removes_valid_copy_but_keeps_archive_root(tmp_path: Path) -> None:
    anchor = tmp_path / "outputs" / "active.bin"
    candidate = tmp_path / "archive" / "nested" / "old.bin"
    anchor.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    anchor.write_bytes(b"identical")
    candidate.write_bytes(b"identical")
    digest = hashlib.sha256(b"identical").hexdigest()
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [{"candidate_type": "exact_duplicate_archive_copy", "relative_path": "archive/nested/old.bin", "size_bytes": "9", "sha256": digest, "retention_anchor": "outputs/active.bin"}])

    result = CLEANUP.cleanup(manifest, tmp_path, apply=True)

    assert result[0].status == "deleted"
    assert not candidate.exists()
    assert (tmp_path / "archive").is_dir()


def test_apply_skips_hash_mismatch_and_disallowed_type(tmp_path: Path) -> None:
    candidate = tmp_path / "archive" / "copy.bin"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"actual")
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [
        {"candidate_type": "exact_duplicate_archive_copy", "relative_path": "archive/copy.bin", "size_bytes": "6", "sha256": "0" * 64, "retention_anchor": "archive/missing.bin"},
        {"candidate_type": "other", "relative_path": "archive/copy.bin", "size_bytes": "6", "sha256": hashlib.sha256(b"actual").hexdigest(), "retention_anchor": ""},
    ])

    result = CLEANUP.cleanup(manifest, tmp_path, apply=True)

    assert [item.status for item in result] == ["skipped", "skipped"]
    assert candidate.exists()


def test_protected_phase1_output_is_never_a_cleanup_target(tmp_path: Path) -> None:
    candidate = tmp_path / "outputs" / "phase1_20260901" / "keep.txt"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("keep", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [{"candidate_type": "regenerable_cache", "relative_path": "outputs/phase1_20260901/keep.txt", "size_bytes": "4", "sha256": hashlib.sha256(b"keep").hexdigest(), "retention_anchor": ""}])

    result = CLEANUP.cleanup(manifest, tmp_path, apply=True)

    assert result[0].status == "skipped"
    assert candidate.exists()
