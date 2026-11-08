from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase1_audit", ROOT / "scripts" / "phase1_audit.py")
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_hash_inventory_reports_duplicate_files_without_touching_excluded_tree(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a.txt").write_text("same", encoding="utf-8")
    (outputs / "b.txt").write_text("same", encoding="utf-8")
    excluded = outputs / "phase1_20260901"
    excluded.mkdir()
    (excluded / "audit.json").write_text("same", encoding="utf-8")

    result = AUDIT.hash_inventory(tmp_path, excluded)

    assert result["hashed_file_count"] == 2
    assert result["duplicate_group_count"] == 1
    assert sorted(result["duplicate_groups"][0]) == ["outputs/a.txt", "outputs/b.txt"]


def test_dependency_status_maps_scikit_learn_distribution_to_sklearn_module(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("scikit-learn>=1.3\n", encoding="utf-8")

    result = AUDIT.dependency_status(requirements)

    assert result == [{"package": "scikit-learn", "module": "sklearn", "available": True, "error": None}]


def test_build_audit_marks_rebuild_blocked_when_local_data_is_empty(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs" / "legacy.txt").write_text("historical result", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy>=1.24\n", encoding="utf-8")
    for name in AUDIT.DEFAULT_ENTRYPOINTS[:3]:
        (tmp_path / name).write_text("# placeholder\n", encoding="utf-8")

    result = AUDIT.build_audit(tmp_path, tmp_path / "outputs" / "phase1_20260901", run_tests=False)

    assert result["rebuild"]["status"] == "BLOCKED_NO_LOCAL_INPUT_DATA"
    assert result["rebuild"]["external_paths_were_not_opened"] is True
