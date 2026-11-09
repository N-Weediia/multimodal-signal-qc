"""Read-only phase-one audit for dependencies, outputs, hashes, and tests.

The audit only reads the repository and its local ``data``/``outputs`` trees.  It
never follows or opens external-drive paths found in historical reports.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


EXTERNAL_PATH_PATTERN = re.compile(r"(?i)(?<![\w])([a-z]:\\[^\r\n,;\"']+)")
DEFAULT_ENTRYPOINTS = (
    "main.py",
    "preprocess.py",
    "low_snr_recovery.py",
    "integrate_tcm_reports.py",
    "expand_all_cases.py",
    "explore_202608_cases.py",
    "summarize_expanded_results.py",
)


def parse_requirements(path: Path) -> list[str]:
    packages: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        packages.append(re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip())
    return packages


def dependency_status(requirements: Path) -> list[dict[str, Any]]:
    statuses = []
    for package in parse_requirements(requirements):
        module = {
            "numpy": "numpy",
            "scipy": "scipy",
            "matplotlib": "matplotlib",
            "scikit-learn": "sklearn",
        }.get(
            package.lower(), package.replace("-", "_")
        )
        try:
            available = importlib.util.find_spec(module) is not None
            error = None
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            available = False
            error = str(exc)
        statuses.append({"package": package, "module": module, "available": available, "error": error})
    return statuses


def imported_top_level_modules(root: Path) -> set[str]:
    modules: set[str] = set()
    source_files = sorted(root.glob("*.py")) + sorted((root / "tests").glob("*.py"))
    for path in source_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".", 1)[0])
    return modules


def source_audit(root: Path) -> dict[str, Any]:
    source_files = sorted(root.glob("*.py")) + sorted((root / "tests").glob("*.py"))
    parse_errors: list[dict[str, str]] = []
    for path in source_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append({"path": str(path.relative_to(root)), "error": str(exc)})
    entrypoints = []
    for name in DEFAULT_ENTRYPOINTS:
        path = root / name
        entrypoints.append({"path": name, "exists": path.is_file()})
    local_modules = {path.stem for path in root.glob("*.py")}
    stdlib_modules = set(getattr(sys, "stdlib_module_names", ())) or {
        "argparse",
        "ast",
        "collections",
        "csv",
        "dataclasses",
        "datetime",
        "fractions",
        "hashlib",
        "importlib",
        "json",
        "math",
        "pathlib",
        "re",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
        "typing",
        "unittest",
    }
    imported_modules = imported_top_level_modules(root)
    external_modules = sorted(imported_modules - local_modules - stdlib_modules)
    return {
        "source_file_count": len(source_files),
        "parse_errors": parse_errors,
        "entrypoints": entrypoints,
        "imported_external_modules": external_modules,
    }


def output_inventory(root: Path, excluded_dir: Path) -> dict[str, Any]:
    outputs = root / "outputs"
    files = [path for path in outputs.rglob("*") if path.is_file() and excluded_dir not in path.parents]
    total_bytes = sum(path.stat().st_size for path in files)
    external_refs: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() not in {".json", ".md", ".csv", ".txt", ".log"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        matches = EXTERNAL_PATH_PATTERN.findall(text)
        if matches:
            external_refs.append(
                {"path": path.relative_to(root).as_posix(), "count": len(matches), "examples": matches[:3]}
            )
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "external_path_references": external_refs,
    }


def hash_inventory(root: Path, excluded_dir: Path) -> dict[str, Any]:
    files = [path for path in (root / "outputs").rglob("*") if path.is_file() and excluded_dir not in path.parents]
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        by_hash[digest.hexdigest()].append(path.relative_to(root).as_posix())
    duplicate_groups = [paths for paths in by_hash.values() if len(paths) > 1]
    return {
        "hashed_file_count": len(files),
        "unique_hash_count": len(by_hash),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }


def test_entrypoint(root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return {"command": " ".join(command), "returncode": completed.returncode, "output_tail": output[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": " ".join(command), "returncode": None, "output_tail": str(exc)}


def build_audit(root: Path, output_dir: Path, run_tests: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = dependency_status(root / "requirements.txt")
    source = source_audit(root)
    declared_modules = {item["module"].split(".", 1)[0] for item in dependencies}
    undeclared = []
    for module in source["imported_external_modules"]:
        if module in declared_modules:
            continue
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        undeclared.append({"module": module, "available": available})
    source["undeclared_external_modules"] = undeclared
    outputs = output_inventory(root, output_dir)
    hashes = hash_inventory(root, output_dir)
    tests = test_entrypoint(root) if run_tests else {"command": "skipped", "returncode": None, "output_tail": ""}
    data_files = [path for path in (root / "data").rglob("*") if path.is_file()]
    entrypoints_ready = all(item["exists"] for item in source["entrypoints"][:3])
    has_historical_outputs = outputs["file_count"] > 0
    if not entrypoints_ready:
        rebuild_status = "BLOCKED_MISSING_ENTRYPOINT"
    elif has_historical_outputs and not data_files:
        rebuild_status = "BLOCKED_NO_LOCAL_INPUT_DATA"
    else:
        rebuild_status = "POSSIBLE_WITH_LOCAL_INPUTS"
    return {
        "audit": {"phase": "phase1", "date": "20260901", "read_only": True},
        "dependencies": dependencies,
        "source": source,
        "outputs": outputs,
        "hashes": hashes,
        "rebuild": {
            "status": rebuild_status,
            "local_data_file_count": len(data_files),
            "external_paths_were_not_opened": True,
        },
        "tests": tests,
    }


def write_reports(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "phase1_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    duplicate_groups = result["hashes"]["duplicate_groups"]
    external_refs = result["outputs"]["external_path_references"]
    dependency_line = ", ".join(
        f"{item['package']}={'OK' if item['available'] else 'MISSING'}" for item in result["dependencies"]
    )
    test_result = result["tests"]
    lines = [
        "# Phase 1 Audit (2026-09-01)",
        "",
        "只读审计；未删除、移动或覆盖任何输入/历史结果文件，也未打开外部盘符路径。",
        "",
        f"- Dependencies: {dependency_line or 'none declared'}",
        f"- Source parse errors: {len(result['source']['parse_errors'])}",
        f"- Undeclared external imports: {len(result['source']['undeclared_external_modules'])}",
        f"- Existing output files: {result['outputs']['file_count']} ({result['outputs']['total_bytes']} bytes)",
        f"- External path references in reports: {len(external_refs)} files (references were not accessed)",
        f"- SHA-256 duplicate groups: {len(duplicate_groups)} across {result['hashes']['hashed_file_count']} files",
        f"- Rebuild assessment: **{result['rebuild']['status']}**; local data files={result['rebuild']['local_data_file_count']}",
        f"- Test entry: `{test_result['command']}` (return code {test_result['returncode']})",
        "",
        "## Rebuild Boundary",
        "",
        "历史结果中若含 J:\\ 等外部来源，只作为文本证据登记；本审计不访问该路径。",
    ]
    if duplicate_groups:
        lines.extend(["", "## Duplicate Hash Examples", ""])
        for group in duplicate_groups[:20]:
            lines.append("- " + " | ".join(group))
    (output_dir / "phase1_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs" / "phase1_20260901").resolve()
    result = build_audit(root, output_dir, run_tests=not args.skip_tests)
    write_reports(result, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rebuild_status": result["rebuild"]["status"], "test_returncode": result["tests"]["returncode"]}, ensure_ascii=False))
    return 0 if not result["source"]["parse_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
