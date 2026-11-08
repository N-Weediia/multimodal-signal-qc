from __future__ import annotations

from pathlib import Path

import expand_all_cases
import explore_202608_cases


def test_expand_data_root_updates_identity_path(monkeypatch, tmp_path: Path) -> None:
    original_root = expand_all_cases.RAW_ROOT
    original_identity = expand_all_cases.IDENTITY_XLSX
    monkeypatch.setattr(expand_all_cases, "RAW_ROOT", original_root)
    monkeypatch.setattr(expand_all_cases, "IDENTITY_XLSX", original_identity)

    expand_all_cases.set_data_root(tmp_path)

    assert expand_all_cases.RAW_ROOT == tmp_path.resolve()
    assert expand_all_cases.IDENTITY_XLSX == tmp_path.resolve() / "编号姓名数据核对表.xlsx"


def test_historical_default_remains_explicit() -> None:
    assert str(expand_all_cases.DEFAULT_OUT).endswith("outputs\\analysis\\20260901_all_cases") or str(expand_all_cases.DEFAULT_OUT).endswith("outputs/analysis/20260901_all_cases")


def test_explore_data_root_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(explore_202608_cases, "RAW_ROOT", explore_202608_cases.RAW_ROOT)
    explore_202608_cases.set_data_root(tmp_path)
    assert explore_202608_cases.RAW_ROOT == tmp_path.resolve()


def test_canonical_sessions_exclude_sec_when_sec_test_exists() -> None:
    sessions, excluded = expand_all_cases.canonicalize_sessions({
        "2026081403": ["base_line", "sec", "sec_test"],
        "2026081501": ["base_line", "sec_test"],
    })
    assert sessions["2026081403"] == ["base_line", "sec_test"]
    assert sessions["2026081501"] == ["base_line", "sec_test"]
    assert excluded == [{"subject": "2026081403", "session": "sec", "canonical_session": "sec_test", "reason": "duplicate follow-up; sec_test retained"}]


def test_explore_canonical_sessions_exclude_sec_when_sec_test_exists() -> None:
    sessions, excluded = explore_202608_cases.canonicalize_sessions({"2026081403": ["sec", "sec_test"]})
    assert sessions == {"2026081403": ["sec_test"]}
    assert excluded[0]["canonical_session"] == "sec_test"
