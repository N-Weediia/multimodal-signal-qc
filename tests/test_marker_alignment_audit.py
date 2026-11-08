from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from scripts.build_marker_alignment_audit import build_audit


def test_marker_alignment_gate_uses_p95_not_marker_count() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "markers.csv"
        pd.DataFrame(
            {
                "subject": ["P01"] * 3,
                "session": ["base_line"] * 3,
                "task_name": ["摸头"] * 3,
                "secondary_marker_disagreement_ms": [10, 200, 1200],
            }
        ).to_csv(source, index=False)
        result = build_audit(source, root / "audit.csv")
    assert result.loc[0, "alignment_gate"] == "FAIL_SEVERE"
    assert result.loc[0, "n_markers"] == 3
    assert result.loc[0, "within_250ms_fraction"] == 2 / 3
