"""Summarise EEG/fNIRS-derived EMG marker alignment quality.

The current acquisition has no independent EMG trigger stream.  Therefore this
audit does not manufacture a hardware synchronization claim: it reports the
first-marker/frame-0 convention and the disagreement between the two observed
marker streams, leaving the final alignment gate explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_audit(marker_csv: Path, output_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(marker_csv) if marker_csv.exists() else pd.DataFrame()
    if table.empty:
        audit = pd.DataFrame(columns=["subject", "session", "task_name", "n_markers", "median_disagreement_ms", "p95_disagreement_ms", "within_250ms_fraction", "over_1s_fraction", "alignment_gate", "alignment_method"])
    else:
        table["subject"] = table["subject"].astype(str)
        table["session"] = table["session"].astype(str)
        table["task_name"] = table["task_name"].astype(str)
        table["disagreement_ms"] = pd.to_numeric(table.get("secondary_marker_disagreement_ms"), errors="coerce")
        rows = []
        for (subject, session, task), group in table.groupby(["subject", "session", "task_name"], sort=True):
            values = group["disagreement_ms"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                gate = "UNKNOWN_NO_SECONDARY_STREAM"
                median = p95 = within = severe = np.nan
            else:
                median = float(np.median(values))
                p95 = float(np.percentile(values, 95))
                within = float(np.mean(values <= 250.0))
                severe = float(np.mean(values > 1000.0))
                gate = "PASS" if p95 <= 250.0 else "CAUTION" if p95 <= 1000.0 else "FAIL_SEVERE"
            rows.append({
                "subject": subject,
                "session": session,
                "task_name": task,
                "n_markers": int(len(values)),
                "median_disagreement_ms": median,
                "p95_disagreement_ms": p95,
                "within_250ms_fraction": within,
                "over_1s_fraction": severe,
                "alignment_gate": gate,
                "alignment_method": "first observed marker -> emg_time[0]; EEG/fNIRS secondary disagreement audit",
            })
        audit = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an explicit marker alignment audit table.")
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.markers, args.output)
    print(f"rows={len(audit)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
