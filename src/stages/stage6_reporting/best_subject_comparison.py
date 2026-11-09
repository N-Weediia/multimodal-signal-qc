"""Run the existing best-visit visual audit against the 2026-09-14 outputs."""
from pathlib import Path

import build_best_subject_preprocessing_comparison_20260910 as report

ROOT = Path(__file__).resolve().parent
report.RAW = ROOT / "data_repro_20260903"
report.STAGE = ROOT / "review_20260914"
report.OUT = ROOT / "best_subject_preprocessing_comparison_20260914"

if __name__ == "__main__":
    report.main()
