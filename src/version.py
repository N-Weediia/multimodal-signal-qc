"""EEF Multimodal Brain-Computer & Motor Rehabilitation System.

Version information and pipeline specifications.
"""

from __future__ import annotations

from typing import Any

# Global software / framework semantic version
__version__ = "8.0.0"

# Preprocessing pipeline version tag (used in NPZ / index.json)
PREPROCESS_PIPELINE_VERSION = "2026-11-09.v8-quality-tiered-recovery"

# QC pipeline version tag
QC_PIPELINE_VERSION = "2026-10-25.v7-full-cohort-qc"

# Full pipeline evolution history
PIPELINE_VERSION_HISTORY: dict[str, dict[str, Any]] = {
    "v5": {
        "tag": "2026-09-14.v5-initial-review",
        "date": "2026-09-14",
        "description": "Baseline preprocessing review state for 107 records across paired 7 subjects.",
        "distribution": {
            "READY_WITH_MASKS": 41,
            "READY_CHANNEL_FILTERED": 33,
            "READY_CONTINUOUS": 14,
            "READY_WITH_MASKS_KNOWN_IR_INTERFERENCE": 16,
            "FAILED": 3,
        },
        "artifacts_location": "outputs/preprocessed/v5_review_20260914",
    },
    "v6": {
        "tag": "2026-09-14.v6-quality-tiered-recovery",
        "date": "2026-09-14",
        "description": "Optimized preprocessing fixing 2026081401 baseline EEG.",
        "distribution": {
            "READY_WITH_MASKS": 42,
            "READY_CHANNEL_FILTERED": 33,
            "READY_CONTINUOUS": 14,
            "READY_WITH_MASKS_KNOWN_IR_INTERFERENCE": 16,
            "FAILED": 2,
        },
        "artifacts_location": "outputs/preprocessed/v6_optimized_20260914",
    },
    "v7": {
        "tag": "2026-10-25.v7-full-cohort-quality-provenance",
        "date": "2026-10-25",
        "description": "Full cohort quality provenance and matrix audit across 18 cases / 202608 cohort.",
        "artifacts_location": "outputs/preprocessed/v7_cohort_20261025",
    },
    "v8": {
        "tag": "2026-11-09.v8-quality-tiered-recovery",
        "date": "2026-11-09",
        "description": "Modular tiered recovery: Hann WOLA harmonic cancellation for EMG, Fishburn TDDR for fNIRS, protocol-aware subspace shrinkage, and enhanced EEG common-mode clamp resilience.",
        "artifacts_location": "outputs/preprocessed/current",
    },
}
