"""fNIRS Modality Preprocessing Algorithms.

Includes:
- Conversion to optical density (natural log OD)
- Fishburn TDDR step and motion artifact repair
- Bandpass filtering (0.01 - 0.2 Hz)
- Temporal alignment of fNIRS markers to EEG clock
- Primary fNIRS preprocessing entrypoint
"""

from __future__ import annotations

from typing import Any
import numpy as np

from src.stages.stage2_preprocess.pipeline import (
    fishburn_tddr,
    align_fnirs_markers_with_eeg,
    preprocess_fnirs,
)

__all__ = [
    "fishburn_tddr",
    "align_fnirs_markers_with_eeg",
    "preprocess_fnirs",
]
