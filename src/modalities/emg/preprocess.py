"""sEMG Modality Preprocessing Algorithms.

Includes:
- Hann weighted overlap-add (WOLA) power-line harmonic cancellation
- Bandpass filtering (20-450 Hz)
- 5 Hz RMS / Hilbert envelope calculation
- Standard muscle channel sorting and validation
"""

from __future__ import annotations

from typing import Any
import numpy as np

from src.stages.stage2_preprocess.pipeline import (
    overlap_add_harmonic_cancel,
    preprocess_emg,
)

__all__ = [
    "overlap_add_harmonic_cancel",
    "preprocess_emg",
]
