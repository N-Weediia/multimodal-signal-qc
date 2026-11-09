"""EEG Modality Preprocessing Algorithms.

Includes:
- Robust scalp referencing (median)
- Bad channel detection via correlation and variance
- Electrolyte gel bridge detection
- Epoch quality masking
- Protocol-aware subspace shrinkage (PASS)
- Primary EEG preprocessing entrypoint
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import numpy as np

# Facade delegates to Stage 2 preprocessing pipeline for consistency
from src.stages.stage2_preprocess.pipeline import (
    eeg_bad_channels,
    detect_eeg_gel_bridges,
    eeg_valid_epoch_mask,
    eeg_epoch_quality,
    protocol_aware_subspace_shrinkage,
    preprocess_eeg,
)

__all__ = [
    "eeg_bad_channels",
    "detect_eeg_gel_bridges",
    "eeg_valid_epoch_mask",
    "eeg_epoch_quality",
    "protocol_aware_subspace_shrinkage",
    "preprocess_eeg",
]
