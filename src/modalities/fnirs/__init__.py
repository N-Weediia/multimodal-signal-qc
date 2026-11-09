"""fNIRS modality processing: NirSmart dual-wavelength optical density, QC, and MBLL sensitivity."""

from __future__ import annotations

from src.modalities.fnirs.qc import (
    FNIRSQualityMetrics,
    FNIRSSeries,
    assess_fnirs_quality,
    scalp_coupling_index,
    identify_known_ir_eye_interference,
)

__all__ = [
    "FNIRSQualityMetrics",
    "FNIRSSeries",
    "assess_fnirs_quality",
    "scalp_coupling_index",
    "identify_known_ir_eye_interference",
]
