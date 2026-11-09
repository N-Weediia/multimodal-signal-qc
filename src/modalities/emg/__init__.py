"""sEMG & IMU modality processing: Delsys Avanti 7-channel logic mapping, QC, and kinematics."""

from __future__ import annotations

from src.modalities.emg.qc import (
    EMGQualityMetrics,
    EMGSeries,
    STANDARD_MUSCLE_MAPPING,
    MAPPING_SHIFTED_V1,
    assess_emg_quality,
)

__all__ = [
    "EMGQualityMetrics",
    "EMGSeries",
    "STANDARD_MUSCLE_MAPPING",
    "MAPPING_SHIFTED_V1",
    "assess_emg_quality",
]
