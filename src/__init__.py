"""EEF Multimodal Brain-Computer & Motor Rehabilitation System.

Modules:
- `src.modalities`: Low-level signal processing and feature extraction divided strictly by modality (EEG, EMG, fNIRS, Multimodal).
- `src.stages`: Execution pipelines and workflows divided strictly by project stage (Stage 0 to Stage 6).
- `src.utils`: Common configurations, safety assertions, and visualization helpers.
"""

from __future__ import annotations

from src.version import (
    PREPROCESS_PIPELINE_VERSION,
    QC_PIPELINE_VERSION,
    PIPELINE_VERSION_HISTORY,
    __version__,
)

__all__ = [
    "__version__",
    "PREPROCESS_PIPELINE_VERSION",
    "QC_PIPELINE_VERSION",
    "PIPELINE_VERSION_HISTORY",
]
