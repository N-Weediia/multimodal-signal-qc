"""Protocol-aware subspace shrinkage (PASS) and spatial artifact recovery."""

from __future__ import annotations

from typing import Any
import numpy as np

from src.stages.stage2_preprocess.pipeline import (
    protocol_aware_subspace_shrinkage,
)

__all__ = ["protocol_aware_subspace_shrinkage"]
