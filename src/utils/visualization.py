"""Matplotlib formatting, styling, and Chinese font configuration."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


def configure_matplotlib_font() -> None:
    """Configure matplotlib with standard fonts supporting Chinese characters."""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# Standard color palette for modalities
MODALITY_COLORS = {
    "EEG": "#1f77b4",       # Blue
    "EMG": "#2ca02c",       # Green
    "FNIRS": "#d62728",     # Red
    "Multimodal": "#9467bd",# Purple
}
