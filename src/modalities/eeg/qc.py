"""EEG Modality Quality Control Algorithms.

Includes:
- Robust median re-referencing and DC offset removal
- Disconnect epoch detection and clamping
- 50 Hz power-line harmonic ratio assessment
- Channel variance and correlation screening
"""

from __future__ import annotations

from typing import Any
import numpy as np


def clamp_eeg_disconnect_epochs(
    raw: np.ndarray,
    digital: np.ndarray | None,
    digital_min: np.ndarray | None,
    digital_max: np.ndarray | None,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and clamp catastrophic amplifier disconnects or huge electrode offsets.

    Operates on 1-second contiguous epochs.
    Returns (cleaned_raw, disconnect_epoch_mask).
    """
    ep_len = int(sample_rate)
    n_ep = raw.shape[1] // ep_len
    if n_ep < 3:
        return raw.copy(), np.zeros(n_ep, dtype=bool)
    raw_ep = raw[:, : n_ep * ep_len].reshape(raw.shape[0], n_ep, ep_len)
    med_p2p_raw = np.median(np.ptp(raw_ep, axis=2), axis=0)

    # Remove per-channel within-epoch DC offset before sample-wise median reference
    raw_ep_centered = raw_ep - np.median(raw_ep, axis=2, keepdims=True)
    reref_ep = raw_ep_centered - np.median(raw_ep_centered, axis=0, keepdims=True)
    med_p2p_reref = np.median(np.ptp(reref_ep, axis=2), axis=0)

    if digital is not None and digital_min is not None and digital_max is not None:
        dig_ep = digital[:, : n_ep * ep_len].reshape(digital.shape[0], n_ep, ep_len)
        clipped_frac = np.mean(
            np.any((dig_ep <= digital_min[:, None, None]) | (dig_ep >= digital_max[:, None, None]), axis=2),
            axis=0,
        )
        disconnect = (clipped_frac >= 0.25) | (med_p2p_reref > 2500.0) | (med_p2p_raw > 50000.0)
    else:
        disconnect = (med_p2p_reref > 2500.0) | (med_p2p_raw > 50000.0)
    if not np.any(disconnect) or np.mean(disconnect) >= 0.50:
        return raw.copy(), np.zeros(n_ep, dtype=bool)
    dilated = np.convolve(disconnect.astype(int), [1, 1, 1], mode="same") > 0
    good_idx = np.where(~dilated)[0]
    cleaned = raw.copy()
    if good_idx.size > 0:
        fill_val = np.median(raw[:, : n_ep * ep_len].reshape(raw.shape[0], n_ep, ep_len)[:, good_idx, :], axis=(1, 2), keepdims=True)
    else:
        fill_val = np.zeros((raw.shape[0], 1, 1), dtype=raw.dtype)
    for ep_i, bad in enumerate(dilated):
        if bad:
            cleaned[:, ep_i * ep_len : (ep_i + 1) * ep_len] = fill_val[:, 0, :]
    return cleaned, dilated
