"""Run the public QC core on synthetic sEMG and fNIRS channels."""

import numpy as np

from emg_qc import EMGSeries, assess_emg_channel
from fnirs_qc import FNIRSChannelSeries, assess_fnirs_channel


def main() -> None:
    sample_rate = 1_000.0
    time = np.arange(10_000) / sample_rate
    rng = np.random.default_rng(42)
    emg = 2e-5 * np.sin(2 * np.pi * 83 * time) + rng.normal(0, 2e-6, time.size)
    emg_metrics = assess_emg_channel(
        EMGSeries(2, "synthetic_emg", "demo", time, emg, sample_rate)
    )

    fnirs_time = np.arange(660) / 11.0
    cardiac = np.sin(2 * np.pi * 1.2 * fnirs_time)
    intensity_a = 200 * (1 + 0.01 * cardiac) + rng.normal(0, 0.1, fnirs_time.size)
    intensity_b = 180 * (1 + 0.012 * cardiac) + rng.normal(0, 0.1, fnirs_time.size)
    fnirs_metrics = assess_fnirs_channel(
        FNIRSChannelSeries(
            1, 1, 1, "synthetic_fnirs", 730, 850,
            intensity_a, intensity_b, (0.0, 0.0)
        ),
        sample_rate=11.0,
    )

    print("sEMG:", emg_metrics.status, f"score={emg_metrics.quality_score:.3f}")
    print("fNIRS:", fnirs_metrics.status, f"score={fnirs_metrics.quality_score:.3f}")


if __name__ == "__main__":
    main()
