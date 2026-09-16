import numpy as np

from emg_qc import EMGSeries, assess_emg_channel
from fnirs_qc import FNIRSChannelSeries, assess_fnirs_channel


def test_clean_emg_is_accepted():
    fs = 1000.0
    time = np.arange(10_000) / fs
    values = 2e-5 * np.sin(2 * np.pi * 83 * time)
    metrics = assess_emg_channel(EMGSeries(2, "demo", "demo", time, values, fs))
    assert metrics.status == "GOOD"
    assert metrics.missing_sample_percent == 0.0


def test_coupled_fnirs_wavelengths_have_high_sci():
    fs = 11.0
    time = np.arange(660) / fs
    cardiac = np.sin(2 * np.pi * 1.2 * time)
    a = 200 * (1 + 0.01 * cardiac)
    b = 180 * (1 + 0.012 * cardiac)
    channel = FNIRSChannelSeries(1, 1, 1, "demo", 730, 850, a, b, (0.0, 0.0))
    metrics = assess_fnirs_channel(channel, fs)
    assert metrics.scalp_coupling_index > 0.95
    assert metrics.status in {"GOOD", "CAUTION"}
