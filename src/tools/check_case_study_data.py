import numpy as np
import json
from pathlib import Path

# 1. EEG 1201
eeg_p = Path("outputs/preprocessed/current/2026081201/base_line/EEG/运动/eeg_cleaned.npz")
npz_eeg = np.load(eeg_p)
sr_eeg = float(npz_eeg["sample_rate_hz"])
data_eeg = npz_eeg["data_uv"]
labels_eeg = [str(lbl) for lbl in npz_eeg["labels"]]
cz_idx = labels_eeg.index("Cz") if "Cz" in labels_eeg else 0
print(f"EEG: sr={sr_eeg}, shape={data_eeg.shape}, channel={labels_eeg[cz_idx]}, duration={data_eeg.shape[1]/sr_eeg:.1f}s")

# 2. EMG 1403
emg_p = Path("outputs/preprocessed/current/2026081403/sec_test/EMG/陈桂珍_Plot_and_Store_Rep_1.1/emg_cleaned.npz")
npz_emg = np.load(emg_p)
sr_emg = float(npz_emg["sample_rate_hz"])
raw_emg = npz_emg["data_raw_source_unit"]
clean_emg = npz_emg["data_source_unit"]
env_emg = npz_emg["envelope_5hz_source_unit"]
sids = npz_emg["sensor_ids"].tolist()
s10_idx = sids.index(10) if 10 in sids else 0
print(f"EMG: sr={sr_emg}, shape={clean_emg.shape}, s10_idx={s10_idx}, duration={clean_emg.shape[1]/sr_emg:.1f}s")

# 3. fNIRS 1801
fnirs_p = Path(r"outputs/preprocessed/current/2026081801/base_line/FNIRS/运动/运动任务/通用_2024-02-15_16-54-56_2026081801_乐明祖_男_1950-01-01_静息态/fnirs_cleaned_od.npz")
npz_nirs = np.load(fnirs_p)
sr_nirs = float(npz_nirs["sample_rate_hz"])
od_nirs = npz_nirs["optical_density"]
raw_nirs = npz_nirs["raw_intensity"]
t_nirs = npz_nirs["time_seconds"]
print(f"fNIRS: sr={sr_nirs:.2f}, OD shape={od_nirs.shape}, raw shape={raw_nirs.shape}, duration={t_nirs[-1]:.1f}s")

# 4. Markers
fnirs_sum = json.loads(fnirs_p.with_name("preprocessing_summary.json").read_text(encoding="utf-8"))
print("fNIRS summary marker_alignment:", fnirs_sum.get("marker_alignment"))
