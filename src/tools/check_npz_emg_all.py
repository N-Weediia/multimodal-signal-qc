import numpy as np
import json
from pathlib import Path

pre_root = Path("outputs/preprocessed/current")
index_doc = json.loads((pre_root / "preprocessing_index.json").read_text(encoding="utf-8"))
emg_recs = [r for r in index_doc["records"] if r["modality"] == "EMG" and r.get("is_primary_for_task")]

total = len(emg_recs)
all_7_count = 0
for r in emg_recs:
    npz_path = Path(r["output_directory"]) / "emg_cleaned.npz"
    if not npz_path.exists():
        continue
    npz = np.load(npz_path, allow_pickle=False)
    gates = npz["channel_signal_gate"]
    status = npz["channel_quality_status"]
    scores = npz["channel_quality_score"]
    true_cnt = int(np.sum(gates))
    tot_cnt = len(gates)
    if true_cnt == tot_cnt and tot_cnt == 7:
        all_7_count += 1
    else:
        print(f"Non-all-7: {r['subject']} {r.get('session',''):8s}: {true_cnt}/{tot_cnt} status={status.tolist()}")

print(f"Total: {total}, all 7 available: {all_7_count}/{total}")
