import json
from pathlib import Path
import numpy as np

pre_root = Path("outputs/preprocessed/current")
index_doc = json.loads((pre_root / "preprocessing_index.json").read_text(encoding="utf-8"))
emg_recs = [r for r in index_doc["records"] if r["modality"] == "EMG"]

updated_count = 0
for r in emg_recs:
    out_dir = Path(r["output_directory"])
    sum_path = out_dir / "preprocessing_summary.json"
    npz_path = out_dir / "emg_cleaned.npz"
    if not (sum_path.is_file() and npz_path.is_file()):
        continue
    
    npz = np.load(npz_path, allow_pickle=False)
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    
    sensor_ids = [str(sid) for sid in npz["sensor_ids"]]
    quality_status = [str(st) for st in npz["channel_quality_status"]]
    quality_scores = [round(float(sc), 3) for sc in npz["channel_quality_score"]]
    
    status_dict = dict(zip(sensor_ids, quality_status))
    score_dict = dict(zip(sensor_ids, quality_scores))
    
    summary["channel_quality_status"] = status_dict
    summary["channel_quality_scores"] = score_dict
    summary["caution_channel_count"] = sum(st in ("CAUTION", "CAUTION_HIGH_RAW_LINE") for st in quality_status)
    summary["poor_channel_count"] = sum(st == "POOR" for st in quality_status)
    summary["good_channel_count"] = sum(st == "GOOD" for st in quality_status)
    
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    updated_count += 1

print(f"Synchronized {updated_count} EMG summaries with emg_cleaned.npz")
