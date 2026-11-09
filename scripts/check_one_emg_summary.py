import json
from pathlib import Path

p = Path("outputs/preprocessed/current/2026081403/sec_test/EMG/运动任务/preprocessing_summary.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    print("channel_quality_status:", d.get("channel_quality_status"))
    print("channel_scores:", d.get("channel_scores"))
    print("caution_channel_count:", d.get("caution_channel_count"))
    print("poor_channel_count:", d.get("poor_channel_count"))
    print("keys in summary:", list(d.keys()))
