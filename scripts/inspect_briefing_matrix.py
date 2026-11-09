import json
from pathlib import Path

p = Path("outputs/preprocessed/current/figures/cohort_quality_briefing_data.json")
d = json.loads(p.read_text(encoding="utf-8"))

targets = ["2026081201", "2026081403", "2026081701", "2026081501", "2026081801", "2026082102", "2026080901"]
for row in d["matrix_rows"]:
    if row["subject"] in targets:
        print(f"=== Subject: {row['subject']} ({row['name']}) ready={row['ready_slots']}/{row['expected_slots']} ===")
        for k, v in row["cells"].items():
            if v["state"] != "UNACQUIRED":
                print(f"  {k}: text={v['text']}, state={v['state']}, score={v['score']}")
