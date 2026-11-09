import json
from pathlib import Path

pre_root = Path('outputs/preprocessed/current')
index_doc = json.loads((pre_root / 'preprocessing_index.json').read_text(encoding='utf-8'))
emg_recs = [r for r in index_doc['records'] if r['modality'] == 'EMG' and r.get('is_primary_for_task')]

print(f"Total primary EMG records: {len(emg_recs)}")
counts_dist = {}
for r in emg_recs:
    sum_path = Path(r['output_directory']) / 'preprocessing_summary.json'
    s = json.loads(sum_path.read_text(encoding='utf-8'))
    st_map = s.get('channel_quality_status', {})
    good_cnt = sum(v in ('GOOD', 'CAUTION') for v in st_map.values())
    tot_cnt = len(st_map)
    pair = f"{good_cnt}/{tot_cnt}"
    counts_dist[pair] = counts_dist.get(pair, 0) + 1
    if pair != "7/7":
        print(f"  Non-7/7: {r['subject']} {r.get('session',''):8s}: {pair} deg={s.get('quality_degradation')} caution={s.get('caution_channel_count')} poor={s.get('poor_channel_count')}")

print("Distribution of good/tot channels:", counts_dist)
