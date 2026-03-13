import pandas as pd
import json
import numpy as np
from pathlib import Path

def transform_to_global(points_ego, ego_x, ego_y, ego_yaw):
    if len(points_ego) == 0:
        return points_ego
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    R = np.array([[c, -s], [s, c]])
    pts = (points_ego @ R.T).astype(np.float32)
    pts += np.array([ego_x, ego_y], dtype=np.float32)
    return pts.tolist()

csv_path = '/home/santhru/FYP38_First Experiment/NuScenesVQA-/QWEN_VL_AD/output/nuscenes_action_tokens_with_predictions.csv'
json_path = '/home/santhru/FYP38_First Experiment/NuScenesVQA-/action_token_templates.json'

print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Loading Templates: {json_path}")
with open(json_path, 'r') as f:
    templates = json.load(f)

def get_ref(row):
    tok = str(row.get('action_token', ''))
    if tok in templates:
        ref_ego = np.array(templates[tok], dtype=np.float32)
        ego_x = float(row['ego_x'])
        ego_y = float(row['ego_y'])
        ego_yaw = float(row.get('ego_yaw', 0.0))
        return json.dumps(transform_to_global(ref_ego, ego_x, ego_y, ego_yaw))
    return None

print("Computing reference trajectories...")
df['reference_trajectory'] = df.apply(get_ref, axis=1)

print(f"Saving updated CSV to: {csv_path}")
df.to_csv(csv_path, index=False)
print("Done!")
