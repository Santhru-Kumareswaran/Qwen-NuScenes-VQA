
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.interpolate import interp1d

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg

INPUT_CSV = os.path.join(cfg.output_dir, 'nuscenes_action_tokens.csv')
TEMPLATE_FILE = os.path.join(cfg.output_dir, 'action_token_templates.json')
OUTPUT_PLOT = os.path.join(cfg.plots_dir, 'global_clusters_scatter.png')
RAW_CSV = cfg.input_file

# Helper for rotation (Upwards)
def rotate(pts):
    return np.stack([-pts[..., 1], pts[..., 0]], axis=-1)

def parse_waypoints(waypoints_str):
    try:
        return np.array(json.loads(waypoints_str))
    except:
        return np.zeros((0, 2))

def transform_to_path_frame(waypoints):
    if len(waypoints) < 2: return waypoints
    start_pos = waypoints[0]
    centered_waypoints = waypoints - start_pos
    
    start_yaw = 0.0
    found_valid_start = False
    for i in range(1, len(centered_waypoints)):
        dist = np.linalg.norm(centered_waypoints[i])
        if dist > 0.5: 
            start_yaw = np.arctan2(centered_waypoints[i, 1], centered_waypoints[i, 0])
            found_valid_start = True
            break
    if not found_valid_start:
         start_yaw = np.arctan2(centered_waypoints[-1, 1], centered_waypoints[-1, 0])
    
    c, s = np.cos(-start_yaw), np.sin(-start_yaw)
    R = np.array([[c, -s], [s, c]])
    return centered_waypoints @ R.T

def resample_trajectory(points, num_points):
    if len(points) < 2: return np.zeros((num_points, 2))
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    if total_len < 1e-3: return np.tile(points[0], (num_points, 1))
    new_dists = np.linspace(0, total_len, num_points)
    fx = interp1d(cum_dists, points[:, 0], kind='linear')
    fy = interp1d(cum_dists, points[:, 1], kind='linear')
    return np.column_stack((fx(new_dists), fy(new_dists)))


def main():
    print("Loading data...")
    # Since nuscenes_action_tokens.csv now contains ALL columns, we don't need to merge!
    df = pd.read_csv(INPUT_CSV)
    
    # Check if waypoints exists (it should)
    if 'waypoints' not in df.columns:
        print("Error: 'waypoints' column missing. Attempting legacy merge...")
        raw_df = pd.read_csv(RAW_CSV)
        df = pd.merge(df, raw_df[['sample_token', 'waypoints']], on='sample_token', how='left')
    
    merged = df # Alias for compatibility
    
    plt.figure(figsize=(12, 12))
    
    # Colors
    colors = {
        'LEFT': 'red',
        'RIGHT': 'blue',
        'STRAIGHT': 'green',
        'U_TURN': 'purple',
        'STATIONARY': 'gray',
        'CREEPING': 'orange'
    }
    
    # Get unique tokens (filter valid strings)
    tokens = [t for t in merged['action_token'].unique() if isinstance(t, str)]
    
    print("Plotting clusters...")
    for token in tokens:
        # Determine base color
        color = 'black'
        for k, c in colors.items():
            if token.startswith(k):
                color = c
                break
                
        # Filter data
        subset = merged[merged['action_token'] == token]
        
        # Downsample for plotting (don't overplot)
        if len(subset) > 50:
            subset = subset.sample(50, random_state=42)
            
        features = []
        for _, row in subset.iterrows():
            wp = parse_waypoints(row['waypoints'])
            if len(wp) < 2: continue
            norm = transform_to_path_frame(wp)
            res = resample_trajectory(norm, 10)
            
            # Rotate for plotting
            res_rot = rotate(res)
            
            plt.plot(res_rot[:, 0], res_rot[:, 1], color=color, alpha=0.1, linewidth=1)
            
    # Overlay centroids (Templates)
    print("Overlaying templates...")
    with open(TEMPLATE_FILE, 'r') as f:
        templates = json.load(f)
        
    for name, points in templates.items():
        color = 'black'
        for k, c in colors.items():
            if name.startswith(k):
                color = c
                break
                
        pts = np.array(points)
        pts_rot = rotate(pts)
        
        plt.plot(pts_rot[:, 0], pts_rot[:, 1], color='black', linewidth=3, linestyle='--') # Shadow
        plt.plot(pts_rot[:, 0], pts_rot[:, 1], color=color, linewidth=2, label=name)
        
        # Arrow
        plt.arrow(pts_rot[-2, 0], pts_rot[-2, 1], 
                  pts_rot[-1, 0] - pts_rot[-2, 0], 
                  pts_rot[-1, 1] - pts_rot[-2, 1], 
                  head_width=1.5, color=color)

    plt.title("Global Cluster Scatter (All Actions)")
    plt.xlabel("Lateral (m)")
    plt.ylabel("Longitudinal (m) [Up]")
    plt.axis('equal')
    plt.grid(True)
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors.values()]
    plt.legend(custom_lines, colors.keys(), loc='upper right')

    plt.savefig(OUTPUT_PLOT)
    print(f"Saved {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
