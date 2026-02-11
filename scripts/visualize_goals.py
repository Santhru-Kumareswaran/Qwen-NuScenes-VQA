
import csv
import cv2
import numpy as np
import ast
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import matplotlib.pyplot as plt

# Config
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "nuscenes_goal_tokens.csv"
OUTPUT_DIR = "verification_images"
NUM_SAMPLES_TO_VISUALIZE = 10

def render_sample(nusc, nusc_map, row, save_path):
    ego_x = float(row['ego_x'])
    ego_y = float(row['ego_y'])
    goal_name = row['goal_name']
    waypoints = ast.literal_eval(row['waypoints']) # list of [x, y]
    
    # 1. Get Map Mask/Image using nusc_map
    # We'll use the explorer to render a patch
    # But NuScenesMap explorer renders to matplotlib axis.
    

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Manual Map Rendering (Lanes only)
    # Find lanes near ego
    records = nusc_map.get_records_in_radius(ego_x, ego_y, radius=50, layer_names=['lane', 'lane_connector'])
    lane_tokens = records.get('lane', []) + records.get('lane_connector', [])
    
    # Discretize and plot
    if lane_tokens:
        try:
             # discretize_lanes returns dict {token: list of points}
             # resolution 1.0m
             lanes_geom = nusc_map.discretize_lanes(lane_tokens, 1.0)
             for token, points in lanes_geom.items():
                 pts = np.array(points)[:, :2]
                 ax.plot(pts[:, 0], pts[:, 1], 'b-', alpha=0.3, linewidth=1)
        except Exception as e:
             print(f"Error plotting lanes: {e}")
    
    # Plot Ego
    ax.plot(ego_x, ego_y, 'ro', markersize=10, label='Ego')
    
    # Plot Waypoints
    if waypoints:
        wps = np.array(waypoints)
        ax.plot(wps[:, 0], wps[:, 1], 'g.-', markersize=5, linewidth=2, label='Future Path')
        
    ax.set_title(f"Goal: {goal_name}")
    ax.legend()
    
    # Save
    plt.savefig(save_path)
    plt.close(fig)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)
    
    # Read CSV
    print(f"Reading {CSV_PATH}...")
    samples = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
            
    # Pick diverse samples if possible, or just random/strided
    indices = np.linspace(0, len(samples)-1, NUM_SAMPLES_TO_VISUALIZE, dtype=int)
    
    # Cache maps
    maps = {}

    print(f"Rendering {len(indices)} samples to {OUTPUT_DIR}...")
    for i, idx in enumerate(indices):
        row = samples[idx]
        scene_token = row['scene_token']
        
        # Get Map Name
        # We can look up scene in nusc
        scene = nusc.get('scene', scene_token)
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        
        if map_name not in maps:
            print(f"Loading map: {map_name}")
            maps[map_name] = NuScenesMap(dataroot=NUSCENES_ROOT, map_name=map_name)
            
        scene_map = maps[map_name]
        
        save_path = os.path.join(OUTPUT_DIR, f"vis_{i:03d}_{row['goal_name']}.png")
        render_sample(nusc, scene_map, row, save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
