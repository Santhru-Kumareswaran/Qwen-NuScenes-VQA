
import csv
import os
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# ================= CONFIG =================
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "/home/santhru/FYP38_First Experiment/nuscenes_action_tokens.csv"
OUTPUT_DIR = "bev_frames_organized"
SCENES_PER_ACTION = 20
# ==========================================

def get_color(category_name):
    if 'vehicle' in category_name:
        return 'r'
    elif 'cycle' in category_name or 'bicycle' in category_name:
        return 'm'
    elif 'pedestrian' in category_name:
        return 'b'
    else:
        return 'k'

def render_bev_frame(nusc, row, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Load Data
    sample_token = row['sample_token']
    sample = nusc.get('sample', sample_token)
    sd_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sd_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # 2. LiDAR Point Cloud
    pcl_path = os.path.join(nusc.dataroot, sd_rec['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    
    # Transform to Ego
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    # Filter points
    mask = np.abs(pc.points[0, :]) < 50
    mask &= np.abs(pc.points[1, :]) < 50
    points = pc.points[:, mask]
    
    ax.scatter(points[0, :], points[1, :], s=0.1, c='gray', alpha=0.5)
    
    # 3. Annotations
    _, boxes, _ = nusc.get_sample_data(sd_token, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=True)
    
    for box in boxes:
        c = get_color(box.name)
        box.render(ax, view=np.eye(4), colors=(c, c, c))

    # 4. Waypoints
    waypoints = ast.literal_eval(row['waypoints'])
    if waypoints:
        wps = np.array(waypoints)
        ego_trans = np.array(pose_record['translation'])
        ego_rot = Quaternion(pose_record['rotation'])
        
        wps_3d = np.hstack([wps, np.zeros((len(wps), 1))])
        wps_centered_3d = wps_3d - ego_trans
        wps_ego = np.dot(wps_centered_3d, ego_rot.inverse.rotation_matrix.T)
        
        ax.plot(wps_ego[:, 0], wps_ego[:, 1], 'g-', linewidth=2)
        
        # Action Label at Start (Index 5)
        if len(wps_ego) > 5:
            lbl_pt = wps_ego[5]
        elif len(wps_ego) > 0:
            lbl_pt = wps_ego[0]
        else:
            lbl_pt = None
            
        if lbl_pt is not None:
             mt = row.get('maneuver_type', '')
             at = row.get('action_token', '')
             ax.text(lbl_pt[0], lbl_pt[1], f"{mt}\n{at}", color='blue', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

    # 5. Ego Car (With Goal Name)
    ax.add_patch(plt.Rectangle((-1, -2), 2, 4, fill=False, edgecolor='red', linewidth=2))
    goal_name = row['goal_name']
    ax.text(0, 0, goal_name, color='red', ha='center', va='center', fontweight='bold', fontsize=9)

    # 6. HUD Text
    info_text = (
        f"Goal: {goal_name} ({row['goal_token']})\n"
        f"Maneuver: {row.get('maneuver_type', '-')}\n"
        f"Action: {row.get('action_token', '-')}\n"
        f"Sample: {sample_token}"
    )
    ax.text(-45, 45, info_text, color='black', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9))

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)
    
    print(f"Indexing CSV by Action Tokens...")
    # Map: action_token -> set(scene_token)
    action_to_scenes = {}
    # Map: scene_token -> list(rows)
    scene_rows = {}
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row['scene_token']
            at = row.get('action_token', '').strip()
            
            if st not in scene_rows:
                scene_rows[st] = []
            scene_rows[st].append(row)
            
            # Add scene to action bucket if meaningful
            if at and at != "N/A":
                if at not in action_to_scenes:
                    action_to_scenes[at] = set()
                action_to_scenes[at].add(st)
                
    actions = sorted(list(action_to_scenes.keys()))
    print(f"Found {len(actions)} unique actions.")
    
    for action in actions:
        candidate_scenes = list(action_to_scenes[action])
        
        # Select up to SCENES_PER_ACTION
        # Shuffle for randomness
        random.shuffle(candidate_scenes)
        selected_scenes = candidate_scenes[:SCENES_PER_ACTION]
        
        print(f"Processing Action: {action} ({len(selected_scenes)} scenes)...")
        
        action_dir = os.path.join(OUTPUT_DIR, action)
        if not os.path.exists(action_dir):
            os.makedirs(action_dir)
            
        for st in tqdm(selected_scenes, desc=f"  Scenes ({action})", leave=False):
            scene_dir = os.path.join(action_dir, st)
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            
            rows = scene_rows[st]
            # Limit frames? No, user asked for full visualization usually.
            # But "create visualisation samples" might imply we don't need 400 images if we just want to verify.
            # But the previous tool generated full scene. Let's stick to full scene.
            
            for i, row in enumerate(rows):
                frame_name = f"frame_{i:03d}.png"
                save_path = os.path.join(scene_dir, frame_name)
                # Skip if exists? No, overwritten logic is safer.
                try:
                    render_bev_frame(nusc, row, save_path)
                except Exception as e:
                    print(f"Error {st} {i}: {e}")

    print(f"Done. Organized output in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
