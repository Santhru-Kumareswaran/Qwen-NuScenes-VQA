
import csv
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# ================= CONFIG =================
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "nuscenes_goal_tokens_refined_subset_2.csv"
OUTPUT_DIR = "bev_frames_refined_subset_2"
# ==========================================

def render_bev_frame(nusc, row, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sample_token = row['sample_token']
    try:
        sample = nusc.get('sample', sample_token)
    except:
        plt.close(fig)
        return

    sd_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sd_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # LiDAR
    pcl_path = os.path.join(nusc.dataroot, sd_rec['filename'])
    if not os.path.exists(pcl_path):
        plt.close(fig)
        return
        
    pc = LidarPointCloud.from_file(pcl_path)
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    mask = (np.abs(pc.points[0, :]) < 50) & (np.abs(pc.points[1, :]) < 50)
    points = pc.points[:, mask]
    ax.scatter(points[0, :], points[1, :], s=0.1, c='gray', alpha=0.5)
    
    # Boxes
    _, boxes, _ = nusc.get_sample_data(sd_token, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=True)
    for box in boxes:
        c = 'r' if 'vehicle' in box.name else 'b'
        box.render(ax, view=np.eye(4), colors=(c, c, c))

    # Waypoints
    waypoints_str = row.get('waypoints', '[]')
    if waypoints_str and waypoints_str != "[]":
        try:
            waypoints = ast.literal_eval(waypoints_str)
            wps = np.array(waypoints)
            ego_trans = np.array(pose_record['translation'])
            ego_rot = Quaternion(pose_record['rotation'])
            
            wps_3d = np.hstack([wps, np.zeros((len(wps), 1))])
            wps_centered_3d = wps_3d - ego_trans
            wps_ego = np.dot(wps_centered_3d, ego_rot.inverse.rotation_matrix.T)
            
            ax.plot(wps_ego[:, 0], wps_ego[:, 1], 'g-', linewidth=2)
            
            # Action Label
            lbl_pt = wps_ego[5] if len(wps_ego) > 5 else (wps_ego[0] if len(wps_ego) > 0 else None)
            if lbl_pt is not None:
                mt = row.get('maneuver_type', '-')
                at = row.get('action_token', '-')
                ax.text(lbl_pt[0], lbl_pt[1], f"{mt}\n{at}", color='blue', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))
        except:
            pass

    # Ego with Goal Name
    ax.add_patch(plt.Rectangle((-1, -2), 2, 4, fill=False, edgecolor='red', linewidth=2))
    goal_name = row.get('goal_name', 'UNKNOWN')
    ax.text(0, 0, goal_name, color='red', ha='center', va='center', fontweight='bold', fontsize=9)

    # HUD
    info_text = (
        f"Goal: {goal_name}\n"
        f"Maneuver: {row.get('maneuver_type', '-')}\n"
        f"Action: {row.get('action_token', '-')}\n"
        f"Sample: {sample_token}"
    )
    ax.text(-45, 45, info_text, color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.9))

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
    
    print(f"Reading {CSV_PATH}...")
    scene_rows = {}
    
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row['scene_token']
            if st not in scene_rows:
                scene_rows[st] = []
            scene_rows[st].append(row)
    
    print(f"Generating BEV frames for {len(scene_rows)} scenes...")
    
    for st in tqdm(scene_rows.keys(), desc="Scenes"):
        scene_dir = os.path.join(OUTPUT_DIR, st)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        
        for i, row in enumerate(scene_rows[st]):
            save_path = os.path.join(scene_dir, f"frame_{i:03d}.png")
            try:
                render_bev_frame(nusc, row, save_path)
            except Exception as e:
                pass

    print(f"Done. Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
