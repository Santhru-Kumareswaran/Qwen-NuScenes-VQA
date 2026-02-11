
import csv
import os
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

# ================= CONFIG =================
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "/home/santhru/FYP38_First Experiment/NuScenesVQA-/QWEN_VL_AD/output/nuscenes_action_tokens.csv"
OUTPUT_DIR = "bev_frames_vis"
NUM_SCENES = 10
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
    """
    Render a single BEV frame using matplotlib.
    """
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
    
    # Transform to Ego Frame
    # First apply sensor calibration (Sensor -> Ego)
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    # We plot in Ego frame, so Ego is at (0,0) facing +X (usually) depending on convention.
    # NuScenes Ego frame: X forward, Y left, Z up.
    
    # Filter points for BEV (e.g. radius 50m)
    mask = np.abs(pc.points[0, :]) < 50
    mask &= np.abs(pc.points[1, :]) < 50
    points = pc.points[:, mask]
    
    # Plot Points
    ax.scatter(points[0, :], points[1, :], s=0.1, c='gray', alpha=0.5)
    
    # 3. Annotations (Boxes)
    _, boxes, _ = nusc.get_sample_data(sd_token, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=True)
    
    for box in boxes:
        # box.center, box.wlh, box.orientation are already likely in proper frame if get_sample_data handled it
        # Wait, get_sample_data returns boxes in SENSOR frame or GLOBAL?
        # "Returns the data path as well as all annotations in the given sample_data (sensor coordinate system)."
        # So we need to transform Box to Ego?
        # Actually get_sample_data(use_flat_vehicle_coordinates=True) implies Ego frame?
        # Documentation says: "If True, the boxes will be converted to the vehicle coordinates (ego frame)..."
        # Let's verify. Yes.
        
        c = get_color(box.name)
        box.render(ax, view=np.eye(4), colors=(c, c, c))

    # 4. Waypoints
    waypoints = ast.literal_eval(row['waypoints'])
    if waypoints:
        wps = np.array(waypoints)
        # These are Global coordinates. Need to transform to Ego.
        
        # Transform global to ego
        # P_ego = R_inv * (P_glob - T_ego)
        ego_trans = np.array(pose_record['translation'])
        ego_rot = Quaternion(pose_record['rotation'])
        
        wps_centered = wps - ego_trans[:2] # X,Y only for 2D waypoints?
        # Waypoints from CSV might be just [x, y]
        
        # Rotate
        # We need full 3D rotation usually, but if flat map...
        # Let's assume 3D transform for robustness
        wps_3d = np.hstack([wps, np.zeros((len(wps), 1))]) # Add Z=0
        wps_centered_3d = wps_3d - ego_trans
        wps_ego = np.dot(wps_centered_3d, ego_rot.inverse.rotation_matrix.T)
        
        ax.plot(wps_ego[:, 0], wps_ego[:, 1], 'g-', linewidth=2, label='Future Path')
        
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


    # 5. Ego Car
    # Draw simple rect at 0,0
    ax.add_patch(plt.Rectangle((-1, -2), 2, 4, fill=False, edgecolor='red', linewidth=2))
    
    # Goal Name on top of Ego
    goal_name = row['goal_name']
    # Place text slightly above the car box or centered?
    # "on top of" -> overlaid.
    ax.text(0, 0, goal_name, color='red', ha='center', va='center', fontweight='bold', fontsize=9)

    # 6. HUD Text
    goal_name = row['goal_name']
    goal_token = row['goal_token']
    
    info_text = (
        f"Goal: {goal_name} ({goal_token})\n"
        f"Maneuver: {row.get('maneuver_type', '-')}\n"
        f"Action: {row.get('action_token', '-')}"
    )
    
    # Place text box in top left
    ax.text(-45, 45, info_text, color='black', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9))

    # Formatting
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
    
    print(f"Reading CSV: {CSV_PATH}")
    samples_by_scene = {}
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row['scene_token']
            if st not in samples_by_scene:
                samples_by_scene[st] = []
            samples_by_scene[st].append(row)
            
    # Select Scenes
    all_scenes = list(samples_by_scene.keys())
    # You can change logic here to pick specific scenes or random
    selected_scenes = all_scenes[:NUM_SCENES]
    
    print(f"Generating frames for {len(selected_scenes)} scenes...")
    
    for st in tqdm(selected_scenes, desc="Scenes"):
        scene_dir = os.path.join(OUTPUT_DIR, st)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
            
        rows = samples_by_scene[st]
        for i, row in enumerate(rows):
            frame_name = f"frame_{i:03d}.png"
            save_path = os.path.join(scene_dir, frame_name)
            
            try:
                render_bev_frame(nusc, row, save_path)
            except Exception as e:
                print(f"Error frame {i} scene {st}: {e}")

    print(f"Done. Output saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
