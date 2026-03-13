
import rerun as rr
import csv
import numpy as np
import ast
import os
import random
import matplotlib.pyplot as plt

import matplotlib.style as mpltstyle
import matplotlib.pyplot as plt

# Robust fix for NuScenes map-api issue with newer matplotlib
def safe_style_use(style):
    try:
        mpltstyle.original_use(style)
    except:
        try:
            if style == 'seaborn-whitegrid':
                mpltstyle.original_use('seaborn-v0_8-whitegrid')
            else:
                mpltstyle.original_use('ggplot')
        except:
            pass

if not hasattr(mpltstyle, 'original_use'):
    mpltstyle.original_use = mpltstyle.use
    mpltstyle.use = safe_style_use
    plt.style.use = safe_style_use

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

def setup_maps(nusc_root):
    """
    NuScenes Map Expansion expects maps in [dataroot]/maps/*.json.
    On this SSD, they are in [dataroot]/maps/expansion/*.json.
    We create a temporary directory to satisfy the devkit.
    """
    temp_map_root = "/tmp/nuscenes_maps_link"
    os.makedirs(os.path.join(temp_map_root, "maps"), exist_ok=True)
    
    src_dir = os.path.join(nusc_root, "maps/expansion")
    if not os.path.exists(src_dir):
        return nusc_root # Fallback
        
    for f in os.listdir(src_dir):
        if f.endswith(".json"):
            dst = os.path.join(temp_map_root, "maps", f)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.join(src_dir, f), dst)
                except: pass
    
    # We also need to link the 'v1.0-trainval' or other metadata folders 
    # if NuScenes object is also using this root, but here we only use it for maps.
    return temp_map_root

from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# Config
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "/home/santhru/FYP38_First Experiment/NuScenesVQA-/QWEN_VL_AD/output/nuscenes_action_tokens_with_predictions.csv"

# Helper for coordinate transform
def transform_to_ego(points, ego_trans, ego_rot_q):
    """
    points: (N, 3) numpy array
    ego_trans: [x, y, z]
    ego_rot_q: [w, x, y, z] (NuScenes format)
    """
    qt = Quaternion(ego_rot_q)
    # Global -> Ego: P_ego = R_inv * (P_global - T)
    p_centered = points - np.array(ego_trans)
    rot_mat = qt.inverse.rotation_matrix
    return np.dot(p_centered, rot_mat.T)


def correct_pred_to_ego(points_global, csv_ego_x, csv_ego_y, csv_ego_yaw, actual_ego_trans, actual_ego_rot):
    """
    Predicted / Reference waypoints were stored in global coords using
    ego_to_global(csv_ego_yaw).  This function corrects for any mismatch
    between csv_ego_yaw and the actual NuScenes quaternion heading:

      1. Undo the CSV rotation  → back to pure ego frame (template space)
      2. Re-apply actual NuScenes yaw → corrected global coords
      3. transform_to_ego using actual quaternion → visualization ego frame
    """
    if not points_global:
        return np.zeros((0, 3))
    pts_2d = np.array([[p[0], p[1]] for p in points_global], dtype=np.float64)

    # 1. Undo csv ego_to_global
    pts_2d -= np.array([csv_ego_x, csv_ego_y])
    c, s = np.cos(-csv_ego_yaw), np.sin(-csv_ego_yaw)
    R_undo = np.array([[c, -s], [s, c]])
    pts_ego = pts_2d @ R_undo.T  # now in pure ego / template frame

    # 2. Re-apply actual NuScenes yaw
    actual_yaw = Quaternion(actual_ego_rot).yaw_pitch_roll[0]
    c2, s2 = np.cos(actual_yaw), np.sin(actual_yaw)
    R_correct = np.array([[c2, -s2], [s2, c2]])
    pts_global_corrected = pts_ego @ R_correct.T + np.array([actual_ego_trans[0], actual_ego_trans[1]])

    # 3. Back to visualization ego space
    pts_3d = np.column_stack([pts_global_corrected, np.zeros(len(pts_global_corrected))])
    return transform_to_ego(pts_3d, actual_ego_trans, actual_ego_rot)

def main():
    print("Initializing Rerun...")
    rr.init("NuScenes Predicted Trajectories", spawn=False)
    
    # Load NuScenes
    print(f"Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)
    
    # Read CSV
    print(f"Reading {CSV_PATH}...")
    samples = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
            
    # Cache maps
    maps = {}
    
    # Group samples by scene
    scenes_map = {}
    for row in samples:
        st = row['scene_token']
        if st not in scenes_map:
            scenes_map[st] = []
        scenes_map[st].append(row)
        
    # Pick a random scene or use specific one
    TARGET_SCENE_TOKEN = None 
    
    if TARGET_SCENE_TOKEN is not None:
        selected_token = TARGET_SCENE_TOKEN
    else:
        selected_token = random.choice(list(scenes_map.keys()))
        
    rows_to_visualize = scenes_map[selected_token]
    print(f"Visualizing {len(rows_to_visualize)} samples from scene {selected_token}...")
    
    for i, row in enumerate(rows_to_visualize):
        rr.set_time("step", sequence=i)
        
        scene_token = row['scene_token']
        try:
            scene = nusc.get('scene', scene_token)
        except KeyError:
            continue
            
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        
        if map_name not in maps:
            map_root = setup_maps(NUSCENES_ROOT)
            maps[map_name] = NuScenesMap(dataroot=map_root, map_name=map_name)

        nusc_map = maps[map_name]
        ego_trans = [float(row['ego_x']), float(row['ego_y']), 0.0] # Basic translation from CSV
        
        # Load Sample Data for accurate Pose
        sample_token = row['sample_token']
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_token)
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        ego_trans = pose_record['translation']
        ego_rot = pose_record['rotation'] # wxyz
        
        # Log Ego Transform
        q_xyzw = [ego_rot[1], ego_rot[2], ego_rot[3], ego_rot[0]]
        rr.log("world/ego", rr.Transform3D(
            translation=ego_trans,
            rotation=rr.Quaternion(xyzw=q_xyzw)
        ))
        
        # 1. Map
        radius = 80
        patch_box = (ego_trans[0] - radius, ego_trans[1] - radius, ego_trans[0] + radius, ego_trans[1] + radius)
        available_layers = nusc_map.layer_names
        layers_to_get = [l for l in ['lane', 'lane_connector'] if l in available_layers]
        
        if layers_to_get:
            try:
                records = nusc_map.get_records_in_patch(patch_box, layer_names=layers_to_get, mode='intersect')
                lane_tokens = []
                for l in layers_to_get:
                    lane_tokens.extend(records.get(l, []))
                
                if lane_tokens:
                    lanes_geom = nusc_map.discretize_lanes(lane_tokens, 0.5)
                    strips = []
                    for t, pts in lanes_geom.items():
                        pts_np = np.array(pts)
                        pts_ego = transform_to_ego(pts_np, ego_trans, ego_rot)
                        strips.append(pts_ego)
                    rr.log("world/ego/map", rr.LineStrips3D(strips, colors=[100, 100, 255]))
            except: pass

        # 2. Annotations
        ann_tokens = sample['anns']
        centers, sizes, quaternions, labels = [], [], [], []
        qt_ego_inv = Quaternion(ego_rot).inverse
        
        for ann_token in ann_tokens:
            ann = nusc.get('sample_annotation', ann_token)
            c_glob = np.array(ann['translation'])
            c_ego = transform_to_ego(c_glob.reshape(1,3), ego_trans, ego_rot)[0]
            centers.append(c_ego)
            s = ann['size']
            sizes.append([s[1], s[0], s[2]]) 
            q_glob = Quaternion(ann['rotation'])
            q_ego = qt_ego_inv * q_glob 
            quaternions.append([q_ego[1], q_ego[2], q_ego[3], q_ego[0]])
            labels.append(ann['category_name'])
            
        if centers:
            rr.log("world/ego/annotations", rr.Boxes3D(
                centers=centers, sizes=sizes, quaternions=quaternions, labels=labels, colors=[255, 100, 100]
            ))
        
        # 3. LiDAR
        pcl_path = os.path.join(nusc.dataroot, sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        points = pc.points[:3, ::5].T 
        rr.log("world/ego/lidar", rr.Points3D(points, colors=[200, 200, 200], radii=0.05))
        
        def safe_parse_waypoints(w_str):
            if not w_str or w_str.strip() == "": return None
            try: return ast.literal_eval(w_str)
            except: return None

        # 4. Waypoints
        # Ground Truth / Original
        goal_name = row['goal_name']
        goal_color = [0, 255, 0] # Green
        if "LEFT" in goal_name: goal_color = [0, 255, 255] # Cyan
        elif "RIGHT" in goal_name: goal_color = [255, 165, 0] # Orange
        
        wps = safe_parse_waypoints(row.get('waypoints'))
        if wps:
            wps_np = np.array([[p[0], p[1], 0] for p in wps])
            wps_ego = transform_to_ego(wps_np, ego_trans, ego_rot)
            rr.log("world/ego/waypoints/gt", rr.LineStrips3D([wps_ego], colors=goal_color, radii=0.2, labels=["GT"]))

        # Predicted Waypoints
        p_wps = safe_parse_waypoints(row.get('predicted_waypoints'))
        if p_wps:
            p_wps_ego = correct_pred_to_ego(
                p_wps,
                float(row['ego_x']), float(row['ego_y']), float(row['ego_yaw']),
                ego_trans, ego_rot
            )
            rr.log("world/ego/waypoints/predicted", rr.LineStrips3D([p_wps_ego], colors=[255, 0, 0], radii=0.15, labels=["Predicted"]))

        # Reference Trajectory
        r_wps = safe_parse_waypoints(row.get('reference_trajectory'))
        if r_wps:
            r_wps_ego = correct_pred_to_ego(
                r_wps,
                float(row['ego_x']), float(row['ego_y']), float(row['ego_yaw']),
                ego_trans, ego_rot
            )
            rr.log("world/ego/waypoints/reference", rr.LineStrips3D([r_wps_ego], colors=[0, 0, 255], radii=0.1, labels=["Reference"]))

        # Past Waypoints
        past = safe_parse_waypoints(row.get('past_waypoints'))
        if past:
            past_np = np.array([[p[0], p[1], 0] for p in past])
            past_ego = transform_to_ego(past_np, ego_trans, ego_rot)
            rr.log("world/ego/waypoints/past", rr.LineStrips3D([past_ego], colors=[150, 150, 150], radii=0.2))

        # 3D label above ego vehicle
        action_token = row.get('action_token', 'N/A')
        label_text = f"🎯 {action_token}\n📍 {goal_name}"
        rr.log("world/ego/label", rr.Points3D(
            [[0, 0, 4.0]],   # 4m above ego in ego-local frame
            labels=[label_text],
            radii=0.01,
            colors=[[255, 255, 0]]   # bright yellow
        ))

        # HUD panel
        rr.log("world/hud/info", rr.TextDocument(
            f"**Action Token:** `{action_token}`\n"
            f"**Goal:** `{goal_name}`\n"
            f"**Sample:** `{sample_token}`"
        ))

    import rerun.blueprint as rbl
    blueprint = rbl.Blueprint(
        rbl.Spatial3DView(origin="world/ego", name="Trajectory Comparison"),
    )
    rr.send_blueprint(blueprint)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "rerun_logs", "predicted_trajectories.rrd")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rr.save(output_path)
    print(f"Saved Rerun log to {output_path}")

if __name__ == "__main__":
    main()
