
import rerun as rr
import csv
import numpy as np
import ast
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# Config
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "/home/santhru/FYP38_First Experiment/NuScenesVQA-/QWEN_VL_AD/output/nuscenes_action_tokens.csv"

# Helper for coordinate transform
def transform_to_ego(points, ego_trans, ego_rot_q):
    """
    points: (N, 3) numpy array
    ego_trans: [x, y, z]
    ego_rot_q: [w, x, y, z] (NuScenes format)
    """
    qt = Quaternion(ego_rot_q)
    # Global -> Ego: P_ego = R_inv * (P_global - T)
    # Centering
    p_centered = points - np.array(ego_trans)
    # Rotation (inverse)
    # Quaternion.rotate handles single vector. For array, use matrix.
    # mat is 3x3. P is Nx3.
    # P_ego = (R_inv @ P_centered.T).T
    
    # pyquaternion rotation_matrix is for q. To rotate FRAME, we need inverse?
    # No, to transform POINT from Global to Ego, we rotate by inverse of Ego orientation.
    # q is orientation of ego in global.
    rot_mat = qt.inverse.rotation_matrix
    
    return np.dot(p_centered, rot_mat.T)

def main():
    print("Initializing Rerun...")
    rr.init("NuScenes Goal Tokens", spawn=False)
    
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
        
    # Scene Selection Logic
    # ---------------------




    TARGET_SCENE_TOKEN = None # Set to a specific token str, or None for random
    # ---------------------
    
    if TARGET_SCENE_TOKEN is not None:
        if TARGET_SCENE_TOKEN in scenes_map:
            selected_token = TARGET_SCENE_TOKEN
            print(f"Selected specific scene: {selected_token}")
        else:
            print(f"Error: Scene {TARGET_SCENE_TOKEN} not found in CSV.")
            return
    else:
        import random
        selected_token = random.choice(list(scenes_map.keys()))
        print(f"Selected random scene: {selected_token}")
        
    rows_to_visualize = scenes_map[selected_token]
    print(f"Visualizing {len(rows_to_visualize)} samples from scene {selected_token}...")
    
    # Iterate roughly by time
    # Rerun logic: we can log everything to a timeline based on timestamp or just index.
    # The samples in CSV are per-scene, so sequential.
    


    for i, row in enumerate(rows_to_visualize):
        # Set time
        rr.set_time("step", sequence=i)
        
        scene_token = row['scene_token']
        try:
            scene = nusc.get('scene', scene_token)
        except KeyError:
            continue
            
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        
        if map_name not in maps:
            # Load map
            maps[map_name] = NuScenesMap(dataroot=NUSCENES_ROOT, map_name=map_name)

        nusc_map = maps[map_name]
        ego_x = float(row['ego_x'])
        ego_y = float(row['ego_y'])
        ego_yaw = float(row.get('ego_yaw', 0.0))
        goal_name = row['goal_name']
        waypoints = ast.literal_eval(row['waypoints']) # list of [x, y]
        
        # Load Sample Data (LiDAR & Pose)
        sample_token = row['sample_token']
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        ego_trans = pose_record['translation']
        ego_rot = pose_record['rotation'] # wxyz
        
        # 0. Log Ego Transform (To align Camera)
        # We define world/ego frame. Camera is here.
        # But we actually don't need to move the frame if we transform everything INTO it.
        # ... Wait. If we put the view at "world/ego", we want to see things relative to it.
        # So logging Identity transform here is conceptually what we want (camera at 0,0,0 of this view).
        # Actually Rerun needs to know how 'world/ego' relates to 'world' only if we render 'world'.
        # But we are rendering everything AS children of 'world/ego'.
        # So 'world/ego' becomes our root for visualization essentially.
        
        # However, to be nice, we can still log it in 'world' if we ever swich views.
        q_xyzw = [ego_rot[1], ego_rot[2], ego_rot[3], ego_rot[0]]
        rr.log("world/ego", rr.Transform3D(
            translation=ego_trans,
            rotation=rr.Quaternion(xyzw=q_xyzw)
        ))
        
        # 1. Map (Transformed to Ego)
        # We only log lanes near ego to be fast
        records = nusc_map.get_records_in_radius(ego_x, ego_y, radius=80, layer_names=['lane', 'lane_connector'])
        lane_tokens = records.get('lane', []) + records.get('lane_connector', [])
        
        if lane_tokens:
            try:
                lanes_geom = nusc_map.discretize_lanes(lane_tokens, 0.5)
                # Group by lane
                strips = []
                for t, pts in lanes_geom.items():
                    # pts is list of [x, y, z]
                    pts_np = np.array(pts)
                    pts_ego = transform_to_ego(pts_np, ego_trans, ego_rot)
                    strips.append(pts_ego)
                
                rr.log("world/ego/map", rr.LineStrips3D(strips, colors=[100, 100, 255]))
            except Exception as e:
                print(e)


        # 2. Annotations (Transformed to Ego)
        # Iterate sample annotations
        ann_tokens = sample['anns']
        # Collect box data
        centers = []
        sizes = []
        quaternions = []
        class_ids = []
        labels = []
        # Optional: Colors based on class? For now just one color or random
        
        qt_ego_inv = Quaternion(ego_rot).inverse
        
        for ann_token in ann_tokens:
            ann = nusc.get('sample_annotation', ann_token)
            
            # Center
            c_glob = np.array(ann['translation'])
            c_ego = transform_to_ego(c_glob.reshape(1,3), ego_trans, ego_rot)[0]
            centers.append(c_ego)
            

            # Size
            # NuScenes size is [width, length, height]
            # We want [length, width, height] because rotation aligns X with length (forward)
            s = ann['size']
            sizes.append([s[1], s[0], s[2]]) 
            
            # Rotation
            q_glob = Quaternion(ann['rotation'])
            q_ego = qt_ego_inv * q_glob # Combined rotation
            # Rerun expects xyzw
            quaternions.append([q_ego[1], q_ego[2], q_ego[3], q_ego[0]])
            
            class_ids.append(hash(ann['category_name']) % 255)
            labels.append(ann['category_name'])
            
        if centers:
            rr.log("world/ego/annotations", rr.Boxes3D(
                centers=centers,
                sizes=sizes,
                quaternions=quaternions,
                class_ids=class_ids,
                labels=labels,
                colors=[255, 100, 100]
            ))
        
        # 3. LiDAR (Already Local -> Ego)
        # Load PC
        pcl_path = os.path.join(nusc.dataroot, sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        points = pc.points[:3, ::5].T 
        rr.log("world/ego/lidar", rr.Points3D(points, colors=[200, 200, 200], radii=0.05))
        

        # Color Mapping
        # RGB
        c_green = [0, 255, 0]
        c_cyan = [0, 255, 255]
        c_orange = [255, 165, 0]
        c_red = [255, 0, 0]
        c_gray = [200, 200, 200]
        
        goal_color = c_green # Default
        if "LEFT" in goal_name:
            goal_color = c_cyan
        elif "RIGHT" in goal_name:
            goal_color = c_orange
        elif goal_name == "UNKNOWN":
            goal_color = c_gray

        # 4. Ego Geometry (At Origin)
        # REDUCED SIZE
        rr.log("world/ego/geometry", rr.Points3D([0, 0, 0], colors=[255, 0, 0], radii=0.2)) 
        arrow_len = 4.0
        rr.log("world/ego/heading", rr.Arrows3D(
            vectors=[[arrow_len, 0, 0]], 
            origins=[[0, 0, 0]],
            colors=goal_color 
        ))
        

        # 5. Waypoints (Transformed)
        midpoint_ego = None
        if waypoints:
            wps_np = np.array([[p[0], p[1], 0] for p in waypoints])
            wps_ego = transform_to_ego(wps_np, ego_trans, ego_rot)
            rr.log("world/ego/waypoints/future", rr.LineStrips3D([wps_ego], colors=goal_color, radii=0.2))
            

            if len(wps_ego) > 5:
                mid_idx = 5 # "Towards beginning with gap"
                midpoint_ego = wps_ego[mid_idx]
            elif len(wps_ego) > 0:
                midpoint_ego = wps_ego[0]
            
        # Check for past waypoints in row
        if 'past_waypoints' in row:
            p_wps = ast.literal_eval(row['past_waypoints'])
            if p_wps:
                p_wps_np = np.array([[p[0], p[1], 0] for p in p_wps])
                p_wps_ego = transform_to_ego(p_wps_np, ego_trans, ego_rot)
                rr.log("world/ego/waypoints/past", rr.LineStrips3D([p_wps_ego], colors=[150, 150, 150], radii=0.2)) # Grey/White for past


        # 6. Goal (Prominent Display)
        # Markdown Header level 1 for maximum visibility
        rr.log(
            "world/hud/goal", 
            rr.TextDocument(f"# {goal_name}\nMap: {map_name}")
        )
        
        # 7. 3D Label above car
        # A point at (0,0,3) in ego frame with the text
        rr.log(
            "world/ego/goal_label",
            rr.Points3D([[0, 0, 3]], labels=[f"GOAL: {goal_name}"], colors=goal_color, radii=0.0)
        )
        
        # 8. New: Maneuver/Action Label at Midpoint
        maneuver_type = row.get('maneuver_type', 'N/A')
        action_token = row.get('action_token', 'N/A')
        
        if midpoint_ego is not None:
             # Raise it slightly (z=2) so it floats above the line
             label_pos = midpoint_ego + np.array([0, 0, 2.0]) 
             rr.log(
                "world/ego/midpoint_label",
                rr.Points3D([label_pos], labels=[f"{maneuver_type}\n{action_token}"], colors=goal_color, radii=0.0)
             )
        

    import rerun.blueprint as rbl
    
    # Define Blueprint to track Ego
    # Single view, tracking world/ego
    blueprint = rbl.Blueprint(
        rbl.Spatial3DView(origin="world/ego", name="Chase Camera"),
    )
    rr.send_blueprint(blueprint)
    
    output_path = "rerun_logs/nuscenes_goals.rrd"
    rr.save(output_path)
    print(f"Saved Rerun log to {output_path}")

if __name__ == "__main__":
    main()
