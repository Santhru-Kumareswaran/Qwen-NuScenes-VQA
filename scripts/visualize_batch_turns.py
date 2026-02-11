
import rerun as rr
import csv
import numpy as np
import ast
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from tqdm import tqdm

# Config
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
CSV_PATH = "nuscenes_goal_tokens_8seconds.csv"
OUTPUT_RRD = "rerun_logs/nuscenes_turns_batch.rrd"

TURN_SCENES = [
    "03ee880dd4e348f4b3407f0d073c7c70", "05d5241d4d5a4c2a9c670dc8c9853c74",
    "0be1ff07a8f148ca9535fb7f0deaf828", "0dae482684ce4cd69a7258f55bc98d73",
    "1b3b150c9d3e4e1593e7ce882a69f870", "1e3f0bd8063a4084b7104345b20bfb80",
    "1fbcc26ebf6948bb964d3ae74939e8ea", "212b65c558254e2084489bf76e371e1a",
    "26540bfbab79463cb1ba76b52ec6013b", "28d385e6db0e495da3a606b58e2432f0",
    "2987b4494fcc4639bdd6d714a94b2c72", "2bfb95d8ba3a4c11869f4d6635784640",
    "2c96ff6afc9e4cf7bc8b107fce955c1f", "2e3c0f308fcf4664bf7994053d7080b9",
    "2eb0dd074d8e4a328fd2283184c4412e", "2eb4d7f00e584a548aa0b899638bfb0a",
    "3289046b886f4c98a676bd7e6a3c6ef1", "3b2ee26cb8484f77895bc336663df502",
    "3be5ff913abf449bac92ce2d2a4cffa0", "433a14f8dcf5457fb2c4def5c749122a",
    "45fc8a7a26c5469a88db999ae8468777", "4772f5283211444eaabd463cd341d360",
    "4bcb814456f044919bc052c208dbebc5", "5a1d2867bb504c888a628ad549cc10b6",
    "5d709891c41d423687ae4ea0473cb9c4", "634a8c5835e44aec912604a9a1972a5d",
    "64ca274b5f69458bb5d21e0e2e071902", "65cfdc10a42e499ea704c617e077afe0",
    "670e01eb31b64b509db5290531254203", "68fc5e3698a544d1bdf5847b88ad77d4"
]

def transform_to_ego(points, ego_trans, ego_rot_q):
    qt = Quaternion(ego_rot_q)
    p_centered = points - np.array(ego_trans)
    rot_mat = qt.inverse.rotation_matrix
    return np.dot(p_centered, rot_mat.T)

def main():
    if not os.path.exists("rerun_logs"):
        os.makedirs("rerun_logs")

    print(f"Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)
    
    print(f"Reading {CSV_PATH}...")
    scenes_map = {}
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row['scene_token']
            if st in TURN_SCENES:
                if st not in scenes_map:
                    scenes_map[st] = []
                scenes_map[st].append(row)

    rr.init("NuScenes Turns Batch", spawn=False)
    
    global_step = 0
    maps = {}

    for st in tqdm(TURN_SCENES, desc="Processing Scenes"):
        if st not in scenes_map:
            continue
            
        rows = scenes_map[st]
        
        scene_rec = nusc.get("scene", st)
        log = nusc.get("log", scene_rec["log_token"])
        map_name = log["location"]
        if map_name not in maps:
            maps[map_name] = NuScenesMap(dataroot=NUSCENES_ROOT, map_name=map_name)
        nusc_map = maps[map_name]

        for i, row in enumerate(rows):
            rr.set_time("step", sequence=global_step)
            global_step += 1
            
            sample_token = row['sample_token']
            sample = nusc.get('sample', sample_token)
            
            sd_token = sample['data']['LIDAR_TOP']
            sd_rec = nusc.get('sample_data', sd_token)
            
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
            ego_trans = pose_record['translation']
            ego_rot = pose_record['rotation']
            
            # --- Visual Elements ---

            # 1. Map (Local View)
            records = nusc_map.get_records_in_radius(ego_trans[0], ego_trans[1], 80.0, ['lane', 'lane_connector'])
            lane_tokens = records.get('lane', []) + records.get('lane_connector', [])
            if lane_tokens:
                lanes_geom = nusc_map.discretize_lanes(lane_tokens, 0.5)
                for t, pts in lanes_geom.items():
                    points_idx = np.array(pts)[:, :3]
                    ego_points = transform_to_ego(points_idx, ego_trans, ego_rot)
                    rr.log(f"world/map/lanes/{t}", rr.LineStrips3D([ego_points], colors=[100, 100, 100], radii=0.1))

            # 2. Annotations (Other Vehicles)
            ann_tokens = sample['anns']
            centers, sizes, quaternions, labels = [], [], [], []
            qt_ego_inv = Quaternion(ego_rot).inverse
            for ann_token in ann_tokens:
                ann = nusc.get('sample_annotation', ann_token)
                c_ego = transform_to_ego(np.array(ann['translation']).reshape(1,3), ego_trans, ego_rot)[0]
                centers.append(c_ego)
                s = ann['size']
                sizes.append([s[1], s[0], s[2]]) 
                q_ego = qt_ego_inv * Quaternion(ann['rotation'])
                quaternions.append([q_ego[1], q_ego[2], q_ego[3], q_ego[0]])
                labels.append(ann['category_name'])
            
            if centers:
                rr.log("world/ego/annotations", rr.Boxes3D(centers=centers, sizes=sizes, quaternions=quaternions, labels=labels, colors=[255, 100, 100]))

            # 3. Lidar
            pc_path = os.path.join(NUSCENES_ROOT, sd_rec['filename'])
            pc = LidarPointCloud.from_file(pc_path)
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))
            points = pc.points[:3, ::10].T 
            rr.log("world/ego/lidar", rr.Points3D(points, colors=[200, 200, 200], radii=0.05))

            # 4. Ego Bounding Box
            rr.log("world/ego/geometry", rr.Boxes3D(centers=[[0,0,0.8]], sizes=[[4.5, 2.0, 1.6]], colors=[255, 0, 0]))

            # 5. Goal Logic & Colors
            goal_name = row['goal_name']
            c_green = [0, 255, 0]
            c_blue = [0, 255, 255]
            c_orange = [255, 130, 0]
            goal_color = c_green
            if "LEFT" in goal_name: goal_color = c_blue
            elif "RIGHT" in goal_name: goal_color = c_orange

            # 6. Waypoints
            waypoints = ast.literal_eval(row['waypoints'])
            if waypoints:
                wps_ego = transform_to_ego(np.array([[p[0], p[1], 0] for p in waypoints]), ego_trans, ego_rot)
                rr.log("world/ego/waypoints/future", rr.LineStrips3D([wps_ego], colors=goal_color, radii=0.25))
            
            if 'past_waypoints' in row:
                p_wps = ast.literal_eval(row['past_waypoints'])
                if p_wps:
                    p_wps_ego = transform_to_ego(np.array([[p[0], p[1], 0] for p in p_wps]), ego_trans, ego_rot)
                    rr.log("world/ego/waypoints/past", rr.LineStrips3D([p_wps_ego], colors=[150, 150, 150], radii=0.2))

            # 7. Heading Arrow
            rr.log("world/ego/heading", rr.Arrows3D(vectors=[[4.0, 0, 0]], origins=[[0, 0, 0]], colors=goal_color))

            # 8. HUD & Labels
            rr.log("world/hud/goal", rr.TextDocument(f"# {goal_name}\nScene: {st}"))
            rr.log("world/ego/goal_label", rr.Points3D([[0, 0, 3]], labels=[f"GOAL: {goal_name}"], colors=goal_color, radii=0.0))

        pass # Completed scene

    print(f"Saving to {OUTPUT_RRD}...")
    rr.save(OUTPUT_RRD)
    print("Done.")

if __name__ == "__main__":
    main()
