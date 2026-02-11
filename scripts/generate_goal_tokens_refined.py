
import csv
import numpy as np
from enum import IntEnum
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os

# ================= CONFIG =================
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION = "v1.0-trainval"
ACTION_CSV_PATH = "/home/santhru/FYP38_First Experiment/nuscenes_action_tokens.csv"
OUTPUT_CSV = "nuscenes_goal_tokens_refined.csv"
MAX_SCENES = None  # Process all scenes
SKIP_SCENES = 0

# Prediction Windows
TURN_WINDOW = 40      # 20 seconds - actual turn detection
PREPARE_THRESHOLD = 10  # If turn is beyond index 10 (~5 seconds away), use PREPARE
# =========================================

class GoalToken(IntEnum):
    GO_STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    PREPARE_LEFT = 3
    PREPARE_RIGHT = 4

def quaternion_yaw(q):
    return R.from_quat(q).as_euler("xyz")[2]

def get_ego_pose(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    x, y, _ = ego_pose["translation"]
    yaw = quaternion_yaw(ego_pose["rotation"])
    return np.array([x, y]), yaw

def get_past_waypoints(nusc, sample, num_past=20):
    waypoints = []
    curr_sample = sample
    for _ in range(num_past):
        if not curr_sample["prev"]:
            break
        curr_sample = nusc.get("sample", curr_sample["prev"])
        xy, _ = get_ego_pose(nusc, curr_sample)
        waypoints.insert(0, xy.tolist())
    return waypoints

def get_future_waypoints(nusc, sample, num_future=40):
    waypoints = []
    curr_sample = sample
    for _ in range(num_future):
        if not curr_sample["next"]:
            break
        curr_sample = nusc.get("sample", curr_sample["next"])
        xy, _ = get_ego_pose(nusc, curr_sample)
        waypoints.append(xy.tolist())
    return waypoints

def detect_turn_in_trajectory(waypoints, ego_xy, ego_yaw, start_idx=0):
    """
    Detect if there's a turn in the trajectory starting from start_idx.
    Returns: (turn_type: 'left'|'right'|None, turn_index: int)
    """
    if len(waypoints) <= start_idx + 5:
        return None, -1
    
    wps = np.array(waypoints)
    wps_centered = wps - ego_xy
    
    cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
    wps_local = np.zeros_like(wps_centered)
    wps_local[:, 0] = cos_yaw * wps_centered[:, 0] - sin_yaw * wps_centered[:, 1]
    wps_local[:, 1] = sin_yaw * wps_centered[:, 0] + cos_yaw * wps_centered[:, 1]
    
    # Calculate headings between consecutive points
    headings = []
    for i in range(len(wps_local) - 1):
        dx = wps_local[i+1, 0] - wps_local[i, 0]
        dy = wps_local[i+1, 1] - wps_local[i, 1]
        if abs(dx) > 0.01 or abs(dy) > 0.01:  # Avoid noise
            headings.append(np.arctan2(dy, dx))
        else:
            headings.append(headings[-1] if headings else 0)
    
    if len(headings) < 3:
        return None, -1
    
    # Look for significant heading change (turn)
    cumulative_heading = 0.0
    for i in range(start_idx, len(headings) - 1):
        diff = headings[i+1] - headings[i]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        cumulative_heading += diff
        
        # Check if we've accumulated enough heading change for a turn
        if cumulative_heading > np.radians(40):  # Left turn threshold
            return 'left', i
        elif cumulative_heading < np.radians(-40):  # Right turn threshold
            return 'right', i
    
    return None, -1

def main():
    if not os.path.exists(NUSCENES_ROOT):
        print(f"Error: NuScenes root not found")
        return

    print(f"Loading action data...")
    merge_map = {}
    if os.path.exists(ACTION_CSV_PATH):
        with open(ACTION_CSV_PATH, 'r') as f:
            for row in csv.DictReader(f):
                merge_map[(row['scene_token'], row['sample_token'])] = {
                    'maneuver_type': row.get('maneuver_type', ''),
                    'action_token': row.get('action_token', '')
                }

    print(f"Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)

    print(f"Generating {OUTPUT_CSV} for ALL scenes...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_token", "sample_token", "ego_x", "ego_y", "ego_yaw",
            "goal_token", "goal_name",
            "waypoints", "past_waypoints",
            "maneuver_type", "action_token"
        ])
        
        scenes = nusc.scene
        if MAX_SCENES is not None:
             scenes = scenes[SKIP_SCENES : SKIP_SCENES + MAX_SCENES]
        
        for scene in tqdm(scenes, desc="Scenes"):
            sample_token = scene["first_sample_token"]
            
            while sample_token:
                sample = nusc.get("sample", sample_token)
                ego_xy, ego_yaw = get_ego_pose(nusc, sample)
                
                waypoints = get_future_waypoints(nusc, sample, TURN_WINDOW)
                past_waypoints = get_past_waypoints(nusc, sample, 20)
                
                # === REFINED LOGIC ===
                # 1. Detect any turn in the entire lookahead (40 samples = 20s)
                turn_type, turn_idx = detect_turn_in_trajectory(waypoints, ego_xy, ego_yaw, start_idx=0)
                
                final_goal = GoalToken.GO_STRAIGHT
                
                if turn_type:
                    # Turn detected - check if it's imminent or upcoming
                    if turn_idx < PREPARE_THRESHOLD:
                        # Turn is happening soon (within ~5 seconds) - actual TURN
                        if turn_type == 'left':
                            final_goal = GoalToken.TURN_LEFT
                        else:
                            final_goal = GoalToken.TURN_RIGHT
                    else:
                        # Turn is further away - PREPARE
                        if turn_type == 'left':
                            final_goal = GoalToken.PREPARE_LEFT
                        else:
                            final_goal = GoalToken.PREPARE_RIGHT
                
                # Merge Action Data
                merge_data = merge_map.get((scene["token"], sample_token), {})
                
                writer.writerow([
                    scene["token"], sample_token, ego_xy[0], ego_xy[1], ego_yaw,
                    int(final_goal), final_goal.name,
                    waypoints, past_waypoints,
                    merge_data.get('maneuver_type', ''), merge_data.get('action_token', '')
                ])
                
                sample_token = sample["next"]
                
    print(f"Done. Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
