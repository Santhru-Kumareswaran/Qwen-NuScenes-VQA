
import csv
import numpy as np
from enum import IntEnum
from tqdm import tqdm
import os

# ================= CONFIG =================
INPUT_CSV = "/home/santhru/FYP38_First Experiment/nuscenes_action_tokens.csv"
OUTPUT_CSV = "nuscenes_goal_tokens_aligned.csv"
# =========================================

class GoalToken(IntEnum):
    UNKNOWN = -1
    KEEP_LANE = 0
    CHANGE_LANE_LEFT = 1
    CHANGE_LANE_RIGHT = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    GO_STRAIGHT = 5
    PREPARE_LEFT = 6
    PREPARE_RIGHT = 7

def maneuver_to_goal(maneuver_type, action_token):
    """
    Map maneuver_type and action_token to a GoalToken.
    This ensures timing alignment with the maneuver labels.
    """
    m = maneuver_type.upper() if maneuver_type else ""
    a = action_token.upper() if action_token else ""
    
    # Priority Check: Action Token First (more specific)
    if "LEFT_TURN" in a or "U_TURN_LEFT" in a:
        return GoalToken.TURN_LEFT
    if "RIGHT_TURN" in a or "U_TURN_RIGHT" in a:
        return GoalToken.TURN_RIGHT
    if "LEFT_SLIDE" in a:
        return GoalToken.CHANGE_LANE_LEFT
    if "RIGHT_SLIDE" in a:
        return GoalToken.CHANGE_LANE_RIGHT
    
    # Fallback: Maneuver Type
    if m == "LEFT":
        # Could be turn or lane change, assume turn if not specified
        return GoalToken.TURN_LEFT
    if m == "RIGHT":
        return GoalToken.TURN_RIGHT
    if m == "STRAIGHT":
        return GoalToken.GO_STRAIGHT
    if m == "CREEPING" or m == "STATIONARY":
        return GoalToken.GO_STRAIGHT # Or keep_lane
        
    # Default
    return GoalToken.GO_STRAIGHT

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return
        
    print(f"Reading {INPUT_CSV}...")
    rows_out = []
    
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in tqdm(reader, desc="Processing"):
            maneuver = row.get('maneuver_type', '')
            action = row.get('action_token', '')
            
            # Derive goal from maneuver
            goal = maneuver_to_goal(maneuver, action)
            
            # Update row
            row['goal_token'] = int(goal)
            row['goal_name'] = goal.name
            
            rows_out.append(row)
    
    print(f"Writing {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
        
    print(f"Done. Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
