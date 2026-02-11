
import numpy as np

def classify_trajectory(waypoints, stationary_thresh=1.0, creeping_thresh=5.0):
    """
    Classifies a SINGLE trajectory based on its normalized path waypoints.
    Assumes waypoints are already transformed (Start at 0,0, facing +X).
    
    Improved Logic:
    - Uses cumulative lateral displacement for LEFT/RIGHT (not just y_end)
    - Uses heading consistency check
    - Higher thresholds for STATIONARY/CREEPING
    """
    if len(waypoints) < 2:
        return 'UNKNOWN'

    x_end = waypoints[-1, 0]
    y_end = waypoints[-1, 1]
    
    # Total displacement magnitude
    total_displacement = np.linalg.norm(waypoints[-1])
    
    # Path length (arc length)
    path_length = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
    
    # 1. STATIONARY 
    if total_displacement < stationary_thresh:
        return 'STATIONARY'
        
    # 2. CREEPING
    if total_displacement < creeping_thresh:
        return 'CREEPING'

    # --- Heading Analysis ---
    # Initial heading (from first movement)
    dx_start = waypoints[1, 0] - waypoints[0, 0]
    dy_start = waypoints[1, 1] - waypoints[0, 1]
    initial_heading = np.degrees(np.arctan2(dy_start, dx_start))
    
    # Final heading
    dx_end = waypoints[-1, 0] - waypoints[-2, 0]
    dy_end = waypoints[-1, 1] - waypoints[-2, 1]
    final_heading = np.degrees(np.arctan2(dy_end, dx_end))
    
    # Heading change (normalized to -180 to 180)
    heading_change = final_heading - initial_heading
    if heading_change > 180: heading_change -= 360
    if heading_change < -180: heading_change += 360
    
    # --- Cumulative Lateral Displacement (More Robust) ---
    # Sum of signed Y movements to determine dominant direction
    y_diffs = np.diff(waypoints[:, 1])
    cumulative_left = np.sum(y_diffs[y_diffs > 0])   # Total leftward movement
    cumulative_right = np.abs(np.sum(y_diffs[y_diffs < 0]))  # Total rightward movement
    
    # Dominant direction ratio
    total_lateral = cumulative_left + cumulative_right
    if total_lateral > 0.1:
        left_ratio = cumulative_left / total_lateral
        right_ratio = cumulative_right / total_lateral
    else:
        left_ratio = right_ratio = 0.5
    
    # 3. U-TURN
    # Geometric: Ends behind start (x < -1) OR Heading change > 135°
    is_u_turn_geom = (x_end < -1.0)
    is_u_turn_angle = (abs(heading_change) > 135)
    
    if (is_u_turn_geom or is_u_turn_angle) and (x_end < 5.0):
        # Direction: Use cumulative lateral displacement
        if left_ratio > 0.6:
            return 'U-TURN_LEFT'
        elif right_ratio > 0.6:
            return 'U-TURN_RIGHT'
        else:
            # Fallback to y_end
            return 'U-TURN_LEFT' if y_end > 0 else 'U-TURN_RIGHT'

    # 4. TURNS (LEFT / RIGHT)
    # Must have significant heading change (>15°) OR lateral offset (>3m)
    # AND consistent direction (>60% of movement in one direction)
    
    is_turn = (abs(heading_change) > 15) or (abs(y_end) > 3.0)
    
    if is_turn:
        if left_ratio > 0.6:
            return 'LEFT'
        elif right_ratio > 0.6:
            return 'RIGHT'
        else:
            # Edge case: Mixed movement. Use final position.
            if y_end > 2.0:
                return 'LEFT'
            elif y_end < -2.0:
                return 'RIGHT'
    
    # 5. STRAIGHT
    return 'STRAIGHT'

def apply_classification(df, normalized_paths):
    """Apply classification to the entire dataframe."""
    print("Running Rule-Based Classification...")
    import src.config as cfg
    
    classes = []
    for path in normalized_paths:
        cls = classify_trajectory(path, cfg.stationary_threshold, cfg.creeping_threshold)
        classes.append(cls)
        
    return classes
