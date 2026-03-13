
import pandas as pd
import numpy as np
import json
from scipy.interpolate import interp1d

def load_data(filepath):
    """Load the NuScenes CSV."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def parse_waypoints(waypoints_str):
    try:
        return np.array(json.loads(waypoints_str))
    except:
        return np.zeros((0, 2))

def transform_to_path_frame(waypoints):
    """
    Transforms waypoints to a frame where the trajectory starts at (0,0)
    and the initial tangent aligns with the +X axis.
    """
    if len(waypoints) < 2:
        return waypoints

    # Translate start to origin
    start_pos = waypoints[0]
    centered_waypoints = waypoints - start_pos
    
    # Robust Initial Heading: Find first point at least 0.5m away
    start_yaw = 0.0
    found_valid_start = False
    for i in range(1, len(centered_waypoints)):
        dist = np.linalg.norm(centered_waypoints[i])
        if dist > 0.5: 
            start_yaw = np.arctan2(centered_waypoints[i, 1], centered_waypoints[i, 0])
            found_valid_start = True
            break
            
    if not found_valid_start:
         start_yaw = np.arctan2(centered_waypoints[-1, 1], centered_waypoints[-1, 0])
    
    # Rotate
    c, s = np.cos(-start_yaw), np.sin(-start_yaw)
    R = np.array([[c, -s], [s, c]])
    
    path_waypoints = centered_waypoints @ R.T
    return path_waypoints

def resample_trajectory(points, num_points):
    """Resample a trajectory to a fixed number of points via linear interpolation."""
    if len(points) < 2:
        return np.zeros((num_points, 2))
        
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    
    if total_len < 1e-3:
        return np.tile(points[0], (num_points, 1))

    new_dists = np.linspace(0, total_len, num_points)
    
    # Linear interp
    fx = interp1d(cum_dists, points[:, 0], kind='linear')
    fy = interp1d(cum_dists, points[:, 1], kind='linear')
    
    return np.column_stack((fx(new_dists), fy(new_dists)))

def extract_features(df, target_points=10):
    """Extract normalized, resampled features from dataframe."""
    import src.config as cfg
    
    features = []
    valid_indices = []
    normalized_paths = []
    
    print("Extracting features (Transformation & Resampling)...")
    for idx, row in df.iterrows():
        waypoints = parse_waypoints(row['waypoints'])
        if len(waypoints) < 2: continue
        
        norm_way = transform_to_path_frame(waypoints)
        resampled = resample_trajectory(norm_way, target_points)
        
        features.append(resampled.flatten())
        normalized_paths.append(resampled) # Store shape (10, 2)
        valid_indices.append(idx)
        
    return np.array(features), valid_indices, np.array(normalized_paths)
