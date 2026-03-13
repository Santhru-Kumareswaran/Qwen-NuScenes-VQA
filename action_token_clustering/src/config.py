
# Configuration for Trajectory Analysis

# File Paths
input_file = '/home/jarvis/QWEN_VL_AD/nuscenes_goal_tokens_8seconds.csv'
output_dir = '/home/jarvis/QWEN_VL_AD/output'
plots_dir = '/home/jarvis/QWEN_VL_AD/plots'

# Parameters
target_points = 10  # Resample trajectories to this many points
stationary_threshold = 0.1  # Meters (truly stopped, only GPS noise)
creeping_threshold = 5.0    # Meters (slow movement)

# Clustering Configuration (Simplified: K=2 for turns)
cluster_config = {
    'LEFT': 2,
    'RIGHT': 2,
    'STRAIGHT': 2,
    'U-TURN_LEFT': 2,
    'U-TURN_RIGHT': 2
}
