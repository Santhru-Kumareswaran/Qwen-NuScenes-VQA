
import numpy as np
import src.config as cfg

def calculate_stats(centroid, target_points=10):
    if centroid.ndim == 1:
        centroid = centroid.reshape(target_points, 2)
        
    length = np.sum(np.linalg.norm(np.diff(centroid, axis=0), axis=1))
    
    # Heading Change
    dx = centroid[-1, 0] - centroid[-2, 0]
    dy = centroid[-1, 1] - centroid[-2, 1]
    final_heading_deg = np.degrees(np.arctan2(dy, dx))
    y_end = centroid[-1, 1]
    
    return length, final_heading_deg, y_end

def generate_action_tokens(m_type, centroids):
    """
    Generate map of cluster_id -> action_token_name.
    Uses RELATIVE comparison between clusters to assign Sharp/Normal.
    """
    name_map = {}
    
    # Calculate stats for all centroids
    stats = []
    for c_id, centroid in enumerate(centroids):
        length, heading, y_end = calculate_stats(centroid, cfg.target_points)
        stats.append({
            'c_id': c_id,
            'length': length,
            'heading': abs(heading),  # Absolute heading change
            'y_end': abs(y_end)
        })
    
    # Sort by heading (most curved = Sharp)
    stats.sort(key=lambda x: x['heading'], reverse=True)
    
    if m_type == 'STRAIGHT':
        # Sort by length: Longer = Fast
        stats.sort(key=lambda x: x['length'], reverse=True)
        for i, s in enumerate(stats):
            if i == 0:
                name_map[s['c_id']] = "STRAIGHT_FAST"
            else:
                name_map[s['c_id']] = "STRAIGHT_SLOW"
                
    elif m_type == 'LEFT':
        # Sort by heading: Higher = actual turn, Lower = slide/drift
        for i, s in enumerate(stats):
            if i == 0:
                name_map[s['c_id']] = "LEFT_TURN"  # Sharp, intersection-like
            else:
                name_map[s['c_id']] = "LEFT_SLIDE"  # Gentle drift/lane change
                
    elif m_type == 'RIGHT':
        for i, s in enumerate(stats):
            if i == 0:
                name_map[s['c_id']] = "RIGHT_TURN"  # Sharp, intersection-like
            else:
                name_map[s['c_id']] = "RIGHT_SLIDE"  # Gentle drift/lane change
                
    elif m_type == 'U-TURN_LEFT':
        # Sort by length: Shorter = Tight
        stats.sort(key=lambda x: x['length'])
        for i, s in enumerate(stats):
            if i == 0:
                name_map[s['c_id']] = "U_TURN_LEFT_TIGHT"
            else:
                name_map[s['c_id']] = "U_TURN_LEFT_WIDE"
                
    elif m_type == 'U-TURN_RIGHT':
        stats.sort(key=lambda x: x['length'])
        for i, s in enumerate(stats):
            if i == 0:
                name_map[s['c_id']] = "U_TURN_RIGHT_TIGHT"
            else:
                name_map[s['c_id']] = "U_TURN_RIGHT_WIDE"
    else:
        # Fallback
        for s in stats:
            name_map[s['c_id']] = f"{m_type}_UNKNOWN"
        
    return name_map
