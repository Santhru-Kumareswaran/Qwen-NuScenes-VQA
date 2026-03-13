
import os
import pandas as pd
import json
import numpy as np

import src.config as cfg
import src.preprocessing as prep
import src.classification as cls
import src.clustering as clust
import src.labeling as lbl
import src.visualization as vis

def main():
    # Setup
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    
    # 1. Load & Preprocess
    df = prep.load_data(cfg.input_file)
    features, valid_indices, normalized_paths = prep.extract_features(df, cfg.target_points)
    
    # 2. Rule-Based Classification
    print("Classifying trajectories...")
    maneuvers = cls.apply_classification(df.loc[valid_indices], normalized_paths)
    
    # Create working DF
    work_df = pd.DataFrame({
        'sample_token': df.loc[valid_indices, 'sample_token'].values, 
        'maneuver_type': maneuvers,
        'feature_idx': range(len(features)) # Pointer to features array
    })
    
    final_output = []
    templates = {}
    
    # 3. Sub-Clustering & Labeling
    for m_type in ['LEFT', 'RIGHT', 'STRAIGHT', 'U-TURN_LEFT', 'U-TURN_RIGHT']:
        print(f"Processing {m_type}...")
        group_df = work_df[work_df['maneuver_type'] == m_type]
        
        if len(group_df) == 0: continue
        
        # Get features
        group_features = features[group_df['feature_idx'].values]
        k = cfg.cluster_config[m_type]
        
        # Cluster
        labels, centroids, medoids = clust.cluster_group(group_features, k)
        if labels is None: continue
        
        # Naming
        name_map = lbl.generate_action_tokens(m_type, centroids)
        # Note: We still name based on Centroids (average stats are good for naming),
        # but we visualize using Medoids (better look).
        
        # Visuals
        vis.plot_grid(m_type, group_features, labels, centroids, 
                      os.path.join(cfg.plots_dir, 'clusters', f"{m_type}_clusters.png"))
        
        # Store Results (Update the main map)
        for i, idx in enumerate(group_df.index):
            c_id = labels[i]
            token_name = name_map[c_id]
            
            # Update the original DF directly (using index)
            df.loc[idx, 'maneuver_type'] = m_type
            df.loc[idx, 'action_token'] = token_name
            
        # Store Templates
        for c_id, name in name_map.items():
            templates[name] = medoids[c_id].reshape(cfg.target_points, 2).tolist()

    # 4. Handle Stationary/Creeping/Unknown
    for m_type in ['STATIONARY', 'CREEPING', 'UNKNOWN']:
        group_df = work_df[work_df['maneuver_type'] == m_type]
        print(f"Processing {m_type} ({len(group_df)} samples)...")
        
        # Update original DF
        df.loc[group_df.index, 'maneuver_type'] = m_type
        df.loc[group_df.index, 'action_token'] = m_type
            
        # Simple Templates
        if m_type == 'STATIONARY':
            templates['STATIONARY'] = [[0.0, 0.0], [0.1, 0.0]]
        elif m_type == 'CREEPING':
            templates['CREEPING'] = [[0.0, 0.0], [1.5, 0.0]]
            
    # 5. Metadata Generation
    print("Generating metadata...")
    metadata = {}
    
    # Counts
    counts = df['action_token'].value_counts().to_dict()
    
    # Organize by Type
    stats = {}
    tokens = df['action_token'].unique()
    
    for token in tokens:
        if not isinstance(token, str): continue
        count = int(counts[token])
        stats[token] = {
            'count': count,
            'description': token.replace('_', ' ').title()
        }
        
    # Save Metadata
    metadata_path = os.path.join(cfg.output_dir, 'action_token_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # 6. Save (Preserve all columns)
    print(f"Saving {len(df)} rows with full columns to {os.path.join(cfg.output_dir, 'nuscenes_action_tokens.csv')}...")
    df.to_csv(os.path.join(cfg.output_dir, 'nuscenes_action_tokens.csv'), index=False)
    
    with open(os.path.join(cfg.output_dir, 'action_token_templates.json'), 'w') as f:
        json.dump(templates, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
