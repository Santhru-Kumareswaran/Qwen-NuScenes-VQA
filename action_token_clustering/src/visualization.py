
import matplotlib.pyplot as plt
import numpy as np
import os
import src.config as cfg

def plot_grid(m_type, features, labels, centroids, output_path):
    """Plot clustering result in a grid (rotated -90 for user pref)."""
    k = len(centroids)
    
    # Create simple plot
    plt.figure(figsize=(10, 10))
    
    X_reshaped = features.reshape(-1, cfg.target_points, 2)
    colors = plt.cm.jet(np.linspace(0, 1, k))
    
    # Rotation (-90) -> NO, User wants Bottom-Up (+90 deg: x->y, y->x)
    def rotate(pts):
        return np.stack([-pts[..., 1], pts[..., 0]], axis=-1)
    
    for cid in range(k):
        mask = (labels == cid)
        subset = X_reshaped[mask]
        
        # Downsample plotting
        if len(subset) > 300:
            indices = np.random.choice(len(subset), 300, replace=False)
            subset = subset[indices]
            
        subset_rot = rotate(subset)
        color = colors[cid]
        
        for traj in subset_rot:
            plt.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.1, linewidth=1)
            
        # Centroid
        cent = centroids[cid].reshape(cfg.target_points, 2)
        cent_rot = rotate(cent)
        
        plt.plot(cent_rot[:, 0], cent_rot[:, 1], color='black', lw=3, ls='--')
        plt.plot(cent_rot[:, 0], cent_rot[:, 1], color=color, lw=2, label=f'Cluster {cid}')
        
    plt.title(f"{m_type} Sub-Clusters (K={k})")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")
