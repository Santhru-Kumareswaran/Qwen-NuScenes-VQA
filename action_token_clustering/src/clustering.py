
import numpy as np
from sklearn.cluster import KMeans
import src.config as cfg

def cluster_group(features, k):
    """Run KMeans on a set of features."""
    if len(features) < k:
        return None, None
        
    # Downsample for fit if huge
    if len(features) > 5000:
        indices = np.random.choice(len(features), 5000, replace=False)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features[indices])
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)
        
    labels = kmeans.predict(features)
    centroids = kmeans.cluster_centers_
    
    # Calculate True Medoids (Point minimizing sum of distances to all others)
    # This is more robust for curves than "closest to centroid" (which biases towards inner curve).
    from sklearn.metrics import pairwise_distances
    medoids = np.zeros_like(centroids)

    for cid in range(k):
        cluster_indices = np.where(labels == cid)[0]
        if len(cluster_indices) == 0: continue
        
        cluster_points = features[cluster_indices]
        N = len(cluster_points)
        
        # If small enough, compute full distance matrix
        if N < 2000:
            D = pairwise_distances(cluster_points, cluster_points)
            # Index of point with min sum of distances
            local_medoid_idx = np.argmin(D.sum(axis=1))
            medoids[cid] = cluster_points[local_medoid_idx]
        else:
            # Approximation for large clusters:
            # Select 1000 random points and find the medoid among them
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(N, 1000, replace=False)
            sample_points = cluster_points[sample_indices]
            
            D = pairwise_distances(sample_points, sample_points)
            local_medoid_in_sample = np.argmin(D.sum(axis=1))
            
            # Map back to real data
            medoids[cid] = sample_points[local_medoid_in_sample]
    
    return labels, centroids, medoids
