
import torch
import random
from typing import Sequence, List
from torch.utils.data import Dataset, Subset

def scene_aware_split(dataset, lengths: Sequence[int], generator: torch.Generator, nusc=None) -> List[Subset]:
    """
    Split dataset ensuring all samples from the same scene end up in the same split.
    This prevents data leakage in video/sequence datasets.
    
    Args:
        dataset: The dataset to split (must have rows with sample_token)
        lengths: Sequence of lengths for each split (must sum to close to len(dataset))
        generator: Generator for reproducibility
        nusc: NuScenes object for scene token lookup (required if scene_token not in dataset)
        
    Returns:
        List of Subsets
    """
    
    if nusc is None:
        print("[dataset] WARNING: No nusc object provided to scene_aware_split. Falling back to random_split (LEAKAGE RISK!)")
        return torch.utils.data.random_split(dataset, lengths, generator=generator)

    # 1. Group indices by scene_token
    scene_to_indices = {}
    missing_scene_tokens = 0
    
    # We need to access the underlying rows. 
    if hasattr(dataset, "rows"):
        rows = dataset.rows
    else:
        print("[dataset] WARNING: Dataset does not have .rows attribute. Falling back to random_split.")
        return torch.utils.data.random_split(dataset, lengths, generator=generator)

    print(f"[dataset] Grouping {len(rows)} samples by scene for splitting...")
    
    for idx, row in enumerate(rows):
        scene_token = row.get("scene_token")
        
        # If not in row, look up via nusc
        if not scene_token:
            sample_token = row.get("sample_token") or row.get("token")
            if sample_token:
                try:
                    sample_rec = nusc.get('sample', sample_token)
                    scene_token = sample_rec['scene_token']
                except Exception:
                    pass
        
        if not scene_token:
            scene_token = "unknown_scene"
            missing_scene_tokens += 1
            
        if scene_token not in scene_to_indices:
            scene_to_indices[scene_token] = []
        scene_to_indices[scene_token].append(idx)
        
    if missing_scene_tokens > 0:
        print(f"[dataset] WARNING: {missing_scene_tokens} samples have no resolvable scene_token (grouped under 'unknown_scene')")

    # 2. Shuffle scenes
    scenes = list(scene_to_indices.keys())
    
    # Seed
    seed = torch.randint(0, 2**32, (1,), generator=generator).item()
    rng = random.Random(seed)
    rng.shuffle(scenes)
    
    # 3. Allocate scenes to splits
    # Logic: approximate the lengths
    target_train_len = lengths[0]
    
    train_indices = []
    val_indices = []
    
    for scene in scenes:
        indices = scene_to_indices[scene]
        # logic: if adding this scene keeps us closer to target than skipping, add it
        # Simple Greedy: Fill train until target met
        if len(train_indices) < target_train_len:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
            
    print(f"[dataset] Scene-Aware Split Result: Train={len(train_indices)}, Val={len(val_indices)}")
    
    return [Subset(dataset, train_indices), Subset(dataset, val_indices)]
