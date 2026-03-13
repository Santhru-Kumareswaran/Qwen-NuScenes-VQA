import argparse
import json
import os
import time
from pathlib import Path
from nuscenes.nuscenes import NuScenes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.append("/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

# Custom Modules
from trajectory_dataset import TrajectoryRefinementDataset
from trajectory_model import TrajectoryRefinementModel
from visualization import create_bev_visualization

# Monkeypatch for OpenPCDet compatibility
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool

def compute_trajectory_metrics(pred_traj, gt_traj):
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu().numpy()
    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.cpu().numpy()
    
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    if min_len == 0:
        return {'ade': 0.0, 'fde': 0.0, 'r2': 0.0, 'nade': 0.0}
    
    displacements = np.sqrt(np.sum((pred_traj - gt_traj) ** 2, axis=1))
    ade = np.mean(displacements)
    fde = displacements[-1]
    
    ss_res = np.sum((pred_traj - gt_traj) ** 2)
    gt_mean = np.mean(gt_traj, axis=0)
    ss_tot = np.sum((gt_traj - gt_mean) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    r2 = max(0.0, min(1.0, r2))
    
    if min_len > 1:
        path_diffs = np.diff(gt_traj, axis=0)
        path_length = np.sum(np.sqrt(np.sum(path_diffs ** 2, axis=1)))
    else:
        path_length = 1.0
    # Clamp path_length to min 1.0m to avoid explosion for stationary samples
    path_length = max(path_length, 1.0)
    nade = ade / path_length
    
    return {
        'ade': float(ade),
        'fde': float(fde),
        'r2': float(r2),
        'nade': float(nade)
    }

def run_inference(model, loader, device, output_dir, num_samples=100):
    model.eval()
    vis_dir = Path(output_dir) / 'inference_visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    all_metrics = {'ade': [], 'fde': [], 'r2': [], 'nade': []}
    
    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="Running Inference"):
            if count >= num_samples: break
            
            # Move only necessary tensor inputs
            batch_dict_cuda = {}
            for k, v in batch_dict.items():
                if k in ['frame_id', 'action_token'] or k.startswith('split_'): 
                    batch_dict_cuda[k] = v
                    continue
                if isinstance(v, np.ndarray):
                    batch_dict_cuda[k] = torch.from_numpy(v).float().to(device)
                elif torch.is_tensor(v):
                    batch_dict_cuda[k] = v.to(device)
                else:
                    batch_dict_cuda[k] = v
            
            ret_dict = model(batch_dict_cuda)
            preds = ret_dict['predicted_deltas'].cpu()
            
            bs = preds.shape[0]
            for i in range(bs):
                if count >= num_samples: break
                
                # Extract Sample Data
                batch_idx_col = batch_dict['points'][:, 0]
                sample_points = batch_dict['points'][batch_idx_col == i][:, 1:]
                
                if 'split_ref_trajectory' in batch_dict:
                     ref_traj = batch_dict['split_ref_trajectory'][i]
                     if isinstance(ref_traj, torch.Tensor): ref_traj = ref_traj.cpu().numpy()
                else:
                     ref_traj = batch_dict['batch_ref_trajectory'][i].cpu()
                     mask = batch_dict['batch_target_mask'][i].cpu()
                     valid_len = int(mask.sum().item())
                     ref_traj = ref_traj[:valid_len].numpy()

                curr_len = len(ref_traj)
                pred_delta = preds[i][:curr_len].numpy()
                pred_traj = ref_traj + pred_delta
                
                if 'split_gt_trajectory' in batch_dict:
                    gt_traj = batch_dict['split_gt_trajectory'][i]
                    if isinstance(gt_traj, torch.Tensor): gt_traj = gt_traj.cpu().numpy()
                else:
                    gt_delta = batch_dict['batch_gt_delta'][i].cpu()[:curr_len]
                    gt_traj = ref_traj + gt_delta.numpy()
                
                # Compute Metrics
                metrics = compute_trajectory_metrics(pred_traj, gt_traj)
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                # Visualize
                sample_token = batch_dict['frame_id'][i]
                action_text = batch_dict['action_token'][i] if 'action_token' in batch_dict else "Unknown"
                fname = f'pred_{sample_token}.png'
                
                create_bev_visualization(
                    points=sample_points,
                    pred_boxes=None,
                    pred_deltas=pred_delta,
                    ref_traj=ref_traj,
                    output_path=vis_dir / fname,
                    gt_traj=gt_traj,
                    action_token=action_text,
                    **metrics
                )
                
                count += 1
    
    # Summary
    metrics_summary = {}
    if all_metrics['ade']:
        print("\n" + "="*30)
        print(f"Inference Results ({count} samples)")
        print("="*30)
        for k, v in all_metrics.items():
            avg_val = np.mean(v)
            metrics_summary[k] = float(avg_val)
            print(f"{k.upper():<5}: {avg_val:.4f}")
        print("="*30)
        print(f"Visualizations saved to: {vis_dir}")
        
        # Save metrics to JSON
        metrics_file = Path(output_dir) / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")
    
    return metrics_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to run')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'], help='Dataset split to use')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    cfg_from_yaml_file('/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml', cfg)
    
    output_dir = Path(config['files']['output_dir']) / 'inference_run'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = common_utils.create_logger()
    
    # Initialize NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=config['files']['nuscenes_root'], verbose=False)
    
    ext_ref_args = {
        'csv_path': config['files']['csv_path'],
        'json_path': config['files']['json_path'],
        'nuscenes_root': config['files']['nuscenes_root'],
        'max_samples': 0, # Load all, we'll subset
        'nusc_obj': nusc
    }
    
    dataset = TrajectoryRefinementDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False, # Use test/val mode
        logger=logger,
        ext_ref_args=ext_ref_args
    )
    
    # Use standardized split logic
    print(f"Validating samples and applying {args.split} split...")
    valid_indices = dataset.validate_dataset(logger=logger)
    validated_dataset = Subset(dataset, valid_indices)
    
    total_size = len(validated_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        validated_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    if args.split == 'train':
        inference_set = train_set
    elif args.split == 'val':
        inference_set = val_set
    elif args.split == 'test':
        inference_set = test_set
    else:
        inference_set = validated_dataset
        
    # Limit samples
    if args.num_samples > 0 and args.num_samples < len(inference_set):
        indices = list(range(args.num_samples))
        inference_set = Subset(inference_set, indices)
    
    print(f"Running inference on {len(inference_set)} samples from {args.split} split.")
    
    loader = DataLoader(
        inference_set, 
        batch_size=config['eval']['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=dataset.collate_batch
    )
    
    # Model
    model = TrajectoryRefinementModel(
        model_cfg=cfg.MODEL, 
        num_class=len(cfg.CLASS_NAMES), 
        dataset=dataset, 
        bev_dim=config['model']['num_bev_features'],
        traj_hidden_dim=config['model'].get('traj_hidden_dim', 128),
        fusion_hidden_dim=config['model'].get('fusion_hidden_dim', 256),
        prediction_horizon=config['model'].get('prediction_horizon', 12)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    run_inference(model, loader, device, output_dir, num_samples=args.num_samples)

if __name__ == '__main__':
    main()
