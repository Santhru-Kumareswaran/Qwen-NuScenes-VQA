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
# Monkeypatch for OpenPCDet compatibility
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.append("/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network

# Custom Modules
from trajectory_dataset import TrajectoryRefinementDataset
from trajectory_model import TrajectoryRefinementModel
from visualization import create_bev_visualization

class CheckpointManager:
    def __init__(self, ckpt_dir, max_to_keep=3):
        self.ckpt_dir = Path(ckpt_dir)
        self.max_to_keep = max_to_keep
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        
    def save(self, model, optimizer, scheduler, epoch, step, val_loss=None, is_best=False):
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss
        }
        
        # Save Latest
        latest_path = self.ckpt_dir / f'checkpoint_epoch_{epoch}_step_{step}.pth'
        torch.save(state, latest_path)
        
        # Manage Latest Count
        ckpts = sorted(list(self.ckpt_dir.glob('checkpoint_epoch_*_step_*.pth')), key=os.path.getmtime)
        while len(ckpts) > self.max_to_keep:
            os.remove(ckpts.pop(0))
            
        # Save Best
        if is_best:
            best_path = self.ckpt_dir / 'checkpoint_best.pth'
            torch.save(state, best_path)
            print(f"Saved new best model with loss {val_loss:.4f}")

    def load_latest(self):
        """Find the most recent checkpoint file"""
        ckpts = sorted(list(self.ckpt_dir.glob('checkpoint_epoch_*_step_*.pth')), key=os.path.getmtime)
        if not ckpts:
            return None
        return ckpts[-1]

    def load_best(self):
        """Find the best checkpoint file"""
        best_path = self.ckpt_dir / 'checkpoint_best.pth'
        if best_path.exists():
            return best_path
        return None

class LossPlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.train_losses = []
        self.val_metrics = {'step': [], 'loss': [], 'ade': [], 'fde': [], 'r2': [], 'nade': []}
        self.steps = []
        
    def update(self, step, train_loss):
        self.steps.append(step)
        self.train_losses.append(train_loss)
        # Periodically plot training loss too (e.g. every 100 steps or handled by eval)
        # But user asked for "every time you print metrics into the log"
        # Since log_interval is 10, we could plot every 10 steps, but that might be slow.
        # Let's plot every 50 steps for train loss sanity.
        if step % 50 == 0:
            self.plot()
        
    def update_val(self, step, metrics_dict):
        """
        metrics_dict: {'loss': val, 'ade': val, ...}
        """
        self.val_metrics['step'].append(step)
        for k, v in metrics_dict.items():
            if k in self.val_metrics:
                self.val_metrics[k].append(v)
        self.plot()
        
    def plot(self):
        # 1. Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_losses, label='Train Loss', alpha=0.5)
        if self.val_metrics['loss']:
            plt.plot(self.val_metrics['step'], self.val_metrics['loss'], 'r-o', label='Val Loss', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'loss_curve.png')
        plt.close()
        
        # 2. Metric Plots
        for metric in ['ade', 'fde', 'r2', 'nade']:
            if self.val_metrics[metric]:
                plt.figure(figsize=(10, 6))
                plt.plot(self.val_metrics['step'], self.val_metrics[metric], 'b-o', label=f'Val {metric.upper()}')
                plt.xlabel('Steps')
                plt.ylabel(metric.upper())
                plt.title(f'Validation {metric.upper()}')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.output_dir / f'val_{metric}.png')
                plt.close()

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_metrics = {'ade': 0.0, 'fde': 0.0, 'r2': 0.0, 'nade': 0.0}
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_dict in val_loader:
             # Move data to device
            for key, val in batch_dict.items():
                if key == 'frame_id': continue
                if key.startswith('split_'): continue
                
                if isinstance(val, np.ndarray):
                    batch_dict[key] = torch.from_numpy(val).float().to(device)
                elif torch.is_tensor(val):
                    batch_dict[key] = val.to(device)
            
            # Forward
            ret_dict = model(batch_dict)
            preds = ret_dict['predicted_deltas'] # (B, T, 2)
            
            # --- Loss Prep ---
            targets = batch_dict['batch_gt_delta']
            mask = batch_dict['batch_target_mask']
            
            min_len = min(preds.shape[1], targets.shape[1])
            preds = preds[:, :min_len]
            targets = targets[:, :min_len]
            mask = mask[:, :min_len]
            
            squared_diff = (preds - targets) ** 2
            masked_diff = squared_diff * mask.unsqueeze(-1)
            loss = masked_diff.sum() / (mask.sum() * 2 + 1e-6)
            total_loss += loss.item()
            
            # --- Metrics ---
            # We need full trajectories for metrics
            # preds are DELTAS. Need to add to Ref Trajectory.
            # Batch dict has 'batch_ref_trajectory' (Tensor)
            bs = preds.shape[0]
            
            batch_ade = []
            batch_fde = []
            batch_r2 = []
            batch_nade = []
            
            ref_traj = batch_dict['batch_ref_trajectory'][:, :min_len].cpu()
            pred_deltas_cpu = preds.cpu()
            target_deltas_cpu = targets.cpu()
            
            # Process each sample in batch (metrics are per-sample)
            for i in range(bs):
                curr_mask = mask[i].cpu()
                valid_len = int(curr_mask.sum().item() / curr_mask.shape[-1] if curr_mask.dim() > 1 else curr_mask.sum().item())
                # Actually mask is (B, T). sum gives length.
                valid_len = int(mask[i].sum().item())
                if valid_len == 0: valid_len = 1 # Avoid crash
                
                # Slicing
                p_delta = pred_deltas_cpu[i, :valid_len]
                t_delta = target_deltas_cpu[i, :valid_len]
                ref = ref_traj[i, :valid_len]
                
                pred_path = ref + p_delta
                gt_path = ref + t_delta
                
                m = compute_trajectory_metrics(pred_path, gt_path)
                batch_ade.append(m['ade'])
                batch_fde.append(m['fde'])
                batch_r2.append(m['r2'])
                batch_nade.append(m['nade'])
                
            total_metrics['ade'] += np.mean(batch_ade)
            total_metrics['fde'] += np.mean(batch_fde)
            total_metrics['r2'] += np.mean(batch_r2)
            total_metrics['nade'] += np.mean(batch_nade)
            
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics

def compute_trajectory_metrics(pred_traj, gt_traj):
    """
    Compute trajectory prediction metrics.
    
    Args:
        pred_traj: (T, 2) Predicted trajectory points [x, y]
        gt_traj: (T, 2) Ground truth trajectory points [x, y]
        
    Returns:
        dict with keys:
            - ade: Average Displacement Error (meters)
            - fde: Final Displacement Error (meters)
            - r2: R² Score (coefficient of determination, 0-1 scale)
            - nade: Normalized ADE (error as fraction of path length, 0-1 scale)
    """
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu().numpy()
    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.cpu().numpy()
    
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    if min_len == 0:
        return {'ade': 0.0, 'fde': 0.0, 'r2': 0.0, 'nade': 0.0}
    
    # L2 displacement at each timestep
    displacements = np.sqrt(np.sum((pred_traj - gt_traj) ** 2, axis=1))
    
    # ADE: Average over all timesteps
    ade = np.mean(displacements)
    
    # FDE: Final displacement
    fde = displacements[-1]
    
    # R² Score: 1 - (SS_res / SS_tot)
    # SS_res = sum of squared residuals
    # SS_tot = total sum of squares (variance of GT)
    ss_res = np.sum((pred_traj - gt_traj) ** 2)
    gt_mean = np.mean(gt_traj, axis=0)
    ss_tot = np.sum((gt_traj - gt_mean) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    r2 = max(0.0, min(1.0, r2))  # Clamp to [0, 1] for display
    
    # Normalized ADE: ADE / total path length of GT
    # Path length = sum of distances between consecutive points
    if min_len > 1:
        path_diffs = np.diff(gt_traj, axis=0)
        path_length = np.sum(np.sqrt(np.sum(path_diffs ** 2, axis=1)))
    else:
        path_length = 1.0
    nade = ade / (path_length + 1e-8)
    
    return {
        'ade': float(ade),
        'fde': float(fde),
        'r2': float(r2),
        'nade': float(nade)
    }


def run_inference_sampling(model, test_loader, device, output_dir, num_samples=5, epoch=None):
    """Run inference on a few samples and save visualizations with ADE/FDE metrics"""
    model.eval()
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    all_ades = []
    all_fdes = []
    all_r2s = []
    all_nades = []
    
    with torch.no_grad():
        for batch_dict in test_loader:
            if count >= num_samples: break
            
            # Move only necessary tensor inputs
            # Note: Need to keep raw data for plotting (points, ref_traj)
            # Batch dict has collimated tensors. We can extract singular samples.
            
            # Run batch inference
            batch_dict_cuda = {}
            for k, v in batch_dict.items():
                if k == 'frame_id': 
                    batch_dict_cuda[k] = v
                    continue
                if k.startswith('split_'): 
                    batch_dict_cuda[k] = v
                    continue
                
                if isinstance(v, np.ndarray):
                    batch_dict_cuda[k] = torch.from_numpy(v).float().to(device)
                elif torch.is_tensor(v):
                    batch_dict_cuda[k] = v.to(device)
                else:
                    batch_dict_cuda[k] = v
            
            ret_dict = model(batch_dict_cuda)
            preds = ret_dict['predicted_deltas'].cpu() # (B, T, 2)
            
            # Iterate batch to save
            bs = preds.shape[0]
            for i in range(bs):
                if count >= num_samples: break
                
                # Get Data
                # Note: 'points' in batch_dict is (B_idx, x, y, z...) on CPU.
                # 'sample_points' extraction below correctly uses this CPU tensor.
                # Visualization happens on CPU, so using batch_dict (original) is safer/easier than moving back from GPU.
                points = batch_dict['points'][batch_dict['points'][:, 0] == i][:, 1:] # batch index is 0th col in openpcdet (B_idx, x, y, z...)
                # But our dataset returns 'points' as (N, 5). Collate likely merges them.
                # Standard pcdet collate: points -> (N_total, 6) [batch_idx, x, y, z, i, t]
                # Let's extract:
                batch_idx_col = batch_dict['points'][:, 0]
                sample_points = batch_dict['points'][batch_idx_col == i][:, 1:] # Remove batch idx
                
                if 'split_ref_trajectory' in batch_dict:
                     ref_traj = batch_dict['split_ref_trajectory'][i] # Numpy, valid len
                     if isinstance(ref_traj, torch.Tensor): ref_traj = ref_traj.cpu().numpy()
                else:
                     ref_traj = batch_dict['batch_ref_trajectory'][i].cpu()
                     mask = batch_dict['batch_target_mask'][i].cpu()
                     valid_len = int(mask.sum().item())
                     ref_traj = ref_traj[:valid_len]

                # Prediction (Delta) - THIS is a tensor, needs explicit slicing by ref length or mask
                # Preds shape: (B, T_max, 2)
                # We should slice pred_delta to match ref_traj length
                curr_len = len(ref_traj)
                pred_delta = preds[i][:curr_len].numpy()
                
                # Compute predicted trajectory (ref + delta)
                pred_traj = ref_traj + pred_delta
                
                # GT Trajectory
                # Prefer raw if available
                if 'split_gt_trajectory' in batch_dict:
                    gt_traj_plot = batch_dict['split_gt_trajectory'][i]
                    if isinstance(gt_traj_plot, torch.Tensor): gt_traj_plot = gt_traj_plot.cpu().numpy()
                else:
                    # Reconstruct or use tensor
                    gt_delta = batch_dict['batch_gt_delta'][i].cpu()[:curr_len]
                    gt_traj_plot = ref_traj + gt_delta.numpy()
                
                # Compute all metrics
                metrics = compute_trajectory_metrics(pred_traj, gt_traj_plot)
                all_ades.append(metrics['ade'])
                all_fdes.append(metrics['fde'])
                all_r2s.append(metrics['r2'])
                all_nades.append(metrics['nade'])
                
                sample_token = batch_dict['frame_id'][i]
                action_text = batch_dict['action_token'][i] if 'action_token' in batch_dict else "Unknown"
                
                # Call Visualization with metrics
                fname = f'epoch_{epoch}_pred_{sample_token}.png' if epoch is not None else f'pred_{sample_token}.png'
                
                create_bev_visualization(
                    points=sample_points,
                    pred_boxes=None, # Optional
                    pred_deltas=pred_delta,
                    ref_traj=ref_traj,
                    output_path=vis_dir / fname,
                    gt_traj=gt_traj_plot,
                    action_token=action_text,
                    ade=metrics['ade'],
                    fde=metrics['fde'],
                    r2=metrics['r2'],
                    nade=metrics['nade']
                )
                
                count += 1
    
    # Log aggregate metrics
    if all_ades:
        avg_ade = np.mean(all_ades)
        avg_fde = np.mean(all_fdes)
        avg_r2 = np.mean(all_r2s)
        avg_nade = np.mean(all_nades)
        print(f"Epoch {epoch} | Samples: {count} | ADE: {avg_ade:.3f}m | FDE: {avg_fde:.3f}m | R²: {avg_r2:.3f} | nADE: {avg_nade:.3f}")
    
    print(f"Saved {count} visualizations to {vis_dir} for epoch {epoch}")

def main():
    # Fix for RTX 5080 (sm_120) compatibility with older PyTorch
    # CuDNN RNN kernels might specific architecture support not present
    torch.backends.cudnn.enabled = False 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--workers', type=int, help='Override number of workers')
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override Config
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        config['eval']['batch_size'] = args.batch_size
    if args.workers:
        config['train']['workers'] = args.workers
        
    # Initialize cfg from yaml
    # Ideally load openpcdet cfg from file if needed for backbone
    # Assuming config['model']['pretrained_backbone'] implies a corresponding yaml or we use a default
    # Let's assume we need to load a base pcdet config. 
    # Hardcoding valid path or adding to json for now.
    # User env has 'cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml'
    cfg_from_yaml_file('/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml', cfg)
    
    # Create output dir with timestamp
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    output_base_dir = Path(config['files']['output_dir'])
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Improved Resume Logic: Find latest existing run if resume is true
    if config['train'].get('resume', False):
        existing_dirs = sorted([d for d in output_base_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
        if existing_dirs:
            # Use the latest directory instead of creating a new one
            output_dir = existing_dirs[-1]
            print(f"Resuming in existing experiment directory: {output_dir}")
        else:
            output_dir = output_base_dir / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Resume requested but no previous experiment found. Starting new: {output_dir}")
    else:
        output_dir = output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving experiment to: {output_dir}")
    
    # Create subfolders
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    logger = common_utils.create_logger(log_file=output_dir / 'logs' / 'log.txt')
    
    # Initialize NuScenes
    logger.info("Initializing NuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=config['files']['nuscenes_root'], verbose=True)
    
    # Dataset & Loaders
    ext_ref_args = {
        'csv_path': config['files']['csv_path'],
        'json_path': config['files']['json_path'],
        'nuscenes_root': config['files']['nuscenes_root'],
        'max_samples': config['train'].get('max_samples'),
        'nusc_obj': nusc
    }
    
    full_dataset = TrajectoryRefinementDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,
        logger=logger,
        ext_ref_args=ext_ref_args
    )
    
    # ===== DATASET VALIDATION =====
    # Validate samples before training (fixes issues 2.1, 2.2, 2.3)
    logger.info("Running dataset validation...")
    valid_indices = full_dataset.validate_dataset(logger=logger)
    
    if len(valid_indices) == 0:
        logger.error("No valid samples found! Cannot proceed with training.")
        return
    
    if len(valid_indices) < len(full_dataset):
        logger.info(f"Using {len(valid_indices)} valid samples out of {len(full_dataset)} total.")
        # Create subset with only valid indices
        validated_dataset = Subset(full_dataset, valid_indices)
    else:
        validated_dataset = full_dataset
    
    # Split Train/Val/Test on VALIDATED data only
    # Simple split: 80/10/10
    total_size = len(validated_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Fix 4.3: Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(validated_dataset, [train_size, val_size, test_size], generator=generator)
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create WeightedRandomSampler for Training Set
    # We need to map the Training Set indices back to the Full Dataset indices to get weights
    # random_split returns Subsets which track indices
    
    # 1. Get indices of training samples in the original/validated dataset
    train_indices = train_set.indices 
    
    # If validated_dataset is itself a Subset (which it is), we need to trace back to full_dataset
    # validated_dataset.indices maps Subset -> Full Dataset
    # train_set.indices maps Train Subset -> Validated Dataset
    
    # Map Train Subset -> Full Dataset
    final_train_indices = [validated_dataset.indices[i] for i in train_indices]
    
    # 2. Get weights for these indices
    train_weights = full_dataset.get_sample_weights(final_train_indices)
    
    # 3. Create Sampler
    # Replacement=True allows oversampling minority classes
    train_sampler = WeightedRandomSampler(
        weights=train_weights, 
        num_samples=len(train_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config['train']['batch_size'], 
        sampler=train_sampler, # Use sampler instead of shuffle
        shuffle=False,         # Must be False when sampler is used
        num_workers=config['train']['workers'], 
        prefetch_factor=config['train'].get('prefetch_factor', 2),
        collate_fn=full_dataset.collate_batch, 
        drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=config['eval']['batch_size'], shuffle=False, 
                           num_workers=config['train']['workers'], prefetch_factor=config['train'].get('prefetch_factor', 2), collate_fn=full_dataset.collate_batch)
    test_loader = DataLoader(test_set, batch_size=config['eval']['batch_size'], shuffle=True, 
                            num_workers=config['train']['workers'], prefetch_factor=config['train'].get('prefetch_factor', 2), collate_fn=full_dataset.collate_batch)
    
    # Model
    model = TrajectoryRefinementModel(
        model_cfg=cfg.MODEL, 
        num_class=len(cfg.CLASS_NAMES), 
        dataset=full_dataset, 
        bev_dim=config['model']['num_bev_features'],
        traj_hidden_dim=config['model'].get('traj_hidden_dim', 128),
        fusion_hidden_dim=config['model'].get('fusion_hidden_dim', 256),
        prediction_horizon=config['model'].get('prediction_horizon', 12)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load Pretrained
    ckpt_path = config['model']['pretrained_backbone']
    if os.path.exists(ckpt_path):
        model.lidar_backbone.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    
    # Fix 4.4 & 4.11: Use OneCycleLR for Warmup and Scheduling
    # OneCycleLR typically needs steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['train']['learning_rate'], 
        epochs=config['train']['epochs'], 
        steps_per_epoch=len(train_loader),
        pct_start=0.3, # 30% warmup
        div_factor=25, # Initial LR = max_lr / 25
        final_div_factor=1000 # Final LR = start_lr / 1000
    )
    
    # Fix 4.2: Remove unused criterion
    # criterion = nn.MSELoss(reduction='none')
    
    # Utilities
    # Subfolders
    ckpt_dir = output_dir / 'checkpoints'
    plot_dir = output_dir / 'plots'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_manager = CheckpointManager(ckpt_dir, max_to_keep=config['train']['save_top_k'])
    plotter = LossPlotter(plot_dir)
    
    start_epoch = 0
    global_step = 0
    eval_interval = config['train']['eval_interval_steps']
    
    # Resume Logic (Fix 4.6)
    if config['train'].get('resume', False):
        resume_mode = config['train'].get('resume_mode', 'latest')
        ckpt_path = None
        
        logger.info(f"Attempting to resume training (Mode: {resume_mode})...")
        
        if resume_mode == 'best':
            ckpt_path = ckpt_manager.load_best()
        else: # latest
            ckpt_path = ckpt_manager.load_latest()
            
        if ckpt_path:
            logger.info(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            start_epoch = checkpoint['epoch'] + 1 # Resume from next epoch
            # If start_epoch >= max_epochs, we might be done, or we extend? 
            # Usually resume implies continuing. If done, user should increase epochs config.
            
            global_step = checkpoint['step']
            if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
                ckpt_manager.best_loss = min(ckpt_manager.best_loss, checkpoint['val_loss'])
                
            logger.info(f"Resumed from Epoch {checkpoint['epoch']}, Step {global_step}")
        else:
            logger.warning("Resume requested but no checkpoint found. Starting from scratch.")

    logger.info(f"Starting Training: {config['train']['epochs']} Epochs (Start: {start_epoch})")
    
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["train"]["epochs"]}')
        
        for batch_dict in pbar:
            # Move to device
            for k, v in batch_dict.items():
                if k == 'frame_id': continue
                if k.startswith('split_'): continue # Lists
                
                if isinstance(v, np.ndarray):
                    batch_dict[k] = torch.from_numpy(v).float().to(device)
                elif torch.is_tensor(v):
                     batch_dict[k] = v.to(device)
            
            optimizer.zero_grad()
            ret_dict = model(batch_dict)
            
            # Loss Calculation
            preds = ret_dict['predicted_deltas']
            targets = batch_dict['batch_gt_delta']
            mask = batch_dict['batch_target_mask']
            
            min_len = min(preds.shape[1], targets.shape[1])
            preds = preds[:, :min_len]
            targets = targets[:, :min_len]
            mask = mask[:, :min_len]
            
            sq_diff = (preds - targets) ** 2
            masked_diff = sq_diff * mask.unsqueeze(-1)
            loss = masked_diff.sum() / (mask.sum() * 2 + 1e-6)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            
            global_step += 1
            current_loss = loss.item()
            
            # Logging & Plotting
            if global_step % config['train']['log_interval'] == 0:
                plotter.update(global_step, current_loss)
                pbar.set_postfix({'loss': current_loss})
                
            # Evaluation
            if global_step % eval_interval == 0:
                metrics = evaluate(model, val_loader, device)
                val_loss = metrics['loss']
                
                plotter.update_val(global_step, metrics)
                
                is_best = val_loss < ckpt_manager.best_loss
                if is_best: 
                    ckpt_manager.best_loss = val_loss
                    logger.info(f"*** New Best Model Found! (Loss: {val_loss:.4f}) Saving... ***")
                
                ckpt_manager.save(model, optimizer, scheduler, epoch, global_step, val_loss, is_best)
                model.train() # Resume train mode
        
        # Step LR Scheduler
        scheduler.step()
        
        # End of Epoch
        run_inference_sampling(model, test_loader, device, output_dir, num_samples=config['eval']['test_samples'], epoch=epoch+1)
        plotter.plot() # Ensure plot is saved

if __name__ == '__main__':
    main()
