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
np.int = int  # Hack to fix spconv/pcdet compatibility with numpy >= 1.24

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

    def save(self, model, optimizer, scheduler, epoch, step, val_loss=None, is_best=False, scaler=None):
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_loss': val_loss
        }
        latest_path = self.ckpt_dir / f'checkpoint_epoch_{epoch}_step_{step}.pth'
        torch.save(state, latest_path)

        ckpts = sorted(list(self.ckpt_dir.glob('checkpoint_epoch_*_step_*.pth')), key=os.path.getmtime)
        while len(ckpts) > self.max_to_keep:
            os.remove(ckpts.pop(0))

        if is_best:
            best_path = self.ckpt_dir / 'checkpoint_best.pth'
            torch.save(state, best_path)
            print(f"Saved new best model with loss {val_loss:.4f}")

    def load_latest(self):
        ckpts = sorted(list(self.ckpt_dir.glob('checkpoint_epoch_*_step_*.pth')), key=os.path.getmtime)
        if not ckpts:
            return None
        return ckpts[-1]

    def load_best(self):
        best_path = self.ckpt_dir / 'checkpoint_best.pth'
        if best_path.exists():
            return best_path
        return None


class LossPlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        # Step-level buffers (for smooth train loss curve)
        self.train_losses = []
        self.steps = []
        # Val metrics tracked at each eval-step
        self.val_metrics = {'step': [], 'loss': [], 'ade': [], 'fde': [], 'r2': [], 'nade': []}
        # Epoch-level buffers
        self.epoch_nums = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_ade = []
        self.epoch_fde = []
        self.epoch_r2 = []
        self.epoch_nade = []

    def update(self, step, train_loss):
        """Called at every log_interval step — just buffer, no auto-plot."""
        self.steps.append(step)
        self.train_losses.append(train_loss)

    def update_val(self, step, metrics_dict):
        """Called at every eval_interval step — just buffer, no auto-plot."""
        self.val_metrics['step'].append(step)
        for k, v in metrics_dict.items():
            if k in self.val_metrics:
                self.val_metrics[k].append(v)

    def update_epoch(self, epoch, avg_train_loss, metrics_dict):
        """Called once per epoch to record epoch-level summary."""
        self.epoch_nums.append(epoch)
        self.epoch_train_losses.append(avg_train_loss)
        self.epoch_val_losses.append(metrics_dict.get('loss', float('nan')))
        self.epoch_ade.append(metrics_dict.get('ade', float('nan')))
        self.epoch_fde.append(metrics_dict.get('fde', float('nan')))
        self.epoch_r2.append(metrics_dict.get('r2', float('nan')))
        self.epoch_nade.append(metrics_dict.get('nade', float('nan')))

    def plot(self):
        """Save all plots.  Called only at epoch end."""
        # ── 1. Step-level train + val loss ──────────────────────────────────
        plt.figure(figsize=(12, 5))
        if self.train_losses:
            plt.plot(self.steps, self.train_losses, alpha=0.4, color='steelblue', label='Train Loss (step)')
        if self.val_metrics['loss']:
            plt.plot(self.val_metrics['step'], self.val_metrics['loss'],
                     'r-o', linewidth=2, markersize=4, label='Val Loss (eval step)')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss (MSE)')
        plt.title('Step-Level Training Progress — Experiment 5')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curve_steps.png', dpi=150)
        plt.close()

        # ── 2. Epoch-level train + val loss ─────────────────────────────────
        if self.epoch_nums:
            plt.figure(figsize=(10, 5))
            plt.plot(self.epoch_nums, self.epoch_train_losses, 'b-o', label='Train Loss', markersize=5)
            plt.plot(self.epoch_nums, self.epoch_val_losses, 'r-o', label='Val Loss', markersize=5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title('Epoch-Level Loss — Experiment 5')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'loss_curve_epochs.png', dpi=150)
            plt.close()

            # ── 3. Epoch-level ADE / FDE ────────────────────────────────────
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            pairs = [
                (axes[0, 0], self.epoch_ade,  'ADE (m)',  'tab:blue'),
                (axes[0, 1], self.epoch_fde,  'FDE (m)',  'tab:orange'),
                (axes[1, 0], self.epoch_r2,   'R²',       'tab:green'),
                (axes[1, 1], self.epoch_nade, 'nADE',     'tab:red'),
            ]
            for ax, data, ylabel, color in pairs:
                ax.plot(self.epoch_nums, data, 'o-', color=color, markersize=5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.set_title(ylabel)
                ax.grid(True, alpha=0.3)
            fig.suptitle('Epoch-Level Val Metrics — Experiment 5', fontsize=13)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'val_metrics_epochs.png', dpi=150)
            plt.close()


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


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
    nade = ade / (path_length + 1e-8)

    return {'ade': float(ade), 'fde': float(fde), 'r2': float(r2), 'nade': float(nade)}


def move_batch_to_device(batch_dict, device):
    """Move all tensor/ndarray values in batch_dict to device, skip special keys."""
    result = {}
    for k, v in batch_dict.items():
        if k in ['frame_id', 'action_token'] or k.startswith('split_'):
            result[k] = v
            continue
        if k == 'past_lidar_frames_batched':
            # List of tensors — move each to device
            result[k] = [t.to(device) for t in v]
            continue
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).float().to(device)
        elif torch.is_tensor(v):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_metrics = {'ade': 0.0, 'fde': 0.0, 'r2': 0.0, 'nade': 0.0}
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch_dict in val_loader:
            batch_dict = move_batch_to_device(batch_dict, device)

            ret_dict = model(batch_dict)
            preds = ret_dict['predicted_deltas']

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

            bs = preds.shape[0]
            batch_ade, batch_fde, batch_r2, batch_nade = [], [], [], []

            ref_traj = batch_dict['batch_ref_trajectory'][:, :min_len].cpu()
            pred_deltas_cpu = preds.cpu()
            target_deltas_cpu = targets.cpu()

            for i in range(bs):
                valid_len = int(mask[i].sum().item())
                if valid_len == 0:
                    valid_len = 1

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


def run_inference_sampling(model, test_loader, device, output_dir, num_samples=5, epoch=None):
    model.eval()
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    all_ades, all_fdes, all_r2s, all_nades = [], [], [], []

    with torch.no_grad():
        for batch_dict in test_loader:
            if count >= num_samples:
                break

            batch_dict_cuda = move_batch_to_device(batch_dict, device)

            ret_dict = model(batch_dict_cuda)
            preds = ret_dict['predicted_deltas'].cpu()

            bs = preds.shape[0]
            for i in range(bs):
                if count >= num_samples:
                    break

                batch_idx_col = batch_dict['points'][:, 0]
                sample_points = batch_dict['points'][batch_idx_col == i][:, 1:]

                if 'split_ref_trajectory' in batch_dict:
                    ref_traj = batch_dict['split_ref_trajectory'][i]
                    if isinstance(ref_traj, torch.Tensor):
                        ref_traj = ref_traj.cpu().numpy()
                else:
                    ref_traj = batch_dict['batch_ref_trajectory'][i].cpu()
                    mask = batch_dict['batch_target_mask'][i].cpu()
                    valid_len = int(mask.sum().item())
                    ref_traj = ref_traj[:valid_len]

                curr_len = len(ref_traj)
                pred_delta = preds[i][:curr_len].numpy()
                pred_traj = ref_traj + pred_delta

                if 'split_gt_trajectory' in batch_dict:
                    gt_traj_plot = batch_dict['split_gt_trajectory'][i]
                    if isinstance(gt_traj_plot, torch.Tensor):
                        gt_traj_plot = gt_traj_plot.cpu().numpy()
                else:
                    gt_delta = batch_dict['batch_gt_delta'][i].cpu()[:curr_len]
                    gt_traj_plot = ref_traj + gt_delta.numpy()

                metrics = compute_trajectory_metrics(pred_traj, gt_traj_plot)
                all_ades.append(metrics['ade'])
                all_fdes.append(metrics['fde'])
                all_r2s.append(metrics['r2'])
                all_nades.append(metrics['nade'])

                sample_token = batch_dict['frame_id'][i]
                action_text = batch_dict['action_token'][i] if 'action_token' in batch_dict else "Unknown"

                # Extract past frames for this sample
                past_frames_list = []
                if 'past_lidar_frames_batched' in batch_dict:
                    # past_lidar_frames_batched is a list of T tensors, each (N_total, 6)
                    # We need to filter for batch index i
                    for t_tensor in batch_dict['past_lidar_frames_batched']:
                        # The 0-th column is batch_idx
                        mask = (t_tensor[:, 0] == i)
                        sample_past_pts = t_tensor[mask]
                        past_frames_list.append(sample_past_pts)

                # 1. Comprehensive Visualization (with past frames)
                fname_comp = f'epoch_{epoch}_pred_{sample_token}_comprehensive.png' if epoch is not None else f'pred_{sample_token}_comprehensive.png'
                create_bev_visualization(
                    points=sample_points,
                    pred_boxes=None,
                    pred_deltas=pred_delta,
                    ref_traj=ref_traj,
                    output_path=vis_dir / fname_comp,
                    gt_traj=gt_traj_plot,
                    action_token=action_text,
                    ade=metrics['ade'],
                    fde=metrics['fde'],
                    r2=metrics['r2'],
                    nade=metrics['nade'],
                    past_frames=past_frames_list
                )

                # 2. Baseline Visualization (no past frames)
                fname_base = f'epoch_{epoch}_pred_{sample_token}_baseline.png' if epoch is not None else f'pred_{sample_token}_baseline.png'
                create_bev_visualization(
                    points=sample_points,
                    pred_boxes=None,
                    pred_deltas=pred_delta,
                    ref_traj=ref_traj,
                    output_path=vis_dir / fname_base,
                    gt_traj=gt_traj_plot,
                    action_token=action_text,
                    ade=metrics['ade'],
                    fde=metrics['fde'],
                    r2=metrics['r2'],
                    nade=metrics['nade'],
                    past_frames=None  # Force baseline style
                )
                count += 1

    if all_ades:
        avg_ade = np.mean(all_ades)
        avg_fde = np.mean(all_fdes)
        avg_r2 = np.mean(all_r2s)
        avg_nade = np.mean(all_nades)
        print(f"Epoch {epoch} | Samples: {count} | ADE: {avg_ade:.3f}m | FDE: {avg_fde:.3f}m | R²: {avg_r2:.3f} | nADE: {avg_nade:.3f}")

    print(f"Saved {count} visualizations to {vis_dir} for epoch {epoch}")


def main():
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        config['eval']['batch_size'] = args.batch_size
    if args.workers:
        config['train']['workers'] = args.workers

    cfg_from_yaml_file(
        '/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml',
        cfg
    )

    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    output_base_dir = Path(config['files']['output_dir'])
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if config['train'].get('resume', False):
        existing_dirs = sorted([d for d in output_base_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
        if existing_dirs:
            output_dir = existing_dirs[-1]
            print(f"Resuming in existing experiment directory: {output_dir}")
        else:
            output_dir = output_base_dir / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving experiment to: {output_dir}")

    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(log_file=output_dir / 'logs' / 'log.txt')

    logger.info("Initializing NuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=config['files']['nuscenes_root'], verbose=True)

    num_past_sweeps = config['model'].get('num_past_sweeps', 5)

    ext_ref_args = {
        'csv_path': config['files']['csv_path'],
        'json_path': config['files']['json_path'],
        'nuscenes_root': config['files']['nuscenes_root'],
        'max_samples': config['train'].get('max_samples'),
        'nusc_obj': nusc,
        'num_past_sweeps': num_past_sweeps,
    }

    full_dataset = TrajectoryRefinementDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,
        logger=logger,
        ext_ref_args=ext_ref_args
    )

    logger.info("Running dataset validation...")
    valid_indices = full_dataset.validate_dataset(logger=logger)

    if len(valid_indices) == 0:
        logger.error("No valid samples found!")
        return

    if len(valid_indices) < len(full_dataset):
        logger.info(f"Using {len(valid_indices)} valid samples out of {len(full_dataset)} total.")
        validated_dataset = Subset(full_dataset, valid_indices)
    else:
        validated_dataset = full_dataset

    total_size = len(validated_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        validated_dataset, [train_size, val_size, test_size], generator=generator
    )
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    train_indices = train_set.indices
    final_train_indices = [validated_dataset.indices[i] for i in train_indices]
    train_weights = full_dataset.get_sample_weights(final_train_indices)
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        shuffle=False,
        num_workers=config['train']['workers'],
        prefetch_factor=config['train'].get('prefetch_factor', 2),
        collate_fn=full_dataset.collate_batch,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config['eval']['batch_size'], shuffle=False,
        num_workers=config['train']['workers'],
        prefetch_factor=config['train'].get('prefetch_factor', 2),
        collate_fn=full_dataset.collate_batch,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config['eval']['batch_size'], shuffle=True,
        num_workers=config['train']['workers'],
        prefetch_factor=config['train'].get('prefetch_factor', 2),
        collate_fn=full_dataset.collate_batch,
        pin_memory=True,
    )

    # Model — note num_past_sweeps passed here
    model = TrajectoryRefinementModel(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=full_dataset,
        bev_dim=config['model']['num_bev_features'],
        traj_hidden_dim=config['model'].get('traj_hidden_dim', 128),
        fusion_hidden_dim=config['model'].get('fusion_hidden_dim', 256),
        prediction_horizon=config['model'].get('prediction_horizon', 12),
        num_past_sweeps=num_past_sweeps,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load Pretrained Backbone
    ckpt_path = config['model']['pretrained_backbone']
    if os.path.exists(ckpt_path):
        model.lidar_backbone.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['train']['learning_rate'],
        epochs=config['train']['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    ckpt_dir = output_dir / 'checkpoints'
    plot_dir = output_dir / 'plots'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    ckpt_manager = CheckpointManager(ckpt_dir, max_to_keep=config['train']['save_top_k'])
    plotter = LossPlotter(plot_dir)

    start_epoch = 0
    global_step = 0
    eval_interval = config['train']['eval_interval_steps']

    # Early Stopping Setup
    patience = config['train'].get('early_stopping_patience', 20)
    early_stopper = EarlyStopping(patience=patience, verbose=True)

    if config['train'].get('resume', False):
        resume_mode = config['train'].get('resume_mode', 'latest')
        ckpt_path = ckpt_manager.load_best() if resume_mode == 'best' else ckpt_manager.load_latest()

        if ckpt_path:
            logger.info(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['step']
            if checkpoint.get('val_loss') is not None:
                ckpt_manager.best_loss = min(ckpt_manager.best_loss, checkpoint['val_loss'])
                early_stopper.best_loss = ckpt_manager.best_loss
            logger.info(f"Resumed from Epoch {checkpoint['epoch']}, Step {global_step}")
        else:
            logger.warning("Resume requested but no checkpoint found. Starting from scratch.")

    logger.info(f"Starting Training: {config['train']['epochs']} Epochs (Start: {start_epoch})")
    logger.info(f"Experiment 5: Using {num_past_sweeps} past LiDAR keyframes")
    logger.info(f"Fused BEV dim: {(1 + num_past_sweeps) * config['model']['num_bev_features']}")
    logger.info(f"Early Stopping Patience: {patience} epochs")

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["train"]["epochs"]}')

        epoch_val_loss = None     # Track val loss for this epoch
        epoch_batch_losses = []   # Collect train losses per log step

        for batch_dict in pbar:
            batch_dict = move_batch_to_device(batch_dict, device)

            optimizer.zero_grad()
            ret_dict = model(batch_dict)

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

            if global_step % config['train']['log_interval'] == 0:
                plotter.update(global_step, current_loss)
                pbar.set_postfix({'loss': current_loss})
                epoch_batch_losses.append(current_loss)

        scheduler.step()

        # ── Epoch-end evaluation ─────────────────────────────────────────────
        epoch_metrics = evaluate(model, val_loader, device)
        epoch_val_loss = epoch_metrics['loss']
        plotter.update_val(global_step, epoch_metrics)

        # ── Save best model once per epoch ──────────────────────────────────
        is_best = epoch_val_loss < ckpt_manager.best_loss
        if is_best:
            ckpt_manager.best_loss = epoch_val_loss
            logger.info(f"*** New Best Model at Epoch {epoch+1}! (Val Loss: {epoch_val_loss:.4f}) ***")
        ckpt_manager.save(model, optimizer, scheduler, epoch, global_step, epoch_val_loss, is_best)

        # ── Epoch-level plot data ────────────────────────────────────────────
        avg_train_loss = float(np.mean(epoch_batch_losses)) if epoch_batch_losses else 0.0
        plotter.update_epoch(epoch + 1, avg_train_loss, epoch_metrics)

        # ── Visualize samples & plots (epoch end only) ───────────────────────
        run_inference_sampling(
            model, test_loader, device, output_dir,
            num_samples=config['eval']['test_samples'], epoch=epoch + 1
        )
        plotter.plot()

        # ── Log epoch summary ────────────────────────────────────────────────
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.3f} | "
            f"Val Loss: {epoch_val_loss:.3f} | "
            f"ADE: {epoch_metrics.get('ade', 0):.3f} | FDE: {epoch_metrics.get('fde', 0):.3f}"
        )

        # ── Early Stopping ───────────────────────────────────────────────────
        early_stopper(epoch_val_loss, model)
        if early_stopper.early_stop:
            logger.info("Early stopping triggered! Training stopped.")
            break


if __name__ == '__main__':
    main()
