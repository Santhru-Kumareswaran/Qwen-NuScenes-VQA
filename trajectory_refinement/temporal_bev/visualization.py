import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import torch
import matplotlib.gridspec as gridspec
from pathlib import Path

def create_bev_visualization(points, pred_boxes, pred_deltas, ref_traj, output_path, gt_traj=None,
                             action_token=None, ade=None, fde=None, r2=None, nade=None,
                             x_lim=(-50, 50), y_lim=(-50, 50), past_frames=None):
    """
    Creates a comprehensive visualization including:
    - Main Plot: Current BEV + Trajectories
    - Bottom Row: Past LiDAR frames (if provided)
    """
    if past_frames and len(past_frames) > 0:
        # Create a figure with main plot on top and small subplots below
        num_past = len(past_frames)
        fig = plt.figure(figsize=(15, 12), facecolor='white')
        gs = gridspec.GridSpec(2, num_past, height_ratios=[3, 1])
        
        # Main plot spans all columns in first row
        ax_main = fig.add_subplot(gs[0, :])
    else:
        fig, ax_main = plt.subplots(figsize=(10, 10), facecolor='white')

    ax_main.set_facecolor('white')
    
    # --- MAIN PLOT (Current Frame) ---
    points = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
        
    # 1. Plot LiDAR
    # The original code rotated points for plotting. The new code seems to plot directly.
    # Assuming the new x_lim/y_lim are for the original coordinate system.
    # If rotation is still desired, it needs to be re-added here.
    # For now, following the diff's implied direct plotting.
    
    # Filter points within the new x_lim and y_lim
    mask = (points[:, 0] >= x_lim[0]) & (points[:, 0] <= x_lim[1]) & \
           (points[:, 1] >= y_lim[0]) & (points[:, 1] <= y_lim[1])
    points_bev = points[mask]
    
    ax_main.scatter(points_bev[:, 0], points_bev[:, 1], s=0.5, c='black', alpha=0.3)
    
    # 2. Plot Ego Vehicle (Red)
    # Box is defined as Width (X) x Length (Y).
    # Rotated 90 degrees: Length (X) x Width (Y)
    ego_length, ego_width = 4.5, 2.0
    ego_rect = Rectangle((-ego_length/2, -ego_width/2), ego_length, ego_width,
                         linewidth=2, edgecolor='red', facecolor='red', alpha=0.5,
                         label='Ego Vehicle')
    ax_main.add_patch(ego_rect)
    
    # 3. Plot Detected Boxes
    if pred_boxes is not None:
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
            
        for box in pred_boxes:
            x, y, z, dx, dy, dz, heading = box[:7]
            
            # Assuming get_box_corners_2d works with original coordinates
            corners = get_box_corners_2d(x, y, dx, dy, heading)
            
            box_patch = patches.Polygon(corners, linewidth=1, edgecolor='blue', facecolor='none', alpha=0.5)
            ax_main.add_patch(box_patch)

    # 4. Plot Trajectories (No rotation applied here, assuming original coordinates)
    def process_traj(traj):
        if traj is None: return None
        if isinstance(traj, torch.Tensor): traj = traj.cpu().numpy()
        return traj

    ref_traj_processed = process_traj(ref_traj)
    if ref_traj_processed is not None and len(ref_traj_processed) > 0:
        ax_main.plot(ref_traj_processed[:, 0], ref_traj_processed[:, 1], 'b--', linewidth=2, label='Reference')
        ax_main.scatter(ref_traj_processed[-1, 0], ref_traj_processed[-1, 1], c='blue', marker='x')

    # Predicted
    if pred_deltas is not None and ref_traj is not None:
        if isinstance(pred_deltas, torch.Tensor): pred_deltas = pred_deltas.cpu().numpy()
        min_len = min(len(ref_traj), len(pred_deltas))
        if min_len > 0:
            pred_traj = ref_traj[:min_len] + pred_deltas[:min_len]
            pred_traj_processed = process_traj(pred_traj)
            
            ax_main.plot(pred_traj_processed[:, 0], pred_traj_processed[:, 1], 'g-', linewidth=2, label='Predicted')
            ax_main.scatter(pred_traj_processed[-1, 0], pred_traj_processed[-1, 1], c='green', marker='o')
        
    # Ground Truth
    gt_traj_processed = process_traj(gt_traj)
    if gt_traj_processed is not None and len(gt_traj_processed) > 0:
        ax_main.plot(gt_traj_processed[:, 0], gt_traj_processed[:, 1], 'r:', linewidth=2, label='Ground Truth')
 
    # Settings
    ax_main.set_xlim(x_lim) 
    ax_main.set_ylim(y_lim)
    
    ax_main.set_aspect('equal')
    ax_main.legend(loc='upper left', facecolor='white', edgecolor='black', labelcolor='black')
    
    # Display metrics if provided
    metrics_parts = []
    if ade is not None:
        metrics_parts.append(f"ADE: {ade:.3f}m")
    if fde is not None:
        metrics_parts.append(f"FDE: {fde:.3f}m")
    if r2 is not None:
        metrics_parts.append(f"R²: {r2:.3f}")
    if nade is not None:
        metrics_parts.append(f"nADE: {nade:.3f}")
    
    if metrics_parts:
        metrics_text = "\n".join(metrics_parts)
        ax_main.text(x_lim[0] + 2, y_lim[0] + 9, metrics_text, color='black', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.set_aspect('equal')
    ax_main.grid(True, color='lightgray', linestyle='--', alpha=0.5)
    ax_main.set_title(f"Sample: {Path(output_path).stem} | Action: {action_token}", color='black')

    # --- PAST FRAMES (Subplots) ---
    if past_frames and len(past_frames) > 0:
        for i, pts in enumerate(past_frames):
            ax = fig.add_subplot(gs[1, i])
            ax.set_facecolor('white')
            
            pts = pts.cpu().numpy() if isinstance(pts, torch.Tensor) else pts
            # Remove batch index column if present (shape N, 6) -> (N, 5)
            if pts.shape[1] == 6:
                pts = pts[:, 1:]
                
            x = pts[:, 0]
            y = pts[:, 1]
            intensity = pts[:, 3] if pts.shape[1] > 3 else np.ones_like(x)
            
            ax.scatter(x, y, s=0.1, c=intensity, cmap='binary', alpha=0.5)
            
            # Draw ego (Rotated 90 deg)
            rect = plt.Rectangle((-2.0, -1.0), 4.0, 2.0, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"T={-(i+1)}", color='black', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def get_box_corners_2d(x, y, dx, dy, heading):
    corners = np.array([
        [-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]
    ])
    rot_mat = np.array([[np.cos(heading), -np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])
    corners = corners @ rot_mat.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners
