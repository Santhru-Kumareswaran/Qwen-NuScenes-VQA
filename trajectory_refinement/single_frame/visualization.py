import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import torch

def create_bev_visualization(points, pred_boxes, pred_deltas, ref_traj, 
                            output_path, point_cloud_range=None, gt_traj=None, action_token=None,
                            ade=None, fde=None, r2=None, nade=None):
    """
    Create BEV visualization with LiDAR, Detections, and Trajectories.
    
    Args:
        points: (N, 5) LiDAR points [x, y, z, intensity, timestamp]
        pred_boxes: (M, 7) Detected boxes [x, y, z, dx, dy, dz, heading]
        pred_deltas: (T, 2) Predicted deltas [dx, dy]
        ref_traj: (T, 2) Reference trajectory points [x, y]
        output_path: Path to save image
        point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        gt_traj: (T, 2) Ground Truth trajectory (optional)
        action_token: str - Action description (optional)
        ade: float - Average Displacement Error in meters (optional)
        fde: float - Final Displacement Error in meters (optional)
        r2: float - R² Score, coefficient of determination (optional)
        nade: float - Normalized ADE, error as fraction of path length (optional)
    """
    if point_cloud_range is None:
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if action_token:
        ax.set_title(f"Action: {action_token}")
    else:
        ax.set_title("Trajectory Refinement Prediction")
    
    # ROTATION TRANSFORMATION: Data (X=Forward) -> Plot (Y=Forward/Up)
    # x_plot = -y_data
    # y_plot = x_data
    
    # 1. Plot LiDAR
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
        
    mask = (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] <= point_cloud_range[3]) & \
           (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] <= point_cloud_range[4])
    points_bev = points[mask]
    
    # Rotate Points
    ax.scatter(-points_bev[:, 1], points_bev[:, 0], s=0.5, c='gray', alpha=0.3)
    
    # 2. Plot Ego Vehicle (Red)
    # Box is defined as Width (X) x Length (Y).
    # Since we rotated data to Y=Forward coverage, the Box (already Y=Forward oriented) is correct!
    ego_length, ego_width = 4.5, 2.0
    ego_rect = Rectangle((-ego_width/2, -ego_length/2), ego_width, ego_length,
                         linewidth=2, edgecolor='red', facecolor='red', alpha=0.5,
                         label='Ego Vehicle')
    ax.add_patch(ego_rect)
    
    # 3. Plot Detected Boxes
    if pred_boxes is not None:
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
            
        for box in pred_boxes:
            x, y, z, dx, dy, dz, heading = box[:7]
            
            # Original heading: 0 = X+ (Forward)
            # Rotated heading: 90 deg offset?
            # Let's rotate center and heading.
            # x_rot = -y
            # y_rot = x
            # heading_rot = heading + pi/2
            
            x_rot, y_rot = -y, x
            heading_rot = heading + np.pi / 2
            
            corners = get_box_corners_2d(x_rot, y_rot, dx, dy, heading_rot)
            # Wait, get_box_corners_2d uses heading to make corners relative to x_rot, y_rot.
            # If we pass rotated params, standard corner gen works? 
            # Standard: Width along X... usually box dims are (dx, dy).
            # If we rotate heading, we rotate the box orientation.
            
            box_patch = patches.Polygon(corners, linewidth=1, edgecolor='blue', facecolor='none', alpha=0.5)
            ax.add_patch(box_patch)

    # 4. Plot Trajectories (Rotate: x,y -> -y,x)
    def rotate_traj(traj):
        if traj is None: return None
        if isinstance(traj, torch.Tensor): traj = traj.cpu().numpy()
        if len(traj) == 0: return traj
        # x_new = -y, y_new = x
        return np.stack([-traj[:, 1], traj[:, 0]], axis=1)

    ref_traj_rot = rotate_traj(ref_traj)
    if ref_traj_rot is not None and len(ref_traj_rot) > 0:
        ax.plot(ref_traj_rot[:, 0], ref_traj_rot[:, 1], 'b--', linewidth=2, label='Reference')
        ax.scatter(ref_traj_rot[-1, 0], ref_traj_rot[-1, 1], c='blue', marker='x')

    # Predicted
    if pred_deltas is not None and ref_traj is not None:
        if isinstance(pred_deltas, torch.Tensor): pred_deltas = pred_deltas.cpu().numpy()
        min_len = min(len(ref_traj), len(pred_deltas))
        if min_len > 0:
            pred_traj = ref_traj[:min_len] + pred_deltas[:min_len]
            pred_traj_rot = rotate_traj(pred_traj)
            
            ax.plot(pred_traj_rot[:, 0], pred_traj_rot[:, 1], 'g-', linewidth=2, label='Predicted')
            ax.scatter(pred_traj_rot[-1, 0], pred_traj_rot[-1, 1], c='green', marker='o')
        
    # Ground Truth
    gt_traj_rot = rotate_traj(gt_traj)
    if gt_traj_rot is not None and len(gt_traj_rot) > 0:
        ax.plot(gt_traj_rot[:, 0], gt_traj_rot[:, 1], 'r:', linewidth=2, label='Ground Truth')
 
    # Settings - Rotate Limits too
    # Old: x (0,3) -> -51, 51. y (1,4) -> -51, 51.
    # New X is -Old Y. New Y is Old X.
    # xlim = [-old_ymax, -old_ymin] -> [-51.2, 51.2]
    # ylim = [old_xmin, old_xmax] -> [-51.2, 51.2]
    # Ranges are symmetric so it stays same
    ax.set_xlim(-point_cloud_range[4], -point_cloud_range[1]) 
    ax.set_ylim(point_cloud_range[0], point_cloud_range[3])
    
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    
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
        ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
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
