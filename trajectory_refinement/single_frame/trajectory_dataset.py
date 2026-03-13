import copy
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet")

from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_yaml_file

from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

class TrajectoryRefinementDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext_ref_args=None):
        """
        Args:
            dataset_cfg: OpenPCDet dataset config
            class_names: List of class names
            training: Boolean
            root_path: Root path for pcdet (usually not used directly if paths are absolute)
            logger: Logger object
            ext_ref_args: Dictionary containing paths to CSV and JSON files
                - csv_path: Path to nuscenes_action_tokens.csv
                - json_path: Path to action_token_templates.json
                - nuscenes_root: Path to NuScenes dataset root (for LiDAR)
        """
        
        # Disable GT Sampling if present (not needed/supported for simple trajectory refinement without DB)
        if hasattr(dataset_cfg, 'DATA_AUGMENTOR'):
            dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling']
            
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, 
            training=training, root_path=root_path, logger=logger
        )
        
        self.ext_ref_args = ext_ref_args
        self.csv_path = ext_ref_args.get('csv_path')
        self.json_path = ext_ref_args.get('json_path')
        self.nuscenes_root = Path(ext_ref_args.get('nuscenes_root'))
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Max Samples Limit
        max_samples = ext_ref_args.get('max_samples')
        if max_samples and max_samples > 0:
            print(f"Limiting dataset to first {max_samples} samples (Total available: {len(self.df)})")
            self.df = self.df.iloc[:max_samples]
        
        # Load JSON Templates
        with open(self.json_path, 'r') as f:
            self.action_templates = json.load(f)
            
        print(f"Loaded {len(self.df)} samples from {self.csv_path}")
        print(f"Loaded {len(self.action_templates)} action templates")

        # Initialize DataProcessor once (Fix 2.8)
        self.processor = None
        if self.dataset_cfg.get('DATA_PROCESSOR', None):
             from pcdet.datasets.processor.data_processor import DataProcessor
             self.processor = DataProcessor(
                self.dataset_cfg.DATA_PROCESSOR, 
                point_cloud_range=self.point_cloud_range, 
                training=self.training,
                num_point_features=self.point_feature_encoder.num_point_features
             )

    def __len__(self):
        return len(self.df)

    def validate_sample(self, index, verbose=False):
        """
        Validate a single sample without full data processing.
        Returns: (is_valid: bool, reason: str or None)
        """
        row = self.df.iloc[index]
        sample_token = row['sample_token']
        
        # Check 1: nusc_obj must exist (Issue 2.2)
        nusc = self.ext_ref_args.get('nusc_obj')
        if nusc is None:
            return False, "nusc_obj_missing"
        
        # Check 2: LiDAR can be loaded (Issue 2.1)
        try:
            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = nusc.get_sample_data_path(lidar_token)
            
            # Check file exists
            if not Path(lidar_path).exists():
                return False, "lidar_file_missing"
            
            # Quick check: load and verify point count
            pc = LidarPointCloud.from_file(lidar_path)
            if pc.points.shape[1] < 100:  # Less than 100 points is suspicious
                return False, "lidar_too_few_points"
                
        except Exception as e:
            if verbose:
                print(f"  [{index}] LiDAR error: {e}")
            return False, "lidar_load_error"
        
        # Check 3: Action token exists in templates (Issue 2.3)
        action_token = row.get('action_token', '')
        if not action_token or action_token not in self.action_templates:
            return False, "action_token_missing"
        
        # Check 4: Ground truth waypoints exist
        waypoints_str = row.get('waypoints', '[]')
        if pd.isna(waypoints_str) or waypoints_str == '[]':
            return False, "gt_waypoints_empty"
        
        try:
            gt_wps = np.array(json.loads(waypoints_str))
            if len(gt_wps) < 2:  # Need at least 2 waypoints
                return False, "gt_waypoints_too_short"
        except:
            return False, "gt_waypoints_parse_error"
        
        return True, None

    def validate_dataset(self, logger=None):
        """
        Validate entire dataset before training.
        Returns: list of valid indices
        """
        log = logger.info if logger else print
        
        log("=" * 60)
        log("DATASET VALIDATION")
        log("=" * 60)
        
        # Pre-check: nusc_obj
        if self.ext_ref_args.get('nusc_obj') is None:
            log("[CRITICAL] nusc_obj is None! All samples will fail.")
            log("           Pass NuScenes object via ext_ref_args['nusc_obj']")
            return []
        
        valid_indices = []
        invalid_reasons = {}
        
        log(f"Validating {len(self.df)} samples...")
        
        for i in range(len(self.df)):
            is_valid, reason = self.validate_sample(i)
            
            if is_valid:
                valid_indices.append(i)
            else:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        
        # Print Summary
        log("-" * 60)
        log(f"VALID:   {len(valid_indices)} / {len(self.df)}")
        log(f"INVALID: {len(self.df) - len(valid_indices)} / {len(self.df)}")
        log("-" * 60)
        
        if invalid_reasons:
            log("Invalid Reasons Breakdown:")
            for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
                log(f"  - {reason}: {count}")
        
        log("=" * 60)
        
        return valid_indices


    def get_sample_weights(self, indices=None):
        """
        Compute weights for per-class balanced sampling.
        Ensures that every unique action token is sampled with equal frequency.
        """
        if indices is None:
            indices = range(len(self.df))
            
        # Extract actions for the specific indices
        # Fill NaNs with a string to allow grouping
        actions = self.df.iloc[indices]['action_token'].fillna('UNKNOWN').astype(str).values
        
        # Calculate counts for each unique action
        unique_actions, counts = np.unique(actions, return_counts=True)
        class_counts = dict(zip(unique_actions, counts))
        
        # Calculate weights (1/count)
        # We can also normalize them so the mean weight is 1.0
        total_samples = len(actions)
        num_classes = len(unique_actions)
        
        # Ideal samples per class if balanced
        ideal_count = total_samples / num_classes
        
        # Class weight = ideal_count / actual_count
        class_weights = {a: ideal_count / class_counts[a] for a in unique_actions}
        
        # Map back to all samples
        weights = np.array([class_weights[a] for a in actions], dtype=np.float64)
        
        # Print summary for visibility
        print(f"Grandular Class Balancing enabled for {num_classes} action types.")
        for action in sorted(unique_actions):
            print(f"  - {action:<20}: count={class_counts[action]:<5} weight={class_weights[action]:.2f}")
        
        return torch.DoubleTensor(weights)


    def parse_waypoints(self, wp_str):
        """Parse string representation of list of lists"""
        try:
            # Handle empty brackets or NaNs
            if pd.isna(wp_str) or wp_str == '[]':
                return np.zeros((0, 2), dtype=np.float32)
            return np.array(json.loads(wp_str), dtype=np.float32)
        except Exception as e:
            # Fallback for potential simple formatting issues
            try:
                import ast
                return np.array(ast.literal_eval(wp_str), dtype=np.float32)
            except:
                print(f"Error parsing waypoints: {wp_str[:20]}... {e}")
                return np.zeros((0, 2), dtype=np.float32)

    def transform_to_ego(self, points_global, ego_x, ego_y, ego_yaw):
        """
        Transform global coordinates to ego vehicle frame
        Args:
            points_global: (N, 2) [x, y]
            ego_x, ego_y, ego_yaw: Vehicle pose
        Returns:
            points_ego: (N, 2) [x, y]
        """
        if len(points_global) == 0:
            return points_global
            
        # Translation
        points_ego = points_global - np.array([ego_x, ego_y])
        
        # Rotation (Global to Ego = Rotate by -yaw)
        # R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]] 
        # But we want to rotate the points "back", so we rotate by -yaw.
        # x' = x*cos(-yaw) - y*sin(-yaw)
        # y' = x*sin(-yaw) + y*cos(-yaw)
        
        # Standard rotation matrix for rotating points by theta
        # [ cos -sin ]
        # [ sin  cos ]
        # Using -yaw
        c = np.cos(-ego_yaw)
        s = np.sin(-ego_yaw)
        
        rotation_matrix = np.array([
            [c, -s],
            [s,  c]
        ])
        
        points_ego = points_ego @ rotation_matrix.T
        return points_ego.astype(np.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample_token = row['sample_token']
        
        # 1. Load LiDAR & Metadata (Fix 2.6: Queries consolidated)
        nusc = self.ext_ref_args.get('nusc_obj') # Passing reference to avoid reload
        
        points = None
        ego_yaw = row['ego_yaw'] # Default to CSV value
        
        if nusc:
            try:
                # fetch all nusc data once
                sample = nusc.get('sample', sample_token)
                lidar_token = sample['data']['LIDAR_TOP']
                sd_record = nusc.get('sample_data', lidar_token)
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
                
                # Update Ego Yaw from NuScenes (more accurate)
                ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
                
                # Load LiDAR
                lidar_path = nusc.get_sample_data_path(lidar_token)
                
                # Load and Transform
                pc = LidarPointCloud.from_file(lidar_path)
                pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                pc.translate(np.array(cs_record['translation']))
                
                # Reshape to (N, 4) -> (N, 5) with padding
                points = pc.points.T
                points = np.hstack([points, np.zeros((len(points), 1))])
                
            except Exception as e:
                # Issue 2.1 is now handled by validate_dataset upstream, but we keep fallback for safety
                print(f"Failed to load NuScenes data for {sample_token}: {e}")
                points = np.zeros((1, 5), dtype=np.float32)
        else:
             # Fallback/Dummy for testing without full nusc (Issue 2.2 handled upstream)
             points = np.zeros((1, 5), dtype=np.float32)

        # 2. Parse Waypoints & Pose
        past_embed_global = self.parse_waypoints(row.get('past_waypoints', '[]'))
        future_embed_global = self.parse_waypoints(row.get('waypoints', '[]'))
        
        ego_x = row['ego_x']
        ego_y = row['ego_y']
        
        # 3. Transform to Ego Frame
        past_embed_ego = self.transform_to_ego(past_embed_global, ego_x, ego_y, ego_yaw)
        gt_trajectory_ego = self.transform_to_ego(future_embed_global, ego_x, ego_y, ego_yaw)
        
        # 4. Get Reference Trajectory
        action_token = row.get('action_token', '') # e.g. "LEFT_SLIDE" or "LEFT_TURN"
        # The JSON keys form might not match perfectly, logic to match might be needed.
        # Assuming direct match or fallback.
        ref_traj_list = self.action_templates.get(action_token, [])
        if len(ref_traj_list) == 0:
            # Try to find a partial match or default
            # print(f"Warning: No template for {action_token}")
            ref_traj_ego = np.zeros((len(gt_trajectory_ego), 2), dtype=np.float32)
        else:
            ref_traj_ego = np.array(ref_traj_list, dtype=np.float32)

        # 5. Calculate Delta (GT - Ref)
        # Note: Lengths might differ. We need to handle this.
        # Usually we predict delta for fixed number of steps or matching steps.
        # Let's align to the minimum length or specific horizon (e.g. 6s / 12 steps).
        min_len = min(len(gt_trajectory_ego), len(ref_traj_ego))
        if min_len > 0:
            gt_trajectory_ego = gt_trajectory_ego[:min_len]
            ref_traj_ego = ref_traj_ego[:min_len]
            delta_xy = gt_trajectory_ego - ref_traj_ego
        else:
            delta_xy = np.zeros((0, 2), dtype=np.float32)
            
        # 6. Prepare Input Dictionary for OpenPCDet
        input_dict = {
            'points': points,
            'frame_id': sample_token,
            'action_token': action_token, # Added for visualization
            # Custom fields
            'past_waypoints': past_embed_ego, # Variable length
            'ref_trajectory': ref_traj_ego,   # Fixed/Variable length
            'gt_delta': delta_xy,
            'gt_trajectory': gt_trajectory_ego
        }

        # Use OpenPCDet's data preparation (point sampling, augmentation if training, etc.)
        # Note: We likely only want the 'VFE/Backbone' parts, so we might skip standard detection formatting
        # unless we simply use 'prepare_data' to handle Voxelization inputs.
        data_dict = self.prepare_data(data_dict=input_dict)
        
        # 7. Post-process variable lengths for Collate
        # We need to return lengths so collate_fn can pad
        data_dict['past_len'] = len(past_embed_ego)
        data_dict['ref_len'] = len(ref_traj_ego)
        
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        Custom collate to handle variable length waypoints
        """
        # 1. Separate variable length keys
        items_for_pcdet = []
        saved_items = []
        
        manual_keys = ['past_waypoints', 'ref_trajectory', 'gt_delta', 'gt_trajectory', 'action_token']
        
        for item in batch_list:
            pcdet_item = item.copy()
            saved = {}
            for k in manual_keys:
                if k in pcdet_item:
                    saved[k] = pcdet_item.pop(k)
            items_for_pcdet.append(pcdet_item)
            saved_items.append(saved)

        # 2. OpenPCDet Standard Collate
        data_dict = DatasetTemplate.collate_batch(items_for_pcdet)
        
        # 3. Manual Collate
        # Recover items
        
        # Pad 'past_waypoints'
        past_lens = [x['past_len'] for x in batch_list]
        max_past_len = max(past_lens) if past_lens else 0
        
        batch_past_wp = []
        batch_past_mask = []
        
        for i, sample in enumerate(saved_items):
            # Access lengths from original batch (scalars are in data_dict now too, stacked)
            # But simpler to use local saved list or the scalar list we built
            curr_len = past_lens[i]
            pad_len = max_past_len - curr_len
            
            wp = sample['past_waypoints']
            # Pad with zeros
            if pad_len > 0:
                wp_padded = np.pad(wp, ((0, pad_len), (0, 0)), mode='constant')
            else:
                wp_padded = wp
                
            # Mask: 1 for valid, 0 for padded
            mask = np.zeros(max_past_len, dtype=np.float32)
            mask[:curr_len] = 1.0
            
            batch_past_wp.append(wp_padded)
            batch_past_mask.append(mask)
            
        data_dict['batch_past_waypoints'] = torch.from_numpy(np.stack(batch_past_wp)).float()
        data_dict['batch_past_mask'] = torch.from_numpy(np.stack(batch_past_mask)).float()
        
        # Handle Ref Trajectory/Delta
        ref_trajs = [x['ref_trajectory'] for x in saved_items]
        gt_deltas = [x['gt_delta'] for x in saved_items]
        # For visualization sampler, we also need raw lists split by batch
        # DatasetTemplate doesn't keep list structure. We should add them to data_dict as simple lists if needed (for inference saving).
        # Tensors stack them.
        
        # Helper for saving raw lists (optional, for vis)
        data_dict['split_ref_trajectory'] = ref_trajs 
        data_dict['split_gt_trajectory'] = [x['gt_trajectory'] for x in saved_items]
        # Add action token list
        data_dict['action_token'] = [x['action_token'] for x in saved_items]
        
        # Determine max length for targets
        max_target_len = max([len(t) for t in ref_trajs]) if ref_trajs else 0
        
        batch_ref = []
        batch_delta = []
        batch_target_mask = []
        
        for i in range(len(saved_items)):
            curr_len = len(ref_trajs[i])
            pad_len = max_target_len - curr_len
            
            ref = ref_trajs[i]
            delta = gt_deltas[i]
            
            if pad_len > 0:
                ref = np.pad(ref, ((0, pad_len), (0, 0)), mode='constant')
                delta = np.pad(delta, ((0, pad_len), (0, 0)), mode='constant')
            
            mask = np.zeros(max_target_len, dtype=np.float32)
            mask[:curr_len] = 1.0
            
            batch_ref.append(ref)
            batch_delta.append(delta)
            batch_target_mask.append(mask)
            
        data_dict['batch_ref_trajectory'] = torch.from_numpy(np.stack(batch_ref)).float()
        data_dict['batch_gt_delta'] = torch.from_numpy(np.stack(batch_delta)).float()
        data_dict['batch_target_mask'] = torch.from_numpy(np.stack(batch_target_mask)).float()

        return data_dict

    def prepare_data(self, data_dict):
        """
        Use Pcdet's processing (Voxelization, etc)
        """
        # Load standard pcdet processor
        # Logic copied/adapted from DatasetTemplate
        if self.training:
            # augmentator logic if needed
            pass
            
        # Point cloud feature extraction happens in the model, 
        # but we need to run 'point_feature_encoder' and 'data_processor' if they exist in config.
        # The easiest way is to let the Model/Collate handle Voxel generation 
        # BUT OpenPCDet usually does Voxelization in the Dataset.__getitem__ -> data_processor.
        
        
        if self.processor:
             data_dict = self.processor.forward(data_dict=data_dict)
             
        return data_dict
