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
        Experiment 5: Past LiDAR Sweeps
        Loads current + N past LiDAR keyframes (via sample['prev'] chain).
        Each past frame is transformed into the current ego coordinate frame.

        ext_ref_args:
            - csv_path: Path to nuscenes_action_tokens.csv
            - json_path: Path to action_token_templates.json
            - nuscenes_root: Path to NuScenes dataset root
            - nusc_obj: NuScenes object
            - num_past_sweeps: Number of past keyframes to load (default: 5)
        """
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
        self.num_past_sweeps = ext_ref_args.get('num_past_sweeps', 5)

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
        print(f"Using {self.num_past_sweeps} past LiDAR keyframes")

        # Initialize DataProcessor once
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
        """Validate a single sample. Returns (is_valid, reason)."""
        row = self.df.iloc[index]
        sample_token = row['sample_token']

        nusc = self.ext_ref_args.get('nusc_obj')
        if nusc is None:
            return False, "nusc_obj_missing"

        try:
            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = nusc.get_sample_data_path(lidar_token)

            if not Path(lidar_path).exists():
                return False, "lidar_file_missing"

            pc = LidarPointCloud.from_file(lidar_path)
            if pc.points.shape[1] < 100:
                return False, "lidar_too_few_points"

        except Exception as e:
            if verbose:
                print(f"  [{index}] LiDAR error: {e}")
            return False, "lidar_load_error"

        action_token = row.get('action_token', '')
        if not action_token or action_token not in self.action_templates:
            return False, "action_token_missing"

        waypoints_str = row.get('waypoints', '[]')
        if pd.isna(waypoints_str) or waypoints_str == '[]':
            return False, "gt_waypoints_empty"

        try:
            gt_wps = np.array(json.loads(waypoints_str))
            if len(gt_wps) < 2:
                return False, "gt_waypoints_too_short"
        except:
            return False, "gt_waypoints_parse_error"

        # Check past waypoints
        past_waypoints_str = row.get('past_waypoints', '[]')
        if pd.isna(past_waypoints_str) or past_waypoints_str == '[]':
            return False, "past_waypoints_empty"
            
        try:
            past_wps = np.array(json.loads(past_waypoints_str))
            if len(past_wps) == 0:
                return False, "past_waypoints_empty"
        except:
            return False, "past_waypoints_parse_error"

        return True, None

    def validate_dataset(self, logger=None):
        """Validate entire dataset. Returns list of valid indices."""
        log = logger.info if logger else print

        log("=" * 60)
        log("DATASET VALIDATION")
        log("=" * 60)

        if self.ext_ref_args.get('nusc_obj') is None:
            log("[CRITICAL] nusc_obj is None! All samples will fail.")
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
        """Compute weights for per-class balanced sampling."""
        if indices is None:
            indices = range(len(self.df))

        actions = self.df.iloc[indices]['action_token'].fillna('UNKNOWN').astype(str).values
        unique_actions, counts = np.unique(actions, return_counts=True)
        class_counts = dict(zip(unique_actions, counts))

        total_samples = len(actions)
        num_classes = len(unique_actions)
        ideal_count = total_samples / num_classes
        class_weights = {a: ideal_count / class_counts[a] for a in unique_actions}
        weights = np.array([class_weights[a] for a in actions], dtype=np.float64)

        print(f"Granular Class Balancing enabled for {num_classes} action types.")
        for action in sorted(unique_actions):
            print(f"  - {action:<20}: count={class_counts[action]:<5} weight={class_weights[action]:.2f}")

        return torch.DoubleTensor(weights)

    def parse_waypoints(self, wp_str):
        """Parse string representation of list of lists."""
        try:
            if pd.isna(wp_str) or wp_str == '[]':
                return np.zeros((0, 2), dtype=np.float32)
            return np.array(json.loads(wp_str), dtype=np.float32)
        except Exception as e:
            try:
                import ast
                return np.array(ast.literal_eval(wp_str), dtype=np.float32)
            except:
                print(f"Error parsing waypoints: {wp_str[:20]}... {e}")
                return np.zeros((0, 2), dtype=np.float32)

    def transform_to_ego(self, points_global, ego_x, ego_y, ego_yaw):
        """Transform global coordinates to ego vehicle frame."""
        if len(points_global) == 0:
            return points_global

        points_ego = points_global - np.array([ego_x, ego_y])
        c = np.cos(-ego_yaw)
        s = np.sin(-ego_yaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        points_ego = points_ego @ rotation_matrix.T
        return points_ego.astype(np.float32)

    def load_lidar_in_ego_frame(self, nusc, sample_token, ref_ego_pose):
        """
        Load a LiDAR point cloud for a given sample_token and transform it
        into the reference ego frame (ref_ego_pose).

        Args:
            nusc: NuScenes object
            sample_token: Token of the sample to load LiDAR for
            ref_ego_pose: The ego_pose dict of the CURRENT (reference) frame

        Returns:
            points: (N, 5) array [x, y, z, intensity, 0] in current ego frame
        """
        try:
            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            sd_record = nusc.get('sample_data', lidar_token)
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

            # Load raw LiDAR
            lidar_path = nusc.get_sample_data_path(lidar_token)
            pc = LidarPointCloud.from_file(lidar_path)

            # 1. Sensor frame → Ego frame (past ego)
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))

            # 2. Past Ego frame → Global frame
            pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix)
            pc.translate(np.array(ego_pose['translation']))

            # 3. Global frame → Current Ego frame (inverse of ref_ego_pose)
            ref_translation = np.array(ref_ego_pose['translation'])
            ref_rotation = Quaternion(ref_ego_pose['rotation'])

            pc.translate(-ref_translation)
            pc.rotate(ref_rotation.inverse.rotation_matrix)

            # (N, 4) → (N, 5) with zero padding for time channel
            points = pc.points.T  # (N, 4): x, y, z, intensity
            points = np.hstack([points, np.zeros((len(points), 1), dtype=np.float32)])

            return points.astype(np.float32)

        except Exception as e:
            print(f"Failed to load LiDAR for sample {sample_token}: {e}")
            return np.zeros((1, 5), dtype=np.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample_token = row['sample_token']

        nusc = self.ext_ref_args.get('nusc_obj')

        # ── 1. Load current frame metadata ──────────────────────────────────
        ego_yaw = row['ego_yaw']
        ref_ego_pose = None
        current_points = np.zeros((1, 5), dtype=np.float32)

        if nusc:
            try:
                sample = nusc.get('sample', sample_token)
                lidar_token = sample['data']['LIDAR_TOP']
                sd_record = nusc.get('sample_data', lidar_token)
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

                ego_yaw = Quaternion(ref_ego_pose['rotation']).yaw_pitch_roll[0]

                # Load current frame in ego frame
                current_points = self.load_lidar_in_ego_frame(nusc, sample_token, ref_ego_pose)

            except Exception as e:
                print(f"Failed to load current NuScenes data for {sample_token}: {e}")
                current_points = np.zeros((1, 5), dtype=np.float32)

        # ── 2. Load past keyframes ───────────────────────────────────────────
        # Each past frame is transformed into the CURRENT ego frame
        past_lidar_frames = []  # list of (N_i, 5) arrays

        if nusc and ref_ego_pose is not None:
            try:
                curr_sample = nusc.get('sample', sample_token)
                for t in range(self.num_past_sweeps):
                    prev_token = curr_sample.get('prev', '')
                    if prev_token == '':
                        break  # No more past frames
                    past_pts = self.load_lidar_in_ego_frame(nusc, prev_token, ref_ego_pose)
                    past_lidar_frames.append(past_pts)
                    curr_sample = nusc.get('sample', prev_token)
            except Exception as e:
                print(f"Failed to load past LiDAR frames for {sample_token}: {e}")

        # Pad with empty frames if fewer than num_past_sweeps available
        while len(past_lidar_frames) < self.num_past_sweeps:
            past_lidar_frames.append(np.zeros((1, 5), dtype=np.float32))

        # ── 3. Parse Waypoints & Pose ────────────────────────────────────────
        past_embed_global = self.parse_waypoints(row.get('past_waypoints', '[]'))
        future_embed_global = self.parse_waypoints(row.get('waypoints', '[]'))

        ego_x = row['ego_x']
        ego_y = row['ego_y']

        past_embed_ego = self.transform_to_ego(past_embed_global, ego_x, ego_y, ego_yaw)
        gt_trajectory_ego = self.transform_to_ego(future_embed_global, ego_x, ego_y, ego_yaw)

        # ── 4. Reference Trajectory ──────────────────────────────────────────
        action_token = row.get('action_token', '')
        ref_traj_list = self.action_templates.get(action_token, [])
        if len(ref_traj_list) == 0:
            ref_traj_ego = np.zeros((len(gt_trajectory_ego), 2), dtype=np.float32)
        else:
            ref_traj_ego = np.array(ref_traj_list, dtype=np.float32)

        # ── 5. Calculate Delta ───────────────────────────────────────────────
        min_len = min(len(gt_trajectory_ego), len(ref_traj_ego))
        if min_len > 0:
            gt_trajectory_ego = gt_trajectory_ego[:min_len]
            ref_traj_ego = ref_traj_ego[:min_len]
            delta_xy = gt_trajectory_ego - ref_traj_ego
        else:
            delta_xy = np.zeros((0, 2), dtype=np.float32)

        # ── 6. Prepare input dict for current frame (for OpenPCDet processor) ─
        input_dict = {
            'points': current_points,
            'frame_id': sample_token,
            'action_token': action_token,
            'past_waypoints': past_embed_ego,
            'ref_trajectory': ref_traj_ego,
            'gt_delta': delta_xy,
            'gt_trajectory': gt_trajectory_ego,
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['past_len'] = len(past_embed_ego)
        data_dict['ref_len'] = len(ref_traj_ego)

        # ── 7. Store past LiDAR frames separately (not through OpenPCDet processor) ─
        # These will be handled in collate_batch
        data_dict['past_lidar_frames'] = past_lidar_frames  # list of num_past_sweeps arrays

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        Custom collate to handle:
        - Variable length waypoints (same as baseline)
        - Multiple past LiDAR frames per sample (NEW for Exp 5)
        """
        items_for_pcdet = []
        saved_items = []

        manual_keys = ['past_waypoints', 'ref_trajectory', 'gt_delta',
                       'gt_trajectory', 'action_token', 'past_lidar_frames']

        for item in batch_list:
            pcdet_item = item.copy()
            saved = {}
            for k in manual_keys:
                if k in pcdet_item:
                    saved[k] = pcdet_item.pop(k)
            items_for_pcdet.append(pcdet_item)
            saved_items.append(saved)

        # Standard OpenPCDet collate for current frame
        data_dict = DatasetTemplate.collate_batch(items_for_pcdet)

        # ── Past Waypoints ───────────────────────────────────────────────────
        past_lens = [x['past_len'] for x in batch_list]
        max_past_len = max(past_lens) if past_lens else 0

        batch_past_wp = []
        batch_past_mask = []

        for i, sample in enumerate(saved_items):
            curr_len = past_lens[i]
            pad_len = max_past_len - curr_len
            wp = sample['past_waypoints']
            if pad_len > 0:
                wp_padded = np.pad(wp, ((0, pad_len), (0, 0)), mode='constant')
            else:
                wp_padded = wp
            mask = np.zeros(max_past_len, dtype=np.float32)
            mask[:curr_len] = 1.0
            batch_past_wp.append(wp_padded)
            batch_past_mask.append(mask)

        data_dict['batch_past_waypoints'] = torch.from_numpy(np.stack(batch_past_wp)).float()
        data_dict['batch_past_mask'] = torch.from_numpy(np.stack(batch_past_mask)).float()

        # ── Ref Trajectory / Delta ───────────────────────────────────────────
        ref_trajs = [x['ref_trajectory'] for x in saved_items]
        gt_deltas = [x['gt_delta'] for x in saved_items]

        data_dict['split_ref_trajectory'] = ref_trajs
        data_dict['split_gt_trajectory'] = [x['gt_trajectory'] for x in saved_items]
        data_dict['action_token'] = [x['action_token'] for x in saved_items]

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

        # ── Past LiDAR Frames (NEW) ──────────────────────────────────────────
        # past_lidar_frames: list[B] of list[num_past_sweeps] of (N_i, 5) arrays
        # We store them as a list-of-lists for the model to process independently.
        # Each past frame gets its own batch_idx column added (like OpenPCDet's 'points').
        num_past = len(saved_items[0]['past_lidar_frames'])
        batch_size = len(saved_items)

        # For each past timestep t, build a stacked (N_total, 6) tensor [batch_idx, x, y, z, i, 0]
        past_frames_batched = []  # list of num_past tensors, each (N_total, 6)

        for t in range(num_past):
            frame_t_list = []
            for b in range(batch_size):
                pts = saved_items[b]['past_lidar_frames'][t]  # (N, 5)
                batch_col = np.full((len(pts), 1), b, dtype=np.float32)
                pts_with_idx = np.hstack([batch_col, pts])  # (N, 6)
                frame_t_list.append(pts_with_idx)
            stacked = np.vstack(frame_t_list)  # (N_total, 6)
            past_frames_batched.append(torch.from_numpy(stacked).float())

        data_dict['past_lidar_frames_batched'] = past_frames_batched  # list of num_past tensors

        return data_dict

    def prepare_data(self, data_dict):
        """Use PCDet's processing (Voxelization, etc)."""
        if self.training:
            pass
        if self.processor:
            data_dict = self.processor.forward(data_dict=data_dict)
        return data_dict
