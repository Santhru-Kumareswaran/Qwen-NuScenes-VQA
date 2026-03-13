import torch
import torch.nn as nn
import sys
sys.path.append("/home/santhru/FYP38_First Experiment/OpenPCDet_Install/OpenPCDet")
from pcdet.models import build_network


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, 2) Past waypoints
            mask: (B, L) 1 for valid, 0 for padded
        Returns:
            embedding: (B, hidden_dim) Last valid hidden state
        """
        out, (h_n, c_n) = self.lstm(x)

        if mask is not None:
            lengths = mask.sum(dim=1).long()
            batch_size = x.shape[0]
            embedding = torch.zeros(batch_size, self.lstm.hidden_size, device=x.device)
            for i in range(batch_size):
                l = lengths[i]
                if l > 0:
                    embedding[i] = out[i, l - 1]
        else:
            embedding = h_n[-1]

        return self.norm(embedding)


class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, query, key_map):
        """
        Args:
            query: (B, query_dim) Trajectory embedding
            key_map: (B, C, H, W) BEV features  — C = 6 * bev_dim for Exp 5
        """
        B, C, H, W = key_map.shape
        key_flat = key_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        q = self.q_proj(query.unsqueeze(1))   # (B, 1, hidden_dim)
        k = self.k_proj(key_flat)             # (B, H*W, hidden_dim)
        v = self.v_proj(key_flat)             # (B, H*W, hidden_dim)

        attn_out, _ = self.multihead_attn(q, k, v)  # (B, 1, hidden_dim)

        x = self.norm(q + attn_out)
        x = x + self.ffn(x)

        return x.squeeze(1)  # (B, hidden_dim)


class TemporalBEVFusion(nn.Module):
    """
    Fuses N BEV feature maps (one per LiDAR keyframe) without information loss.

    Architecture:
        Input:  N BEV maps  (B, N*C, H, W)  — channel-concatenated
        5× [1×1 Conv + BN + ReLU]           — temporal channel mixing
        1× [3×3 Conv + BN + ReLU]           — spatial-temporal mixing
        Output: (B, N*C, H, W)              — full channel resolution preserved
    """
    def __init__(self, num_frames, bev_channels):
        super().__init__()
        total_channels = num_frames * bev_channels

        layers = []
        # 5 × 1×1 conv blocks for temporal channel mixing
        for _ in range(5):
            layers += [
                nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(total_channels),
                nn.ReLU(inplace=True),
            ]
        # 1 × 3×3 conv block for spatial-temporal mixing
        layers += [
            nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
        ]

        self.fusion = nn.Sequential(*layers)

    def forward(self, bev_maps):
        """
        Args:
            bev_maps: list of N tensors, each (B, C, H, W)
        Returns:
            fused: (B, N*C, H, W)
        """
        x = torch.cat(bev_maps, dim=1)  # (B, N*C, H, W)
        return self.fusion(x)           # (B, N*C, H, W)


class TrajectoryRefinementModel(nn.Module):
    """
    Experiment 5: Past LiDAR Sweeps model.

    Key difference from baseline:
    - Runs the shared backbone on (1 + num_past_sweeps) LiDAR frames independently
    - Fuses the resulting BEV maps via TemporalBEVFusion (5×1×1 + 3×3 conv)
    - Passes the fused BEV (B, num_frames*C, H, W) to CrossAttentionFusion
    - Decoder and trajectory encoders are identical to baseline
    """
    def __init__(self, model_cfg, num_class, dataset,
                 bev_dim=None, traj_hidden_dim=128, fusion_hidden_dim=256,
                 prediction_horizon=12, num_past_sweeps=5):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_past_sweeps = num_past_sweeps
        self.num_frames = 1 + num_past_sweeps  # current + past

        # ── 1. Shared LiDAR Backbone ─────────────────────────────────────────
        self.lidar_backbone = build_network(model_cfg, num_class, dataset)

        # Freeze backbone
        for param in self.lidar_backbone.parameters():
            param.requires_grad = False

        # ── 2. Temporal BEV Fusion (NEW) ─────────────────────────────────────
        self.bev_dim = bev_dim if bev_dim is not None else model_cfg.get('NUM_BEV_FEATURES', 256)
        self.fused_bev_dim = self.num_frames * self.bev_dim  # e.g. 6 * 384 = 2304

        self.temporal_fusion = TemporalBEVFusion(
            num_frames=self.num_frames,
            bev_channels=self.bev_dim
        )

        # ── 3. Trajectory Encoders (identical to baseline) ───────────────────
        self.past_encoder = TrajectoryEncoder(input_dim=2, hidden_dim=traj_hidden_dim)
        self.ref_lstm = TrajectoryEncoder(input_dim=2, hidden_dim=traj_hidden_dim)

        self.traj_dim = traj_hidden_dim * 2  # past + ref

        # ── 4. Cross-Attention Fusion (key_dim = fused_bev_dim) ──────────────
        self.fusion_module = CrossAttentionFusion(
            query_dim=self.traj_dim,
            key_dim=self.fused_bev_dim,   # 6C instead of C
            hidden_dim=fusion_hidden_dim
        )

        # ── 5. Decoder (identical to baseline) ───────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * prediction_horizon)
        )
        self.pred_horizon = prediction_horizon

    def _run_backbone(self, batch_dict):
        """
        Run the shared backbone on a batch_dict and return spatial_features_2d.
        Skips detection heads.
        """
        for module in self.lidar_backbone.module_list:
            name = module.model_cfg.NAME
            if name in ['DenseHead', 'PointHead', 'ROIHead', 'CenterHead'] or 'Head' in name:
                continue
            batch_dict = module(batch_dict)
        return batch_dict['spatial_features_2d']  # (B, C, H, W)

    def _make_past_batch_dict(self, past_points_tensor, batch_dict):
        """
        Build a minimal batch_dict for a past LiDAR frame so it can be
        processed by the backbone.

        past_points_tensor: (N_total, 6) [batch_idx, x, y, z, intensity, 0]
        """
        import copy
        past_bd = {}
        # Copy voxel-related keys that the backbone needs (from current frame)
        # We only replace 'points' — voxelization happens inside the backbone modules
        for k in batch_dict:
            if k not in ['points', 'voxels', 'voxel_coords', 'voxel_num_points',
                         'spatial_features', 'spatial_features_2d',
                         'batch_past_waypoints', 'batch_past_mask',
                         'batch_ref_trajectory', 'batch_gt_delta', 'batch_target_mask',
                         'past_lidar_frames_batched']:
                past_bd[k] = batch_dict[k]

        past_bd['points'] = past_points_tensor
        return past_bd

    def forward(self, batch_dict):
        # ── 1. Encode current LiDAR frame ────────────────────────────────────
        bev_current = self._run_backbone(batch_dict)  # (B, C, H, W)

        # ── 2. Encode each past LiDAR frame independently ────────────────────
        past_frames_batched = batch_dict.get('past_lidar_frames_batched', [])
        bev_maps = [bev_current]

        for t, past_pts in enumerate(past_frames_batched):
            past_pts = past_pts.to(bev_current.device)
            past_bd = self._make_past_batch_dict(past_pts, batch_dict)
            bev_past = self._run_backbone(past_bd)  # (B, C, H, W)
            bev_maps.append(bev_past)

        # Pad with zeros if fewer past frames than expected
        while len(bev_maps) < self.num_frames:
            bev_maps.append(torch.zeros_like(bev_current))

        # ── 3. Temporal BEV Fusion ────────────────────────────────────────────
        # Input:  list of num_frames tensors (B, C, H, W)
        # Output: (B, num_frames*C, H, W)
        fused_bev = self.temporal_fusion(bev_maps)  # (B, 6C, H, W)

        # ── 4. Trajectory Encoding (identical to baseline) ───────────────────
        past_embed = self.past_encoder(
            batch_dict['batch_past_waypoints'],
            batch_dict['batch_past_mask']
        )

        batch_ref = batch_dict['batch_ref_trajectory']
        ref_mask = batch_dict['batch_target_mask']
        ref_embed = self.ref_lstm(batch_ref, ref_mask)

        query_emb = torch.cat([past_embed, ref_embed], dim=1)  # (B, 256)

        # ── 5. Cross-Attention Fusion ─────────────────────────────────────────
        fused_emb = self.fusion_module(query_emb, fused_bev)  # (B, fusion_hidden_dim)

        # ── 6. Decode ─────────────────────────────────────────────────────────
        deltas = self.decoder(fused_emb)              # (B, 2*T)
        deltas = deltas.view(-1, self.pred_horizon, 2)  # (B, T, 2)

        return {'predicted_deltas': deltas}
