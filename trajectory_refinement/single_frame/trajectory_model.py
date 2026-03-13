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
        # Pack padded sequence if mask is provided could be optimized, 
        # but for simplicity using standard output extract
        out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state (h_n[-1] for single layer)
        # However, for padded sequences, h_n might reflect the padded zeros step.
        # We should ideally gather the state at the last valid index.
        # Simplification: Since 0 input to LSTM (bias=True) drifts state, masking is safer to use.
        # Let's use the mean of valid outputs or just extraction.
        
        # Extract last valid state using mask
        if mask is not None:
             # Lengths
            lengths = mask.sum(dim=1).long()
            # Handle empty sequences (length 0) -> return 0 embedding
            batch_size = x.shape[0]
            embedding = torch.zeros(batch_size, self.lstm.hidden_size, device=x.device)
            
            for i in range(batch_size):
                l = lengths[i]
                if l > 0:
                    embedding[i] = out[i, l-1]
        else:
            embedding = h_n[-1]
            
        return self.norm(embedding)

class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Projections to common dimension
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
            query: (B, query_dim) Trajectory Embedding
            key_map: (B, C, H, W) BEV Features
        """
        B, C, H, W = key_map.shape
        # Flatten spatial dimensions: (B, H*W, C)
        key_flat = key_map.view(B, C, -1).permute(0, 2, 1) # (B, N, C)
        
        # Prepare Q, K, V
        # Query needs sequence dim: (B, 1, query_dim)
        q = self.q_proj(query.unsqueeze(1)) 
        k = self.k_proj(key_flat)
        v = self.v_proj(key_flat)
        
        # Attention
        # Output: (B, 1, hidden_dim)
        attn_out, _ = self.multihead_attn(q, k, v)
        
        # Add & Norm
        x = self.norm(q + attn_out)
        
        # FFN
        x = x + self.ffn(x)
        
        return x.squeeze(1) # (B, hidden_dim)


class TrajectoryRefinementModel(nn.Module):
    def __init__(self, model_cfg, num_class, dataset, bev_dim=None, 
                 traj_hidden_dim=128, fusion_hidden_dim=256, prediction_horizon=12):
        super().__init__()
        self.model_cfg = model_cfg
        
        # 1. OpenPCDet Backbone (BEV Encoder)
        # We reuse the standard build_network but typically we only need up to 'backbone_2d' or 'dense_head' features.
        # Actually, `build_network` builds a detector. We can treat it as a submodule.
        self.lidar_backbone = build_network(model_cfg, num_class, dataset)
        
        # Freeze backbone
        for param in self.lidar_backbone.parameters():
             param.requires_grad = False
             
        # 2. Trajectory Encoders
        self.past_encoder = TrajectoryEncoder(input_dim=2, hidden_dim=traj_hidden_dim)

        self.ref_lstm = TrajectoryEncoder(input_dim=2, hidden_dim=traj_hidden_dim)

        # 3. Fusion
        # Dimensions: 
        # BEV Feature dim depends on model config (e.g. 256 for PointPillars, 128 for others).
        # We need to inspect `num_bev_features` from backbone. 
        # Typically detected automatically or hardcoded.
        # Let's assume standard 256 for now or add a projection.
        self.bev_dim = bev_dim if bev_dim is not None else model_cfg.get('NUM_BEV_FEATURES', 256) 
        self.traj_dim = traj_hidden_dim * 2 # Past + Ref
        
        self.fusion_module = CrossAttentionFusion(
            query_dim=self.traj_dim,
            key_dim=self.bev_dim, 
            hidden_dim=fusion_hidden_dim
        )
        
        # 4. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * prediction_horizon) # Predict T steps * 2 (dx, dy)
        )
        self.pred_horizon = prediction_horizon # Steps

    def forward(self, batch_dict):
        # 1. LiDAR Encoding
        # Run OpenPCDet model through its phases
        # We need to access intermediate feature map `spatial_features_2d`.
        # Standard OpenPCDet `forward` returns `pred_dicts`. We need to hook or modify it.
        # Modification: We can call submodules manually if we know the structure (vfe -> backbone -> map_to_bev -> backbone_2d).
        
        # Execute VFE, MapToBEV, Backbone2D
        # This order depends on the config (PointPillars vs VoxelNet).
        # Wrapper to be safe: use `self.lidar_backbone` but stop before HEAD.
        
        # Hack/Standard way:
        # Most pcdet models have `module_list`. We can run them sequentially.
        for module in self.lidar_backbone.module_list:
            name = module.model_cfg.NAME
            if name in ['DenseHead', 'PointHead', 'ROIHead', 'CenterHead'] or 'Head' in name:
                continue # Skip detection heads
            batch_dict = module(batch_dict)
            
        # Now batch_dict['spatial_features_2d'] should exist (B, C, H, W)
        bev_features = batch_dict['spatial_features_2d']
        
        # 2. Trajectory Encoding
        past_embed = self.past_encoder(batch_dict['batch_past_waypoints'], batch_dict['batch_past_mask'])
        
        # Mask for ref? If we padded it.
        batch_ref = batch_dict['batch_ref_trajectory']
        ref_mask = batch_dict['batch_target_mask']
        ref_embed = self.ref_lstm(batch_ref, ref_mask)
        
        # Concatenate Query Context
        query_emb = torch.cat([past_embed, ref_embed], dim=1) # (B, 256)
        
        # 3. Fusion
        fused_emb = self.fusion_module(query_emb, bev_features)
        
        # 4. Decode
        deltas = self.decoder(fused_emb) # (B, 24)
        deltas = deltas.view(-1, self.pred_horizon, 2)
        
        # If Ground Truth has different length, we align in Loss computation.
        
        return {
            'predicted_deltas': deltas
        }
