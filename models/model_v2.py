import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MPLAM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.theta_tw = nn.Parameter(torch.tensor(3.5))
        self.theta_r = nn.Parameter(torch.tensor(0.12))
        self.theta_s = nn.Parameter(torch.tensor(0.03))
        self.theta_n = nn.Parameter(torch.tensor(0.07))
        
        self.warp_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x, adj, coords):
        batch_size, num_nodes, time_steps, features = x.shape
        
        warp_weights = self.warp_net(x.mean(dim=1))
        warp_matrix = torch.softmax(warp_weights @ warp_weights.transpose(-2, -1) / math.sqrt(features), dim=-1)
        x_tw = torch.matmul(warp_matrix, x.mean(dim=1)).unsqueeze(1).expand(-1, num_nodes, -1, -1)
        x_tw = self.theta_tw * x + (1 - self.theta_tw) * x_tw
        
        if self.training:
            theta_r_clamped = torch.clamp(self.theta_r, 0, 0.3)
            mask = torch.bernoulli(torch.ones_like(adj) * (1 - theta_r_clamped))
            adj_aug = adj * mask
        else:
            adj_aug = adj
        
        theta_s_clamped = torch.clamp(self.theta_s, 0, 0.1)
        coords_aug = coords + theta_s_clamped * torch.randn_like(coords)
        
        theta_n_clamped = torch.clamp(self.theta_n, 0, 0.15)
        noise = torch.randn_like(x_tw) * theta_n_clamped
        x_aug = x_tw + noise
        
        return x_aug, adj_aug, coords_aug


class MFPE(nn.Module):
    def __init__(self, d_pe=64, d_te=64, d_h=64, max_len=5000):
        super().__init__()
        self.d_pe = d_pe
        self.d_te = d_te
        self.d_h = d_h
        
        self.omega_1 = nn.Parameter(torch.tensor(2 * np.pi / 24))
        self.omega_2 = nn.Parameter(torch.tensor(2 * np.pi / 168))
        
        self.attention_weights = nn.Sequential(
            nn.Linear(d_pe + d_te + 5, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )
        
        self.proj1 = nn.Sequential(
            nn.Linear(d_pe, d_h),
            nn.LayerNorm(d_h)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(d_te, d_h),
            nn.LayerNorm(d_h)
        )
        self.proj3 = nn.Sequential(
            nn.Linear(4, d_h),
            nn.LayerNorm(d_h)
        )
        self.proj4 = nn.Sequential(
            nn.Linear(1, d_h),
            nn.LayerNorm(d_h)
        )
        
        pe = torch.zeros(max_len, d_pe)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pe, 2).float() * (-math.log(10000.0) / d_pe))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_cache', pe)
        
        te = torch.zeros(max_len, d_te)
        te[:, 0::2] = torch.sin(position * div_term[:d_te//2])
        te[:, 1::2] = torch.cos(position * div_term[:d_te//2])
        self.register_buffer('te_cache', te)
        
    def forward(self, num_nodes, time_steps, adj):
        pe = self.pe_cache[:num_nodes]
        te = self.te_cache[:time_steps]
        tm = self.learnable_temporal_frequencies(time_steps)
        fe = self.graph_topology_encoding(adj, num_nodes)
        
        pe_exp = pe.unsqueeze(1).expand(-1, time_steps, -1)
        te_exp = te.unsqueeze(0).expand(num_nodes, -1, -1)
        tm_exp = tm.unsqueeze(0).expand(num_nodes, -1, -1)
        fe_exp = fe.unsqueeze(1).expand(-1, time_steps, -1)
        
        concat = torch.cat([pe_exp, te_exp, tm_exp, fe_exp], dim=-1)
        alphas = F.softmax(self.attention_weights(concat), dim=-1)
        
        pe_proj = self.proj1(pe_exp)
        te_proj = self.proj2(te_exp)
        tm_proj = self.proj3(tm_exp)
        fe_proj = self.proj4(fe_exp)
        
        fused = (alphas[..., 0:1] * pe_proj + 
                alphas[..., 1:2] * te_proj + 
                alphas[..., 2:3] * tm_proj + 
                alphas[..., 3:4] * fe_proj)
        
        return fused
    
    def learnable_temporal_frequencies(self, time_steps):
        t = torch.arange(0, time_steps, dtype=torch.float).to(self.omega_1.device)
        tm = torch.stack([
            torch.sin(self.omega_1 * t),
            torch.cos(self.omega_1 * t),
            torch.sin(self.omega_2 * t),
            torch.cos(self.omega_2 * t)
        ], dim=-1)
        return tm
    
    def graph_topology_encoding(self, adj, num_nodes):
        device = adj.device
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d + 1e-6, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.mm(d_mat_inv_sqrt, torch.mm(adj, d_mat_inv_sqrt))
        fe = torch.mm(norm_adj, torch.ones(num_nodes, 1).to(device))
        return fe


class GraphAttention(nn.Module):
    def __init__(self, d_h, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(d_h, d_h)
        self.key = nn.Linear(d_h, d_h)
        self.value = nn.Linear(d_h, d_h)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_h)
        
    def forward(self, h, adj):
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = attn * adj.unsqueeze(1).unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        return out


class AdaptiveAdjacency(nn.Module):
    def __init__(self, d_h, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_h, 2 * d_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_h, d_h),
            nn.LayerNorm(d_h)
        )
        self.graph_attn = GraphAttention(d_h, dropout)
        self.alpha = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, h, static_adj):
        h_mean = h.mean(dim=2)
        h_transformed = self.mlp(h_mean)
        
        similarity = torch.matmul(h_transformed, h_transformed.transpose(-2, -1))
        learned_adj = torch.sigmoid(similarity)
        
        alpha_clamped = torch.clamp(self.alpha, 0.5, 0.9)
        adaptive_adj = alpha_clamped * static_adj + (1 - alpha_clamped) * learned_adj
        
        return adaptive_adj


class MultiScaleGCN(nn.Module):
    def __init__(self, d_h, scales=[1, 3, 5, 7], dropout=0.1):
        super().__init__()
        self.scales = scales
        self.weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.LayerNorm(d_h),
                nn.Dropout(dropout)
            ) for _ in scales
        ])
        self.residual_weights = nn.ModuleList([nn.Linear(d_h, d_h) for _ in scales])
        
        self.scale_mlp = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h // 2, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
            nn.ReLU()
        )
        
    def forward(self, h, adj):
        batch_size, num_nodes, time_steps, d_h = h.shape
        
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d + 1e-6, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        
        h_scales = []
        h_global = []
        
        adj_powers = [adj]
        for k in range(1, max(self.scales)):
            adj_powers.append(torch.matmul(adj_powers[-1], adj))
        
        for i, k in enumerate(self.scales):
            adj_k = adj_powers[k-1]
            norm_adj = torch.matmul(d_mat_inv_sqrt, torch.matmul(adj_k, d_mat_inv_sqrt))
            
            h_reshaped = h.reshape(batch_size * time_steps, num_nodes, d_h)
            h_conv = torch.matmul(norm_adj, h_reshaped)
            h_conv = h_conv.reshape(batch_size, num_nodes, time_steps, d_h)
            
            h_k = self.weights[i](h_conv)
            h_k = F.relu(h_k + self.residual_weights[i](h))
            
            h_scales.append(h_k)
            h_global.append(h_k.mean(dim=(1, 2)))
        
        betas = torch.stack([self.scale_mlp(h_g) for h_g in h_global], dim=-1)
        betas = F.softmax(betas, dim=-1)
        
        h_fused = sum(betas[..., i:i+1].unsqueeze(1).unsqueeze(1) * h_scales[i] 
                     for i in range(len(self.scales)))
        
        h_fused = self.fusion(h_fused)
        
        return h_fused


class HierarchicalPooling(nn.Module):
    def __init__(self, pool_rate=0.5):
        super().__init__()
        self.pool_rate = pool_rate
        self.score_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h, adj):
        batch_size, num_nodes, time_steps, d_h = h.shape
        
        node_importance = torch.norm(h, p=2, dim=-1).mean(dim=-1, keepdim=True) / math.sqrt(d_h)
        scores = self.score_net(node_importance).squeeze(-1)
        
        k = max(1, int(num_nodes * self.pool_rate))
        _, idx = torch.topk(scores, k, dim=-1)
        idx_sorted, _ = torch.sort(idx, dim=-1)
        
        h_pool = torch.gather(h, 1, idx_sorted.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, time_steps, d_h))
        
        adj_pool = torch.gather(adj, 1, idx_sorted.unsqueeze(-1).expand(-1, -1, num_nodes))
        adj_pool = torch.gather(adj_pool, 2, idx_sorted.unsqueeze(1).expand(-1, k, -1))
        
        return h_pool, adj_pool, idx_sorted


class SpatialProcessingModule(nn.Module):
    def __init__(self, d_h=64, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.adaptive_adjs = nn.ModuleList([AdaptiveAdjacency(d_h, dropout) for _ in range(num_layers)])
        self.gcns = nn.ModuleList([MultiScaleGCN(d_h, dropout=dropout) for _ in range(num_layers)])
        self.poolings = nn.ModuleList([
            HierarchicalPooling() if i % 2 == 1 else nn.Identity() 
            for i in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h, adj):
        pool_indices = []
        pool_adjs = []
        current_h = h
        current_adj = adj
        
        for i in range(self.num_layers):
            residual = current_h
            
            adaptive_adj = self.adaptive_adjs[i](current_h, current_adj)
            
            if self.training:
                mask = torch.bernoulli(torch.ones_like(adaptive_adj) * 0.88)
                adaptive_adj = adaptive_adj * mask
            
            current_h = self.gcns[i](current_h, adaptive_adj)
            current_h = self.layer_norms[i](current_h)
            current_h = self.dropout(current_h)
            
            if current_h.shape[1] == residual.shape[1]:
                current_h = current_h + residual
            
            if isinstance(self.poolings[i], HierarchicalPooling):
                current_h, current_adj, idx = self.poolings[i](current_h, adaptive_adj)
                pool_indices.append(idx)
                pool_adjs.append(adaptive_adj)
        
        h_spatial = self.unpool(current_h, h.shape[1], pool_indices, pool_adjs)
        
        return h_spatial
    
    def unpool(self, h_pooled, original_num_nodes, pool_indices, pool_adjs):
        if not pool_indices or h_pooled.shape[1] == original_num_nodes:
            return h_pooled
        
        batch_size, _, time_steps, d_h = h_pooled.shape
        device = h_pooled.device
        
        h_unpooled = torch.zeros(batch_size, original_num_nodes, time_steps, d_h).to(device)
        
        for b in range(batch_size):
            idx = pool_indices[-1][b]
            h_unpooled[b, idx] = h_pooled[b, :len(idx)]
            
            missing_mask = torch.ones(original_num_nodes, dtype=torch.bool, device=device)
            missing_mask[idx] = False
            missing_idx = torch.where(missing_mask)[0]
            
            if len(missing_idx) > 0 and len(pool_adjs) > 0:
                adj = pool_adjs[-1][b]
                for m_idx in missing_idx:
                    neighbors = torch.where(adj[m_idx] > 0)[0]
                    valid_neighbors = neighbors[neighbors < len(idx)]
                    if len(valid_neighbors) > 0:
                        h_unpooled[b, m_idx] = h_pooled[b, valid_neighbors].mean(dim=0)
                    else:
                        h_unpooled[b, m_idx] = h_pooled[b].mean(dim=0)
        
        return h_unpooled


class LipschitzMultiHeadAttention(nn.Module):
    def __init__(self, d_h, n_heads=8, dropout=0.1, lipschitz_k=1.5):
        super().__init__()
        self.d_h = d_h
        self.n_heads = n_heads
        self.d_k = d_h // n_heads
        self.lipschitz_k = lipschitz_k
        
        self.q_linear = nn.Linear(d_h, d_h)
        self.k_linear = nn.Linear(d_h, d_h)
        self.v_linear = nn.Linear(d_h, d_h)
        self.out_linear = nn.Linear(d_h, d_h)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_h)
        
    def forward(self, x, mask=None):
        batch_size, num_nodes, time_steps, d_h = x.shape
        
        q = self.q_linear(x).view(batch_size, num_nodes, time_steps, self.n_heads, self.d_k).transpose(2, 3)
        k = self.k_linear(x).view(batch_size, num_nodes, time_steps, self.n_heads, self.d_k).transpose(2, 3)
        v = self.v_linear(x).view(batch_size, num_nodes, time_steps, self.n_heads, self.d_k).transpose(2, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(batch_size, num_nodes, time_steps, d_h)
        out = self.out_linear(out)
        
        out = self.spectral_norm_constraint(out)
        
        out = self.layer_norm(out + x)
        
        return out
    
    def spectral_norm_constraint(self, x):
        with torch.no_grad():
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            scale = torch.clamp(norm / self.lipschitz_k, min=1.0)
        return x / scale


class TemporalProcessingModule(nn.Module):
    def __init__(self, d_h=64, d_m=10, n_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_h = d_h
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        self.meta_proj = nn.Sequential(
            nn.Linear(d_m, d_h),
            nn.LayerNorm(d_h),
            nn.Dropout(dropout)
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
            nn.Dropout(dropout)
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(64, d_h),
            nn.LayerNorm(d_h),
            nn.Dropout(dropout)
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            nn.Sigmoid()
        )
        
        self.attention_layers = nn.ModuleList([
            LipschitzMultiHeadAttention(d_h, n_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_h, 4 * d_h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_h, d_h),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(num_layers)])
        
    def forward(self, h_spatial, meta_knowledge, temporal_embeddings):
        batch_size, num_nodes, time_steps, d_h = h_spatial.shape
        
        meta_proj = self.meta_proj(meta_knowledge)
        spatial_proj = self.spatial_proj(h_spatial)
        temp_proj = self.temporal_proj(temporal_embeddings)
        
        meta_exp = meta_proj.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        temp_exp = temp_proj.unsqueeze(1).expand(batch_size, num_nodes, -1, -1)
        
        gate = self.fusion_gate(torch.cat([meta_exp, spatial_proj, temp_exp], dim=-1))
        c_fuse = gate * (meta_exp + spatial_proj + temp_exp)
        c_fuse = F.relu(c_fuse)
        
        h_trans = c_fuse
        for i in range(self.num_layers):
            h_attn = self.attention_layers[i](h_trans)
            
            h_ffn = self.ffn_layers[i](h_attn)
            h_trans = self.layer_norms[i](h_attn + h_ffn)
        
        return h_trans


class FusionModule(nn.Module):
    def __init__(self, d_h=64, d_f=128, dropout=0.1):
        super().__init__()
        self.d_f = d_f
        
        self.proj_s = nn.Sequential(
            nn.Linear(d_h, d_f),
            nn.LayerNorm(d_f),
            nn.Dropout(dropout)
        )
        self.proj_t = nn.Sequential(
            nn.Linear(d_h, d_f),
            nn.LayerNorm(d_f),
            nn.Dropout(dropout)
        )
        self.proj_i = nn.Sequential(
            nn.Linear(d_h, d_f),
            nn.LayerNorm(d_f),
            nn.Dropout(dropout)
        )
        
        self.fusion_attn = nn.Sequential(
            nn.Linear(3 * d_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        self.final_proj = nn.Sequential(
            nn.Linear(d_f, d_f),
            nn.LayerNorm(d_f),
            nn.ReLU()
        )
        
    def forward(self, h_spatial, h_temporal, h_input):
        h_s = self.proj_s(h_spatial)
        h_t = self.proj_t(h_temporal)
        h_i = self.proj_i(h_input)
        
        concat = torch.cat([h_s, h_t, h_i], dim=-1)
        alphas = F.softmax(self.fusion_attn(concat), dim=-1)
        
        h_fused = alphas[..., 0:1] * h_s + alphas[..., 1:2] * h_t + alphas[..., 2:3] * h_i
        h_fused = self.final_proj(h_fused)
        
        return h_fused


class DualPathDecoder(nn.Module):
    def __init__(self, d_h=64, d_f=128, d_o=1, horizon=12, K=3, dropout=0.1):
        super().__init__()
        self.d_h = d_h
        self.d_f = d_f
        self.d_o = d_o
        self.horizon = horizon
        self.K = K
        
        self.theta_spatial = nn.ParameterList([nn.Parameter(torch.randn(1) * 0.1) for _ in range(K + 1)])
        self.spatial_conv = nn.Sequential(
            nn.Linear(d_f, d_f),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_f, d_o)
        )
        
        self.q_proj = nn.Linear(64, d_f)
        self.k_proj = nn.Linear(d_f, d_f)
        self.v_proj = nn.Linear(d_f, d_f)
        
        self.temporal_conv = nn.Sequential(
            nn.Linear(d_f, d_f),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_f, d_o)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_o, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_o),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(d_o, d_o * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_o * 2, d_o, kernel_size=1)
        )
        
    def forward(self, h_fused, adj, temporal_embeddings, time_steps):
        batch_size, num_nodes, _, d_f = h_fused.shape
        
        L = self.compute_normalized_laplacian(adj)
        lambda_max = self.estimate_lambda_max(L)
        L_scaled = 2 * L / (lambda_max + 1e-8) - torch.eye(L.shape[0]).to(L.device)
        
        chebyshev_polys = self.chebyshev_polynomials(L_scaled, self.K)
        
        h_spatial_features = torch.zeros(batch_size, num_nodes, time_steps, d_f).to(h_fused.device)
        for k in range(self.K + 1):
            h_spatial_k = torch.matmul(chebyshev_polys[k], h_fused.mean(dim=2))
            h_spatial_k = h_spatial_k.unsqueeze(2).expand(-1, -1, time_steps, -1)
            h_spatial_features += self.theta_spatial[k] * h_spatial_k
        
        h_spatial_out = self.spatial_conv(h_spatial_features)
        h_spatial_pred = h_spatial_out[:, :, -self.horizon:, :]
        
        future_te = temporal_embeddings[:, time_steps:time_steps + self.horizon, :]
        queries = self.q_proj(future_te).unsqueeze(1).expand(-1, num_nodes, -1, -1)
        keys = self.k_proj(h_fused)
        values = self.v_proj(h_fused)
        
        queries_flat = queries.reshape(batch_size * num_nodes, self.horizon, d_f)
        keys_flat = keys.reshape(batch_size * num_nodes, time_steps, d_f)
        values_flat = values.reshape(batch_size * num_nodes, time_steps, d_f)
        
        attn_scores = torch.matmul(queries_flat, keys_flat.transpose(-2, -1)) / math.sqrt(d_f)
        
        causal_mask = torch.tril(torch.ones(self.horizon, time_steps)).to(h_fused.device)
        attn_scores = attn_scores * causal_mask.unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_temporal_attn = torch.matmul(attn_weights, values_flat)
        
        h_temporal_out = h_temporal_attn.reshape(batch_size, num_nodes, self.horizon, d_f)
        h_temporal_pred = self.temporal_conv(h_temporal_out)
        
        gate = self.gate_net(torch.cat([h_spatial_pred, h_temporal_pred], dim=-1))
        y_pred = gate * h_spatial_pred + (1 - gate) * h_temporal_pred
        
        y_pred_refined = self.final_conv(y_pred.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        return y_pred_refined
    
    def compute_normalized_laplacian(self, adj):
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d + 1e-6, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        I = torch.eye(adj.shape[0]).to(adj.device)
        L = I - torch.matmul(d_mat_inv_sqrt, torch.matmul(adj, d_mat_inv_sqrt))
        return L
    
    def estimate_lambda_max(self, L):
        return 2.0
    
    def chebyshev_polynomials(self, L, K):
        polys = [torch.eye(L.shape[0]).to(L.device)]
        if K > 0:
            polys.append(L)
        for k in range(2, K + 1):
            poly_k = 2 * torch.matmul(L, polys[-1]) - polys[-2]
            polys.append(poly_k)
        return polys


class MASTNet(nn.Module):
    def __init__(self, num_nodes, input_dim=1, d_h=64, d_m=10, n_heads=8, 
                 num_spatial_layers=4, num_temporal_layers=4, horizon=12, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.d_h = d_h
        self.horizon = horizon
        
        self.mplam = MPLAM(input_dim, d_h)
        self.mfpe = MFPE(d_pe=64, d_te=64, d_h=d_h)
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + d_h, d_h),
            nn.LayerNorm(d_h),
            nn.Dropout(dropout)
        )
        
        self.spatial_module = SpatialProcessingModule(d_h=d_h, num_layers=num_spatial_layers, dropout=dropout)
        self.temporal_module = TemporalProcessingModule(d_h=d_h, d_m=d_m, n_heads=n_heads, 
                                                                num_layers=num_temporal_layers, dropout=dropout)
        
        self.fusion_module = FusionModule(d_h=d_h, d_f=128, dropout=dropout)
        self.decoder = DualPathDecoder(d_h=d_h, d_f=128, d_o=input_dim, horizon=horizon, dropout=dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, adj, coords, meta_knowledge):
        batch_size, num_nodes, time_steps, input_dim = x.shape
        
        x_aug, adj_aug, coords_aug = self.mplam(x, adj, coords)
        
        pos_embeddings = self.mfpe(num_nodes, time_steps, adj_aug)
        
        x_with_pos = torch.cat([x_aug, pos_embeddings], dim=-1)
        h_0 = self.input_proj(x_with_pos)
        
        h_spatial = self.spatial_module(h_0, adj_aug)
        
        temporal_embeddings = self.mfpe.sinusoidal_temporal_encoding(time_steps + self.horizon)
        h_temporal = self.temporal_module(h_spatial, meta_knowledge, temporal_embeddings[:time_steps])
        
        h_fused = self.fusion_module(h_spatial, h_temporal, h_0)
        
        y_pred = self.decoder(h_fused, adj_aug, temporal_embeddings, time_steps)
        
        return y_pred
    
    def get_regularization_loss(self):
        aug_reg = (self.mplam.theta_tw ** 2 + self.mplam.theta_r ** 2 + 
                  self.mplam.theta_s ** 2 + self.mplam.theta_n ** 2)
        
        spatial_reg = 0
        for i in range(len(self.spatial_module.adaptive_adjs) - 1):
            alpha_diff = (self.spatial_module.adaptive_adjs[i].alpha - 
                         self.spatial_module.adaptive_adjs[i+1].alpha) ** 2
            spatial_reg += alpha_diff
        
        return 0.01 * aug_reg + 0.01 * spatial_reg


def create_enhanced_mast_net(num_nodes, input_dim=1, hidden_dim=64, meta_dim=10, 
                            n_heads=8, horizon=12, dropout=0.1):
    model = MASTNet(
        num_nodes=num_nodes,
        input_dim=input_dim,
        d_h=hidden_dim,
        d_m=meta_dim,
        n_heads=n_heads,
        num_spatial_layers=4,
        num_temporal_layers=4,
        horizon=horizon,
        dropout=dropout
    )
    return model