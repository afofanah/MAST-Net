"""
MAST-Net — Algorithm 1 (Batch Loss Computation)
Compatible with any input feature dimension F (1, 3, 4, …) and any horizon H.
Shapes:  B batch · N nodes · T seq_len · H pred_len · d d_h · F in_features
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Manifold Preservation  (used inside MPLAM)
# d_M(X, X_aug) = λ_s·θ_s + λ_tw·θ_tw + λ_n·θ_n + λ_c·θ_c
# ─────────────────────────────────────────────────────────────────────────────

class ManifoldPreservation(nn.Module):
    """
    Four-component bound keeping X_aug on the traffic-data manifold.

    θ_s  = ‖U⊤(X−X_aug)‖_F           road-topology  (top-k Laplacian eigenvectors)
    θ_tw = Σ_τ ‖∇_τX − ∇_τX_aug‖_F   temporal-differences  (rush-hour periodicity)
    θ_n  = 1/N Σ w_ij‖Δ_i−Δ_j‖        similarity-weighted neighbourhood residuals
    θ_c  = ‖C⊙(XX⊤−X_aug X_aug⊤)‖_F  collaborative co-evolution  (learned C)

    λ weights learned via softmax; C is a trainable collaborative adjacency.
    Works for any F: all operations reshape (B,N,T,F) → (B,N,T·F) internally.
    """

    def __init__(self, num_nodes: int, n_eigen: int = 16,
                 sigma: float = 1.0, epsilon: float = 1.0):
        super().__init__()
        self.n_eigen = n_eigen
        self.sigma   = sigma
        self.epsilon = epsilon
        self.log_lam = nn.Parameter(torch.zeros(4))
        self.C       = nn.Parameter(torch.eye(num_nodes) * 0.1)

    def _lam(self):
        return F.softmax(self.log_lam, dim=0)

    def _flat(self, X: torch.Tensor) -> torch.Tensor:
        """(B,N,T,F) → (B,N,T·F)  — F-agnostic flattening."""
        B, N, T, Fd = X.shape
        return X.reshape(B, N, T * Fd)

    def _theta_s(self, X, X_aug, L):
        """Project deviation onto k smoothest Laplacian modes."""
        k = min(self.n_eigen, L.shape[-1])
        try:
            _, U = torch.linalg.eigh(L)                   # (N,N) ascending
            U = U[:, :k]                                   # (N,k)
        except Exception:
            return X.new_zeros(1).squeeze()
        diff = self._flat(X - X_aug)                       # (B,N,T·F)
        return (diff.transpose(-1, -2) @ U).norm(dim=(-2,-1)).mean()

    def _theta_tw(self, X, X_aug):
        """Sum of temporal-difference norms across all lags τ."""
        T, acc = X.shape[2], X.new_zeros(1).squeeze()
        for tau in range(1, min(T, 4)):                    # cap at 4 for efficiency
            d  = (X[:,:,tau:] - X[:,:,:-tau])
            da = (X_aug[:,:,tau:] - X_aug[:,:,:-tau])
            acc = acc + (d - da).norm()
        return acc / max(min(T, 4) - 1, 1)

    def _theta_n(self, X, X_aug):
        """Similarity-weighted neighbourhood residual consistency."""
        Xf     = self._flat(X)                             # (B,N,T·F)
        Xf_aug = self._flat(X_aug)
        Xm     = Xf.mean(0)                               # (N,T·F)
        # Normalise before distance to be scale-invariant across datasets
        Xm_n   = F.normalize(Xm, dim=-1)
        dist2  = torch.cdist(Xm_n, Xm_n).pow(2)
        W      = torch.exp(-dist2 / (self.sigma ** 2 + 1e-8))
        W      = W * (dist2.sqrt() <= self.epsilon).float()
        W      = W / (W.sum(-1, keepdim=True).clamp(1e-8))  # row-normalise
        delta  = Xf - Xf_aug                               # (B,N,T·F)
        norms  = (delta.unsqueeze(2) - delta.unsqueeze(1)).norm(dim=-1)  # (B,N,N)
        return (W.unsqueeze(0) * norms).sum(dim=(-2,-1)).mean() / (Xm.shape[0])

    def _theta_c(self, X, X_aug):
        """Collaborative Gram-matrix deviation under learned C."""
        Xf    = self._flat(X)
        Xf_a  = self._flat(X_aug)
        G     = torch.bmm(Xf, Xf.transpose(-1,-2))        # (B,N,N)
        G_aug = torch.bmm(Xf_a, Xf_a.transpose(-1,-2))
        return (self.C.unsqueeze(0) * (G - G_aug)).norm(dim=(-2,-1)).mean()

    def forward(self, X, X_aug, L):
        """Returns (d_M scalar, component dict)."""
        lam = self._lam()
        ts  = self._theta_s(X, X_aug, L)
        ttw = self._theta_tw(X, X_aug)
        tn  = self._theta_n(X, X_aug)
        tc  = self._theta_c(X, X_aug)
        d_M = lam[0]*ts + lam[1]*ttw + lam[2]*tn + lam[3]*tc
        return d_M, {'theta_s': ts, 'theta_tw': ttw, 'theta_n': tn,
                     'theta_c': tc, 'd_M': d_M}


# ─────────────────────────────────────────────────────────────────────────────
# MPLAM — Learnable Augmentation + Manifold Preservation
# X_aug = LTW(X;θ_tw) + N(0,θ_n²I)   A_aug = A⊙Bernoulli(1−θ_r)
# S_aug = S + θ_s N(0,I)
# Joint opt: min_{Θ,W} L_pred + λ‖Θ‖₂ + λ₄·d_M  (no bi-level)
# ─────────────────────────────────────────────────────────────────────────────

class MPLAM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.theta_tw = nn.Parameter(torch.tensor(3.5))
        self.theta_r  = nn.Parameter(torch.tensor(0.12))
        self.theta_s  = nn.Parameter(torch.tensor(0.03))
        self.theta_n  = nn.Parameter(torch.tensor(0.07))
        # Warp network: mean over nodes → attention matrix over time
        self.warp_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.manifold = ManifoldPreservation(num_nodes)

    @staticmethod
    def _norm_lap(adj: torch.Tensor) -> torch.Tensor:
        d = adj.sum(-1).clamp(1e-6) ** (-0.5)
        return torch.eye(adj.shape[-1], device=adj.device) - \
               d.unsqueeze(-1) * adj * d.unsqueeze(-2)

    def forward(self, x, adj, coords):
        # x: (B,N,T,F)
        tw = self.theta_tw.clamp(0.1, 10.0)
        tr = self.theta_r.clamp(0.0, 0.3)
        ts = self.theta_s.clamp(0.0, 0.1)
        tn = self.theta_n.clamp(0.0, 0.15)

        # Learnable time-warp: attention over T using node-mean features
        wf    = self.warp_net(x.mean(dim=1))               # (B,T,d)
        W     = F.softmax(tw * (wf @ wf.transpose(-2,-1)) /
                          math.sqrt(wf.shape[-1]), dim=-1) # (B,T,T)
        x_tw  = torch.einsum('bts,bnsf->bntf', W, x)

        adj_aug = (adj * torch.bernoulli((1-tr) * torch.ones_like(adj))
                   if self.training else adj)
        x_aug = x_tw + tn * torch.randn_like(x_tw)
        s_aug = coords + ts * torch.randn_like(coords)

        if self.training:
            L   = self._norm_lap(adj)
            d_M, mp = self.manifold(x, x_aug, L)
        else:
            d_M = x.new_zeros(1).squeeze()
            mp  = {}

        return x_aug, adj_aug, s_aug, d_M, mp

    def aug_reg_loss(self):
        return self.theta_tw**2 + self.theta_r**2 + self.theta_s**2 + self.theta_n**2


# ─────────────────────────────────────────────────────────────────────────────
# MFPE — Multi-scale Frequency Positional Embedding
# ST_e = Σ_k α_k · W_k^proj E_k,   H^(0) = W_proj[X_aug ‖ ST_e]
# ─────────────────────────────────────────────────────────────────────────────

class MFPE(nn.Module):

    def __init__(self, d_pe=64, d_te=64, d_h=64, max_len=5000):
        super().__init__()
        self.omega1   = nn.Parameter(torch.tensor(2.0 * math.pi / 24.0))
        self.omega2   = nn.Parameter(torch.tensor(2.0 * math.pi / 168.0))
        self.attn_mlp = nn.Linear(d_pe + d_te + 5, 4)
        self.projs    = nn.ModuleList([
            nn.Sequential(nn.Linear(d_pe, d_h), nn.LayerNorm(d_h)),
            nn.Sequential(nn.Linear(d_te, d_h), nn.LayerNorm(d_h)),
            nn.Sequential(nn.Linear(4,    d_h), nn.LayerNorm(d_h)),
            nn.Sequential(nn.Linear(1,    d_h), nn.LayerNorm(d_h)),
        ])
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        dpe = torch.exp(torch.arange(0, d_pe, 2).float() * (-math.log(10000.0)/d_pe))
        dte = torch.exp(torch.arange(0, d_te, 2).float() * (-math.log(10000.0)/d_te))
        pe, te = torch.zeros(max_len, d_pe), torch.zeros(max_len, d_te)
        pe[:,0::2] = torch.sin(pos*dpe); pe[:,1::2] = torch.cos(pos*dpe)
        te[:,0::2] = torch.sin(pos*dte); te[:,1::2] = torch.cos(pos*dte)
        self.register_buffer('pe_cache', pe)
        self.register_buffer('te_cache', te)

    def sinusoidal_temporal_encoding(self, length: int) -> torch.Tensor:
        return self.te_cache[:length]

    def _topo(self, adj, N):
        d = adj.sum(-1).clamp(1e-6)
        i = d.pow(-0.5)
        return (i.unsqueeze(-1) * adj * i.unsqueeze(-2)) @ torch.ones(N, 1, device=adj.device)

    def _freqs(self, T, device):
        t = torch.arange(T, dtype=torch.float, device=device)
        return torch.stack([torch.sin(self.omega1*t), torch.cos(self.omega1*t),
                            torch.sin(self.omega2*t), torch.cos(self.omega2*t)], dim=-1)

    def forward(self, N: int, T: int, adj: torch.Tensor) -> torch.Tensor:
        """Returns ST_e: (N, T, d_h) — broadcast over B in caller."""
        pe, te = self.pe_cache[:N], self.te_cache[:T]
        tm, fe = self._freqs(T, adj.device), self._topo(adj, N)
        pe_e = pe.unsqueeze(1).expand(-1, T, -1)
        te_e = te.unsqueeze(0).expand(N, -1, -1)
        tm_e = tm.unsqueeze(0).expand(N, -1, -1)
        fe_e = fe.unsqueeze(1).expand(-1, T, -1)
        alpha = F.softmax(self.attn_mlp(torch.cat([pe_e, te_e, tm_e, fe_e], dim=-1)), dim=-1)
        embs  = [self.projs[0](pe_e), self.projs[1](te_e),
                 self.projs[2](tm_e), self.projs[3](fe_e)]
        return sum(alpha[..., k:k+1] * embs[k] for k in range(4))


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 1 — FDTA: Frequency-Domain Temporal Augmentation
# Learnable spectral filter W=W_r+jW_i via RFFT circular convolution.
# O(BNT log T) captures rush-hour/weekly periodicities missed by attention.
# out = LN(X + sigmoid([X, X_freq]) · X_freq)
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyMixing(nn.Module):

    def __init__(self, d_h: int, seq_len: int):
        super().__init__()
        fd       = seq_len // 2 + 1
        self.W_r = nn.Parameter(torch.ones(fd, d_h))    # identity filter init
        self.W_i = nn.Parameter(torch.zeros(fd, d_h))   # zero phase-shift init
        self.gate = nn.Sequential(nn.Linear(2*d_h, d_h), nn.Sigmoid())
        self.ln   = nn.LayerNorm(d_h)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,T,d)
        B, N, T, d = x.shape
        flat  = x.reshape(B*N, T, d)
        xf    = torch.fft.rfft(flat, n=T, dim=1)          # (B*N, fd, d)
        fd    = xf.shape[1]
        Wr, Wi = self.W_r[:fd], self.W_i[:fd]
        x_freq = torch.fft.irfft(
            torch.complex(xf.real*Wr - xf.imag*Wi,
                          xf.real*Wi + xf.imag*Wr), n=T, dim=1
        ).reshape(B, N, T, d)
        return self.ln(x + self.gate(torch.cat([x, x_freq], dim=-1)) * x_freq)


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 2 — ADGD: Asymmetric Directed Graph Diffusion
# Learns asymmetric A[i,j]=σ(srcᵢ·tgtⱼ/√d) from separate src/tgt MLPs.
# Parallel k-hop forward (downstream) + backward (upstream) diffusions,
# fused by per-scale direction gate g.
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAdjacency(nn.Module):
    """Asymmetric adaptive adjacency: src MLP × tgt MLP → (N,N)."""

    def __init__(self, d_h: int, dropout: float = 0.1):
        super().__init__()
        self.mlp_src = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_h, d_h))
        self.mlp_tgt = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_h, d_h))

    def forward(self, h, _):
        hm  = h.mean(dim=(0, 2))                           # (N, d)
        src = self.mlp_src(hm)
        tgt = self.mlp_tgt(hm)
        return torch.sigmoid(src @ tgt.t() / math.sqrt(src.shape[-1]))  # (N,N)


class MultiScaleGCN(nn.Module):
    """
    k-hop forward/backward diffusion for k∈{1,3,5,7}.
    Direction gate g = σ(MLP([pool_fwd, pool_bwd])) per scale.
    β-softmax over scale global-mean scores for final fusion.
    """

    def __init__(self, d_h: int, scales=(1, 3, 5, 7), dropout: float = 0.1):
        super().__init__()
        self.scales    = scales
        self.convs_fwd = nn.ModuleList([
            nn.Sequential(nn.Linear(d_h, d_h), nn.LayerNorm(d_h), nn.Dropout(dropout))
            for _ in scales])
        self.convs_bwd = nn.ModuleList([
            nn.Sequential(nn.Linear(d_h, d_h), nn.LayerNorm(d_h), nn.Dropout(dropout))
            for _ in scales])
        self.residuals  = nn.ModuleList([nn.Linear(d_h, d_h) for _ in scales])
        self.dir_gates  = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d_h, d_h), nn.ReLU(),
                          nn.Linear(d_h, 1), nn.Sigmoid())
            for _ in scales])
        self.scale_mlp  = nn.Sequential(nn.Linear(d_h, d_h//2), nn.ReLU(),
                                        nn.Linear(d_h//2, 1))
        self.out_norm   = nn.LayerNorm(d_h)

    @staticmethod
    def _norm(A: torch.Tensor) -> torch.Tensor:
        d = A.sum(-1).clamp(1e-6)
        i = d.pow(-0.5)
        return i.unsqueeze(-1) * A * i.unsqueeze(-2)

    def _powers(self, adj: torch.Tensor):
        Af, Ab = self._norm(adj), self._norm(adj.t())
        pf, pb = {1: Af}, {1: Ab}
        for k in self.scales:
            if k not in pf:
                cf, cb = Af, Ab
                for _ in range(k - 1):
                    cf, cb = cf @ Af, cb @ Ab
                pf[k], pb[k] = cf, cb
        return pf, pb

    def forward(self, h: torch.Tensor,
                adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pf, pb = self._powers(adj)
        scale_outs, globals_ = [], []
        for i, k in enumerate(self.scales):
            h_fwd = torch.einsum('mn,bntd->bmtd', pf[k], h)
            h_bwd = torch.einsum('mn,bntd->bmtd', pb[k], h)
            gf    = h_fwd.mean((0, 1, 2))
            gb    = h_bwd.mean((0, 1, 2))
            g     = self.dir_gates[i](torch.cat([gf, gb], dim=-1))    # (1,)
            hk    = F.relu(g * self.convs_fwd[i](h_fwd) +
                           (1-g) * self.convs_bwd[i](h_bwd) +
                           self.residuals[i](h))
            scale_outs.append(hk)
            globals_.append(hk.mean((0, 1, 2)))
        betas = F.softmax(
            torch.stack([self.scale_mlp(g).squeeze(-1) for g in globals_]), dim=0)
        return self.out_norm(
            sum(betas[i] * scale_outs[i] for i in range(len(self.scales)))
        ), betas


class HierarchicalPooling(nn.Module):

    def __init__(self, pool_ratio: float = 0.5):
        super().__init__()
        self.pool_ratio = pool_ratio

    def forward(self, h, adj):
        # h: (B,N,T,d)  adj: (N,N)
        N   = h.shape[1]
        scores = h.norm(p=2, dim=-1).mean(dim=(0, 2)) / math.sqrt(h.shape[-1])
        k   = max(1, int(N * self.pool_ratio))
        _, idx = torch.topk(scores, k)
        idx, _ = idx.sort()
        return h[:, idx], adj[idx][:, idx], idx


class SpatialProcessingModule(nn.Module):

    def __init__(self, d_h=64, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers    = num_layers
        self.adaptive_adjs = nn.ModuleList([AdaptiveAdjacency(d_h, dropout)
                                            for _ in range(num_layers)])
        self.gcns          = nn.ModuleList([MultiScaleGCN(d_h, dropout=dropout)
                                            for _ in range(num_layers)])
        self.layer_norms   = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(num_layers)])
        self.pooling       = HierarchicalPooling(0.5)
        self.dropout       = nn.Dropout(dropout)
        self.stored_adjs:  List[torch.Tensor] = []
        self.stored_betas: List[torch.Tensor] = []

    def forward(self, h: torch.Tensor, static_adj: torch.Tensor) -> torch.Tensor:
        self.stored_adjs, self.stored_betas = [], []
        pool_indices, cur_adj = [], static_adj
        orig_N = h.shape[1]

        for l in range(self.num_layers):
            res  = h
            la   = self.adaptive_adjs[l](h, cur_adj)
            self.stored_adjs.append(la)
            h_new, beta = self.gcns[l](h, la)
            self.stored_betas.append(beta)
            h_new = self.dropout(self.layer_norms[l](h_new))
            if h_new.shape[1] == res.shape[1]:
                h_new = h_new + res
            if l in (1, 3):
                pre_N = h_new.shape[1]
                h_new, cur_adj, idx = self.pooling(h_new, la)
                pool_indices.append((idx, pre_N))
            else:
                cur_adj = la
            h = h_new

        return self._unpool(h, orig_N, pool_indices)

    def _unpool(self, h, orig_N, pool_indices):
        for idx, pre_N in reversed(pool_indices):
            B, _, T, d = h.shape
            h_up = torch.zeros(B, pre_N, T, d, device=h.device)
            h_up[:, idx] = h
            missing = torch.ones(pre_N, dtype=torch.bool, device=h.device)
            missing[idx] = False
            mi = missing.nonzero(as_tuple=True)[0]
            if mi.numel() > 0:
                h_up[:, mi] = h.mean(dim=1, keepdim=True).expand(-1, mi.numel(), -1, -1)
            h = h_up
        return h

    def spatial_reg_loss(self) -> torch.Tensor:
        if not self.stored_adjs:
            return torch.tensor(0.0)
        dev   = self.stored_adjs[0].device
        adj_s = torch.tensor(0.0, device=dev)
        for l in range(len(self.stored_adjs) - 1):
            if self.stored_adjs[l+1].shape == self.stored_adjs[l].shape:
                adj_s = adj_s + (self.stored_adjs[l+1] - self.stored_adjs[l]).pow(2).sum()
        n    = self.stored_betas[0].shape[0]
        unif = torch.full_like(self.stored_betas[0], 1.0 / n)
        return adj_s + sum((b - unif).pow(2).sum() for b in self.stored_betas)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Processing — Lipschitz-constrained attention + FDTA per layer
# FDTA (FrequencyMixing) runs in parallel with each transformer layer;
# adaptive gate mixes attention output with spectral output.
# ─────────────────────────────────────────────────────────────────────────────

class LipschitzMHA(nn.Module):
    """Multi-head attention with Lipschitz normalisation ‖out‖ ≤ K."""

    def __init__(self, d_h, n_heads=8, dropout=0.1, lipschitz_k=1.5):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_h // n_heads
        self.lipschitz_k = lipschitz_k
        self.q  = nn.Linear(d_h, d_h)
        self.k  = nn.Linear(d_h, d_h)
        self.v  = nn.Linear(d_h, d_h)
        self.o  = nn.Linear(d_h, d_h)
        self.dp = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_h)

    def forward(self, x):
        B, N, T, d = x.shape
        H, dk = self.n_heads, self.d_k
        f = x.reshape(B*N, T, d)
        def proj(lin): return lin(f).view(B*N, T, H, dk).transpose(1, 2)
        q, k, v = proj(self.q), proj(self.k), proj(self.v)
        a   = self.dp(F.softmax(q @ k.transpose(-2,-1) / math.sqrt(dk), dim=-1))
        out = self.o((a @ v).transpose(1,2).contiguous().view(B*N, T, d))
        with torch.no_grad():
            scale = (out.norm(2, dim=-1, keepdim=True).clamp(1e-8) /
                     self.lipschitz_k).clamp(min=1.0)
        return self.ln(out / scale + f).reshape(B, N, T, d)


class TemporalProcessingModule(nn.Module):
    """
    Input fusion → L × (LipschitzMHA + FFN + FDTA gate).
    FDTA branch runs per-layer for progressive spectral refinement.
    """

    def __init__(self, d_h=64, d_m=16, n_heads=8, num_layers=4,
                 seq_len=12, dropout=0.1, lipschitz_k=1.5):
        super().__init__()
        self.lipschitz_k   = lipschitz_k
        self.lipschitz_val = torch.tensor(0.0)
        self.meta_proj = nn.Sequential(nn.Linear(d_m, d_h), nn.LayerNorm(d_h), nn.Dropout(dropout))
        self.spat_proj = nn.Sequential(nn.Linear(d_h, d_h), nn.LayerNorm(d_h), nn.Dropout(dropout))
        self.te_proj   = nn.Sequential(nn.Linear(64,   d_h), nn.LayerNorm(d_h), nn.Dropout(dropout))
        self.gate      = nn.Sequential(nn.Linear(3*d_h, d_h), nn.Sigmoid())
        self.attn      = nn.ModuleList([LipschitzMHA(d_h, n_heads, dropout, lipschitz_k)
                                        for _ in range(num_layers)])
        self.ffn       = nn.ModuleList([
            nn.Sequential(nn.Linear(d_h, 4*d_h), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(4*d_h, d_h), nn.Dropout(dropout))
            for _ in range(num_layers)])
        self.ln        = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.freq_mix  = nn.ModuleList([FrequencyMixing(d_h, seq_len) for _ in range(num_layers)])
        self.freq_gate = nn.Sequential(nn.Linear(2*d_h, d_h), nn.Sigmoid())

    def forward(self, h_sp, meta, te):
        B, N, T, d = h_sp.shape
        mp = self.meta_proj(meta).unsqueeze(1).unsqueeze(2).expand(-1, N, T, -1)
        sp = self.spat_proj(h_sp)
        tp = self.te_proj(te).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        h  = F.relu(self.gate(torch.cat([mp, sp, tp], dim=-1)) * (mp + sp + tp))

        for l in range(self.num_layers):
            ha  = self.attn[l](h)
            h_t = self.ln[l](ha + self.ffn[l](ha.reshape(B*N, T, d)).reshape(B, N, T, d))
            h_f = self.freq_mix[l](h_t)
            g   = self.freq_gate(torch.cat([h_t, h_f], dim=-1))
            h   = g * h_t + (1.0 - g) * h_f

        dh  = (h[:,:,1:,:] - h[:,:,:-1,:]).detach()
        dte = (te[1:] - te[:-1]).detach()
        self.lipschitz_val = dh.norm() / dte.norm().clamp(1e-8)
        return h

    def temp_reg_loss(self) -> torch.Tensor:
        lv = self.lipschitz_val
        if not isinstance(lv, torch.Tensor):
            lv = torch.tensor(lv)
        return F.relu(lv - self.lipschitz_k)


class FusionModule(nn.Module):

    def __init__(self, d_h=64, d_f=128, dropout=0.1):
        super().__init__()
        self.ps = nn.Sequential(nn.Linear(d_h, d_f), nn.LayerNorm(d_f), nn.Dropout(dropout))
        self.pt = nn.Sequential(nn.Linear(d_h, d_f), nn.LayerNorm(d_f), nn.Dropout(dropout))
        self.pi = nn.Sequential(nn.Linear(d_h, d_f), nn.LayerNorm(d_f), nn.Dropout(dropout))
        self.a  = nn.Sequential(nn.Linear(3*d_f, 128), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(128, 3))
        self.o  = nn.Sequential(nn.Linear(d_f, d_f), nn.LayerNorm(d_f), nn.ReLU())

    def forward(self, hs, ht, hi):
        ps, pt, pi = self.ps(hs), self.pt(ht), self.pi(hi)
        a = F.softmax(self.a(torch.cat([ps, pt, pi], dim=-1)), dim=-1)
        return self.o(a[...,0:1]*ps + a[...,1:2]*pt + a[...,2:3]*pi)


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 3 — IPR: Iterative Prediction Refinement
# Y_1 = Y_0 + α·tanh(Head([CrossAttn(E(Y_0), H_ctx), E(Y_0)]))
# α init 0.1 ensures stable training; tanh bounds each correction step.
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRefinement(nn.Module):
    """Cross-attention correction step over encoder context H_ctx."""

    def __init__(self, d_o: int, d_f: int, d_r: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc    = nn.Linear(d_o, d_r)
        self.q_proj = nn.Linear(d_r, d_r)
        self.k_proj = nn.Linear(d_f, d_r)
        self.v_proj = nn.Linear(d_f, d_r)
        self.head   = nn.Sequential(nn.Linear(2*d_r, d_r), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(d_r, d_o))
        self.ln     = nn.LayerNorm(d_r)
        self.alpha  = nn.Parameter(torch.tensor(0.1))

    def forward(self, y0: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        # y0: (B,N,H,d_o)   h_ctx: (B,N,T,d_f)
        B, N, Hp, _ = y0.shape
        T_ctx = h_ctx.shape[2]
        e0  = self.enc(y0).reshape(B*N, Hp, -1)
        ctx = h_ctx.reshape(B*N, T_ctx, -1)
        q   = self.q_proj(e0)
        k, v = self.k_proj(ctx), self.v_proj(ctx)
        att = self.ln(F.softmax(q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1]),
                                dim=-1) @ v)                      # (B*N, H, d_r)
        delta = self.head(torch.cat([att, e0], dim=-1)).reshape(B, N, Hp, -1)
        return y0 + self.alpha * torch.tanh(delta)


# ─────────────────────────────────────────────────────────────────────────────
# Dual-Path Decoder  (fixed for any T vs H relationship)
#
# Spectral path:
#   h_sp = Σ_k θ_k T_k(L̃) H    (Chebyshev over spatial dim, full T)
#   sp   = TemporalAttnPool(h_sp) → (B,N,1,d_f) → expand+MLP → (B,N,H,d_o)
#   (Temporal attention pooling avoids the T-slice bug; works for all T,H)
#
# Temporal path:
#   te[T:T+H] queries → cross-attend h → (B,N,H,d_o)
#
# Gate + IPR correction:
#   g = σ(W_g[sp ‖ tp])
#   Y_0 = g·sp + (1-g)·tp
#   Y_1 = IPR(Y_0, h)
# ─────────────────────────────────────────────────────────────────────────────

class DualPathDecoder(nn.Module):

    def __init__(self, d_f=128, d_o=1, horizon=12, K=3, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.K       = K

        # Chebyshev coefficients
        self.theta_s = nn.ParameterList([nn.Parameter(torch.randn(1)*0.1)
                                         for _ in range(K+1)])

        # Spectral path: temporal attention pooling T → 1, then project to H
        self.temp_pool_q = nn.Parameter(torch.randn(1, 1, d_f) * 0.02)
        self.temp_pool_k = nn.Linear(d_f, d_f)
        self.temp_pool_v = nn.Linear(d_f, d_f)
        self.sp_expand   = nn.Linear(d_f, d_f * horizon)   # (d_f,) → (H·d_f,)
        self.sc          = nn.Sequential(nn.Linear(d_f, d_f), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(d_f, d_o))

        # Temporal path: horizon query vectors cross-attend to encoder h
        self.qp = nn.Linear(64, d_f)
        self.kp = nn.Linear(d_f, d_f)
        self.vp = nn.Linear(d_f, d_f)
        self.tc = nn.Sequential(nn.Linear(d_f, d_f), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_f, d_o))

        # Gate + refinement
        self.gate   = nn.Sequential(nn.Linear(2*d_o, 64), nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(64, d_o), nn.Sigmoid())
        self.refine = PredictionRefinement(d_o=d_o, d_f=d_f, dropout=dropout)
        self.out_ln = nn.LayerNorm(d_o)

    def _cheby(self, adj: torch.Tensor) -> List[torch.Tensor]:
        d  = adj.sum(-1).clamp(1e-6)
        i  = d.pow(-0.5)
        L  = torch.eye(adj.shape[0], device=adj.device) - i.unsqueeze(-1)*adj*i.unsqueeze(-2)
        Ls = L - torch.eye(adj.shape[0], device=adj.device)
        ps = [torch.eye(adj.shape[0], device=adj.device), Ls]
        for _ in range(2, self.K+1):
            ps.append(2.0*Ls@ps[-1] - ps[-2])
        return ps[:self.K+1]

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                te_full: torch.Tensor, T: int) -> torch.Tensor:
        # h: (B,N,T,d_f)
        B, N, _, d_f = h.shape

        # ── Spectral path ─────────────────────────────────────────────────────
        ps   = self._cheby(adj)
        h_sp = sum(self.theta_s[k] * torch.einsum('mn,bntd->bmtd', ps[k], h)
                   for k in range(self.K+1))               # (B,N,T,d_f)

        # Temporal attention pooling: single query vector summarises T → 1 step
        # This is T-agnostic and works for any relationship between T and H.
        BN   = B * N
        q    = self.temp_pool_q.expand(BN, -1, -1)        # (B*N, 1, d_f)
        k_sp = self.temp_pool_k(h_sp.reshape(BN, T, d_f))
        v_sp = self.temp_pool_v(h_sp.reshape(BN, T, d_f))
        pool = F.softmax(q @ k_sp.transpose(-2,-1) / math.sqrt(d_f), dim=-1) @ v_sp
        # pool: (B*N, 1, d_f)  →  expand to H steps
        pool_h = self.sp_expand(pool.squeeze(1)).reshape(BN, self.horizon, d_f)
        sp     = self.sc(pool_h).reshape(B, N, self.horizon, d_o:=self.sc[-1].out_features)

        # ── Temporal path ──────────────────────────────────────────────────────
        # te_full has shape (T+H, 64); slice future H steps as horizon queries
        te_h   = te_full[T:T + self.horizon]               # (H, 64) — always valid
        fte    = self.qp(te_h)                             # (H, d_f)
        flat   = h.reshape(BN, T, d_f)
        qe     = fte.unsqueeze(0).expand(BN, -1, -1)
        aw     = F.softmax(qe @ self.kp(flat).transpose(-2,-1) / math.sqrt(d_f), dim=-1)
        tp     = self.tc((aw @ self.vp(flat)).reshape(B, N, self.horizon, d_f))

        # ── Gate + IPR ─────────────────────────────────────────────────────────
        g  = self.gate(torch.cat([sp, tp], dim=-1))
        y0 = g * sp + (1.0 - g) * tp                      # (B,N,H,d_o)
        y1 = self.refine(y0, h)                            # IPR correction
        return self.out_ln(y1)                             # (B,N,H,d_o)


# ─────────────────────────────────────────────────────────────────────────────
# MASTNet — top-level model
# ─────────────────────────────────────────────────────────────────────────────

class MASTNet(nn.Module):
    """
    MPLAM(+ManifoldPreservation) → MFPE → SPM(ADGD) → TPM(FDTA) →
    Fusion → DualPathDecoder(IPR)

    Compatible with any in_features F ∈ {1,2,3,4,…} and any horizon H.
    Input:  x:(B,N,T,F)   meta:(B,d_m)   adj registered as buffer (N,N).
    Output: (B,N,H,F)
    """

    def __init__(self, num_nodes: int, input_dim: int = 1, d_h: int = 64,
                 d_m: int = 16, n_heads: int = 8, num_spatial_layers: int = 4,
                 num_temporal_layers: int = 4, horizon: int = 12,
                 seq_len: int = 12, dropout: float = 0.1, adj=None):
        super().__init__()
        self.horizon  = horizon
        self.seq_len  = seq_len
        self.input_dim = input_dim

        self.mplam   = MPLAM(input_dim, d_h, num_nodes)
        self.mfpe    = MFPE(d_pe=64, d_te=64, d_h=d_h)
        # in_proj: fuses raw features (F dims) with positional embedding (d_h dims)
        self.in_proj = nn.Sequential(nn.Linear(input_dim + d_h, d_h),
                                     nn.LayerNorm(d_h), nn.Dropout(dropout))
        self.spatial  = SpatialProcessingModule(d_h, num_spatial_layers, dropout)
        self.temporal = TemporalProcessingModule(d_h, d_m, n_heads,
                                                  num_temporal_layers, seq_len, dropout)
        self.fusion   = FusionModule(d_h, 128, dropout)
        self.decoder  = DualPathDecoder(128, input_dim, horizon, dropout=dropout)

        _adj = (adj if isinstance(adj, torch.Tensor)
                else torch.tensor(adj, dtype=torch.float) if adj is not None
                else torch.eye(num_nodes))
        self.register_buffer('adj',    _adj.float())
        self.register_buffer('coords', torch.zeros(num_nodes, 2))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # x: (B,N,T,F)
        B, N, T, _ = x.shape
        x_a, adj_a, _, d_M, mp = self.mplam(x, self.adj, self.coords)
        st = self.mfpe(N, T, adj_a).unsqueeze(0).expand(B, -1, -1, -1)
        h0 = self.in_proj(torch.cat([x_a, st], dim=-1))   # (B,N,T,d_h)
        hs = self.spatial(h0, adj_a)                       
        te = self.mfpe.sinusoidal_temporal_encoding(T + self.horizon).to(x.device)
        ht = self.temporal(hs, meta, te[:T])               
        hf = self.fusion(hs, ht, h0)                       
        yp = self.decoder(hf, adj_a, te, T)
        return yp, d_M, mp

    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, meta: torch.Tensor,
                       lambda1: float = 0.01, lambda2: float = 0.1,
                       lambda3: float = 0.01, lambda4: float = 0.1):
        """
        J = L_pred + λ1·L_spatial + λ2·L_temp + λ3·‖Θ_aug‖² + λ4·d_M
        Returns (total, pred, spatial, temp, aug, d_M, predictions).
        """
        yp, d_M, _   = self.forward(x, meta)
        pred_loss     = F.l1_loss(yp, y)
        spatial_loss  = self.spatial.spatial_reg_loss()
        temp_loss     = self.temporal.temp_reg_loss()
        aug_loss      = self.mplam.aug_reg_loss()
        total = (pred_loss
                 + lambda1 * spatial_loss
                 + lambda2 * temp_loss
                 + lambda3 * aug_loss
                 + lambda4 * d_M)
        return total, pred_loss, spatial_loss, temp_loss, aug_loss, d_M, yp