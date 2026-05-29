"""
utils.py — utilities for MAST-Net
"""

import os, random, datetime
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model):
    total, trainable = count_parameters(model)
    print('=' * 60)
    print(f'Total parameters:     {total:,}')
    print(f'Trainable parameters: {trainable:,}')
    print('=' * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Masked metrics  (null_val mask excludes missing / zero sensors)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mae(pred, true, null_val: float = 0.0):
    mask = (true != null_val).float()
    mask = mask / mask.mean().clamp(min=1e-8)
    loss = (pred - true).abs() * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()


def masked_mse(pred, true, null_val: float = 0.0):
    mask = (true != null_val).float()
    mask = mask / mask.mean().clamp(min=1e-8)
    loss = (pred - true).pow(2) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()


def masked_rmse(pred, true, null_val: float = 0.0):
    return masked_mse(pred, true, null_val).sqrt()


def masked_mape(pred, true, null_val: float = 0.0):
    mask = (true != null_val).float()
    mask = mask / mask.mean().clamp(min=1e-8)
    loss = ((pred - true).abs() / (true.abs() + 1e-5)) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean() * 100.0


def compute_metrics(pred, true, null_val: float = 0.0) -> dict:
    """All three scalar metrics. pred/true: torch tensors, any shape."""
    return {
        'MAE':  masked_mae(pred,  true, null_val).item(),
        'RMSE': masked_rmse(pred, true, null_val).item(),
        'MAPE': masked_mape(pred, true, null_val).item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-horizon metrics on numpy arrays  (used after loading test_results.npz)
# ─────────────────────────────────────────────────────────────────────────────

def per_horizon_metrics(preds: np.ndarray, gts: np.ndarray,
                        steps=(3, 6, 12)) -> dict:
    """
    preds, gts: (B, N, H, F) numpy arrays in normalised space.
    Returns metrics dict keyed by horizon step (1-indexed).
    Uses feature-0 to match traffic-forecasting conventions.
    """
    H       = preds.shape[2]
    results = {}
    for step in steps:
        if step > H:
            continue
        p = preds[:, :, step - 1, 0].flatten()
        g = gts[:,   :, step - 1, 0].flatten()
        mask  = np.isfinite(p) & np.isfinite(g) & (np.abs(g) > 1e-5)
        if mask.sum() == 0:
            continue
        err   = p[mask] - g[mask]
        results[step] = {
            'MAE':  float(np.abs(err).mean()),
            'RMSE': float(np.sqrt((err ** 2).mean())),
            'MAPE': float(np.abs(err / g[mask]).mean() * 100),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'loss':      loss,
    }, path)


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['loss']


# ─────────────────────────────────────────────────────────────────────────────
# Experiment directory
# ─────────────────────────────────────────────────────────────────────────────

def create_exp_dir(base: str = 'experiments') -> str:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    d  = os.path.join(base, f'exp_{ts}')
    os.makedirs(os.path.join(d, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(d, 'logs'),        exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(data, mean, std):
    return data * std + mean


def format_time(seconds: float) -> str:
    if seconds < 60:   return f'{seconds:.1f}s'
    if seconds < 3600: return f'{int(seconds//60)}m {seconds%60:.0f}s'
    return f'{int(seconds//3600)}h {int((seconds%3600)//60)}m'