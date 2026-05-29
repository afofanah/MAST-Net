import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# File loading  (handles .npz / .npy / .h5 / .pkl / .csv)
# ─────────────────────────────────────────────────────────────────────────────

def load_traffic_data(data_dir: str, dataset_name: str = None) -> np.ndarray:
    """Try common file names and formats; return raw ndarray."""
    candidates = ([dataset_name] if dataset_name else []) + [
        'data', 'traffic', 'pems', 'metr', 'bay', 'expy', 'atp'
    ]
    for name in candidates:
        for ext in ['.npz', '.npy', '.h5', '.pkl', '.csv']:
            path = os.path.join(data_dir, name + ext)
            if not os.path.exists(path):
                continue
            if ext == '.npz':
                f = np.load(path)
                return f['data'] if 'data' in f else f[list(f.keys())[0]]
            if ext == '.npy':
                return np.load(path)
            if ext == '.h5':
                import h5py
                with h5py.File(path, 'r') as f:
                    return f['data'][:] if 'data' in f else f[list(f.keys())[0]][:]
            if ext == '.pkl':
                with open(path, 'rb') as f:
                    d = pickle.load(f)
                    return d.get('data', list(d.values())[0]) \
                           if isinstance(d, dict) else d
            if ext == '.csv':
                return np.loadtxt(path, delimiter=',')
    raise FileNotFoundError(f'No traffic data found in {data_dir}')


def load_adjacency(data_dir: str, dataset_name: str = None) -> Optional[np.ndarray]:
    """Try common adj file names; return ndarray or None."""
    patterns = ([f'adj_{dataset_name}', 'adj'] if dataset_name else []) + \
               ['adj', 'adjacency', 'distance']
    for p in patterns:
        for ext in ['.npy', '.pkl', '.csv', '.npz']:
            path = os.path.join(data_dir, p + ext)
            if not os.path.exists(path):
                continue
            if ext == '.npy':
                return np.load(path)
            if ext == '.pkl':
                with open(path, 'rb') as f:
                    a = pickle.load(f)
                    return a[0] if isinstance(a, (tuple, list)) else a
            if ext == '.csv':
                d = np.loadtxt(path, delimiter=',')
                s = np.std(d)
                a = np.exp(-d ** 2 / (2 * (s if s > 0 else 1.0) ** 2))
                a[a < 0.1] = 0
                return a
            if ext == '.npz':
                f = np.load(path)
                return f['adj'] if 'adj' in f else f[list(f.keys())[0]]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_data(data: np.ndarray) -> np.ndarray:
    assert isinstance(data, np.ndarray), 'data must be ndarray'
    assert 2 <= data.ndim <= 3, f'data must be 2D or 3D, got {data.shape}'
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0)
    if np.isinf(data).any():
        data = np.clip(data, -1e10, 1e10)
    return data


def validate_adj(adj: Optional[np.ndarray], n_nodes: int) -> Optional[np.ndarray]:
    if adj is None:
        return None
    assert adj.shape == (n_nodes, n_nodes), \
        f'adj shape mismatch: {adj.shape} vs ({n_nodes},{n_nodes})'
    assert not np.isnan(adj).any(), 'adj contains NaN'
    if not np.allclose(adj, adj.T, atol=1e-6):
        adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Raw-data shape normalisation  (T,N,F) canonical form
# ─────────────────────────────────────────────────────────────────────────────

def to_TNF(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    if data.ndim == 2:
        return data[:, :, None]
    if data.ndim == 3:
        if data.shape[2] > data.shape[0] and data.shape[0] < 1000:
            return data.transpose(2, 0, 1)   # (N,F,T) → (T,N,F)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation  (train stats applied to val/test — correct approach)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(data: np.ndarray, method: str = 'zscore',
              mean=None, std=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 
    if method == 'zscore':
        if mean is None:
            mean = data.mean(axis=(0, 1), keepdims=True).astype(np.float32)
            std  = data.std(axis=(0, 1),  keepdims=True).astype(np.float32)
        std = np.where(std == 0, 1.0, std)
        return ((data - mean) / std).astype(np.float32), mean, std

    if method == 'minmax':
        if mean is None:
            mean = data.min(axis=(0, 1), keepdims=True).astype(np.float32)
            std  = (data.max(axis=(0, 1), keepdims=True) - mean).astype(np.float32)
        std = np.where(std == 0, 1.0, std)
        return ((data - mean) / std).astype(np.float32), mean, std

    return data.astype(np.float32), np.zeros((1,1,1), np.float32), \
           np.ones((1,1,1), np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Meta features  (rich 16-dim temporal encoding per timestep)
# ─────────────────────────────────────────────────────────────────────────────

def generate_meta_features(n_timesteps: int, interval_minutes: int = 5,
                            d_m: int = 16) -> np.ndarray:
 
    slots = (24 * 60) // interval_minutes
    t     = np.arange(n_timesteps)
    tod   = (t % slots) * interval_minutes / (24 * 60)          # ∈ [0,1)
    dow   = (t // slots) % 7 / 7.0

    cols = [
        np.sin(2 * np.pi * tod),
        np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow),
        np.cos(2 * np.pi * dow),
        ((t // slots) % 7 >= 5).astype(np.float32),             # weekend
        ((7  <= (t % slots) * interval_minutes // 60)           # rush hour
         & ((t % slots) * interval_minutes // 60 <= 9)
         | (17 <= (t % slots) * interval_minutes // 60)
         & ((t % slots) * interval_minutes // 60 <= 19)
        ).astype(np.float32),
    ]
    # Higher harmonics until we reach d_m
    k = 2
    while len(cols) < d_m - 1:
        cols += [np.sin(k * 2 * np.pi * tod),
                 np.cos(k * 2 * np.pi * tod)]
        k += 1

    meta = np.stack(cols[:d_m], axis=1).astype(np.float32)  # (T, d_m)
    if meta.shape[1] < d_m:
        pad  = np.zeros((n_timesteps, d_m - meta.shape[1]), np.float32)
        meta = np.concatenate([meta, pad], axis=1)
    return meta


def load_meta_knowledge(data_dir: str, n_timesteps: int,
                         interval_minutes: int = 5, d_m: int = 16) -> np.ndarray:
    """Load meta from file if present, else generate from timestamps."""
    for fname in ['meta.npy', 'meta.npz', 'weather.npy', 'events.npy']:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        m = (np.load(path).astype(np.float32) if fname.endswith('.npy')
             else np.load(path)['meta'].astype(np.float32))
        if m.shape[0] == n_timesteps:
            # Pad / truncate to d_m
            if m.shape[1] < d_m:
                m = np.concatenate(
                    [m, np.zeros((n_timesteps, d_m - m.shape[1]), np.float32)],
                    axis=1)
            return m[:, :d_m]
    return generate_meta_features(n_timesteps, interval_minutes, d_m)


# ─────────────────────────────────────────────────────────────────────────────
# Normalised adjacency helper
# ─────────────────────────────────────────────────────────────────────────────

def normalized_adj(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} (A+I) D^{-1/2}."""
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = np.sum(A, axis=1, keepdims=True).clip(1e-6) ** (-0.5)
    return (d * A * d.T).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    def __init__(self, data: np.ndarray, adj: np.ndarray,
                 seq_len: int, pred_len: int,
                 meta: np.ndarray,
                 mean: Optional[np.ndarray] = None,
                 std:  Optional[np.ndarray] = None):
        # data: (T_split, N, F)   meta: (T_split, d_m)
        assert data.ndim == 3
        T, N, F        = data.shape
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.N, self.F = N, F

        # Store normalisation stats for external denormalisation
        self.mean = mean if mean is not None else \
                    data.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std  = std  if std  is not None else \
                    data.std(axis=(0, 1),  keepdims=True).astype(np.float32)
        self.std  = np.where(self.std < 1e-6, 1.0, self.std)

        self._data = data.astype(np.float32)   # (T, N, F) — already normalised
        self._meta = meta.astype(np.float32)   # (T, d_m)
        self._n    = T - seq_len - pred_len + 1
        assert self._n > 0, 'Insufficient data for seq_len + pred_len'

        self.adj = torch.FloatTensor(normalized_adj(adj))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        # Raw slices are (T, N, F); transpose to (N, T, F) for MASTNet
        x = torch.FloatTensor(
            self._data[i: i + self.seq_len]
        ).permute(1, 0, 2)                                     # (N, T, F)

        y = torch.FloatTensor(
            self._data[i + self.seq_len: i + self.seq_len + self.pred_len]
        ).permute(1, 0, 2)                                     # (N, H, F)

        meta = torch.FloatTensor(
            self._meta[i + self.seq_len - 1]                   # (d_m,)
        )
        return x, y, meta

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        """arr: (..., F) — inverse of zscore/minmax normalisation."""
        return arr * self.std[0] + self.mean[0]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder  (single data-dir, any format)
# ─────────────────────────────────────────────────────────────────────────────

def split(data: np.ndarray, train_ratio: float = 0.6,
          val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n  = data.shape[0]
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return data[:t1], data[t1:t2], data[t2:]


def prepare_traffic_dataset(
        data_dir: str,
        dataset_name: str = None,
        seq_len: int = 12,
        pred_len: int = 12,
        train_ratio: float = 0.6,
        val_ratio:   float = 0.2,
        normalize_method: str = 'zscore',
        interval_minutes: int = 5,
        d_m: int = 16,
) -> Tuple['TrafficDataset', 'TrafficDataset', 'TrafficDataset',
           torch.Tensor, Dict]:

    assert os.path.exists(data_dir), f'data_dir not found: {data_dir}'

    raw  = validate_data(load_traffic_data(data_dir, dataset_name))
    data = to_TNF(raw)                             # (T, N, F)
    T, N, F = data.shape
    assert T >= seq_len + pred_len and N >= 2

    adj_raw = validate_adj(load_adjacency(data_dir, dataset_name), N)
    adj     = adj_raw if adj_raw is not None else np.eye(N, dtype=np.float32)

    tr_d, va_d, te_d = split(data, train_ratio, val_ratio)

    # Fit normalisation on train only
    tr_n, mean, std = normalize(tr_d, normalize_method)
    va_n, *_        = normalize(va_d, normalize_method, mean, std)
    te_n, *_        = normalize(te_d, normalize_method, mean, std)

    # Meta features — generate per split length
    tr_m = load_meta_knowledge(data_dir, tr_n.shape[0], interval_minutes, d_m)
    va_m = load_meta_knowledge(data_dir, va_n.shape[0], interval_minutes, d_m)
    te_m = load_meta_knowledge(data_dir, te_n.shape[0], interval_minutes, d_m)

    train_ds = TrafficDataset(tr_n, adj, seq_len, pred_len, tr_m, mean, std)
    val_ds   = TrafficDataset(va_n, adj, seq_len, pred_len, va_m, mean, std)
    test_ds  = TrafficDataset(te_n, adj, seq_len, pred_len, te_m, mean, std)

    metadata = {
        'mean': mean, 'std': std,
        'n_nodes': N, 'n_features': F, 'n_timesteps': T,
        'seq_len': seq_len, 'pred_len': pred_len,
        'normalize_method': normalize_method,
    }
    return train_ds, val_ds, test_ds, torch.FloatTensor(adj), metadata


# ─────────────────────────────────────────────────────────────────────────────
# DataManager  (multi-dataset training — source pre-train + target fine-tune)
# ─────────────────────────────────────────────────────────────────────────────

class DataManager:

    def __init__(self, data_config: Dict, task_config: Dict):
        self.data_config  = data_config
        self.task_config  = task_config
        self._cache: Dict[str, TrafficDataset] = {}

    def _load(self, dataset_name: str, stage: str,
              target_days: int = 3) -> TrafficDataset:
        key = f"{dataset_name}_{stage}_{target_days}"
        if key in self._cache:
            return self._cache[key]

        cfg  = self.data_config[dataset_name]
        his  = self.task_config['his_num']
        pred = self.task_config['pred_num']
        d_m  = self.task_config.get('d_m', 16)
        iv   = self.task_config.get('interval_min', 5)
        norm = self.task_config.get('normalize_method', 'zscore')
        raw  = validate_data(load_traffic_data(cfg.get('data_dir', '.'),
                                               cfg.get('dataset_name')))
        data = to_TNF(raw)                                      # (T, N, F)
        if 'dataset_path' in cfg:
            data = to_TNF(validate_data(np.load(cfg['dataset_path'])))

        T, N, F = data.shape
        if 'adjacency_matrix_path' in cfg:
            adj_raw = np.load(cfg['adjacency_matrix_path']).astype(np.float32)
        else:
            adj_raw = load_adjacency(cfg.get('data_dir', '.'),
                                     cfg.get('dataset_name'))
        adj = validate_adj(adj_raw, N) if adj_raw is not None \
              else np.eye(N, dtype=np.float32)

        t1 = int(T * 0.7)
        t2 = int(T * 0.8)
        splits = {
            'source': data[:t1],
            'val':    data[t1:t2],
            'target': data[:min(288 * target_days, t1)],
            'test':   data[t2:],
        }
        Xs = splits[stage]
        _, mean, std = normalize(data[:t1], norm)
        Xs_n, _, _ = normalize(Xs, norm, mean, std)
        meta = generate_meta_features(Xs_n.shape[0], iv, d_m)

        ds = TrafficDataset(Xs_n, adj, his, pred, meta, mean, std)
        self._cache[key] = ds
        return ds

    def create_loaders(self, test_dataset: str, target_days: int = 3,
                       batch_size: int = 32, test_batch_size: int = 32
                       ) -> Dict:
        loaders: Dict = {}

        source = {}
        for name in self.data_config['data_keys']:
            if name == test_dataset:
                continue
            ds = self._load(name, 'source', target_days)
            source[name] = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False, drop_last=True)
        loaders['source'] = source

        val_ds  = self._load(test_dataset, 'val',    target_days)
        tgt_ds  = self._load(test_dataset, 'target', target_days)
        test_ds = self._load(test_dataset, 'test',   target_days)

        loaders['val']    = DataLoader(val_ds,  batch_size=test_batch_size,
                                       shuffle=False, num_workers=0)
        loaders['target'] = DataLoader(tgt_ds,  batch_size=batch_size,
                                       shuffle=True,  num_workers=0,
                                       pin_memory=False)
        loaders['test']   = DataLoader(test_ds, batch_size=test_batch_size,
                                       shuffle=False, num_workers=0)

        loaders['test_stats'] = {'mean': test_ds.mean, 'std': test_ds.std}
        loaders['test_adj']   = test_ds.adj
        loaders['test_N']     = test_ds.N

        return loaders

    def get_adj(self, dataset_name: str) -> torch.Tensor:
        """Convenience: normalised adj for any dataset."""
        return self._load(dataset_name, 'source').adj