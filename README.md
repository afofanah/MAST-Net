# MAST-Net

**Manifold-Aware Spatio-Temporal Network** for traffic forecasting.

MAST-Net predicts multi-step traffic flow across sensor networks by combining asymmetric graph diffusion, frequency-domain temporal mixing, and manifold-preserving data augmentation. It works with any number of input features and any prediction horizon.

---

## Architecture

```
Input (B, N, T, F)
  └─ MPLAM          manifold-preserving augmentation + adaptive adj/coords
  └─ MFPE           multi-scale frequency positional embedding (N, T, d_h)
  └─ in_proj        fuse raw features + positional embedding → (B, N, T, d_h)
  └─ SPM            spatial processing: ADGD (asymmetric directed graph diffusion)
                      └─ AdaptiveAdjacency   asymmetric src×tgt learned adj
                      └─ MultiScaleGCN       k-hop fwd/bwd diffusion, k∈{1,3,5,7}
                      └─ HierarchicalPooling node pooling at layers 1,3
  └─ TPM            temporal processing: LipschitzMHA + FDTA per layer
                      └─ LipschitzMHA        Lipschitz-constrained multi-head attention
                      └─ FrequencyMixing     learnable spectral filter via RFFT
  └─ FusionModule   weighted fusion of spatial, temporal, input streams
  └─ DualPathDecoder
        ├─ Spectral path   Chebyshev GCN + temporal attention pooling
        ├─ Temporal path   future TE queries cross-attend encoder context
        └─ IPR             iterative prediction refinement (cross-attention correction)
Output (B, N, H, F)
```

**Loss:**
```
J = L_pred + λ₁·L_spatial + λ₂·L_temporal + λ₃·‖Θ_aug‖² + λ₄·d_M
```
where `d_M` is the manifold preservation distance from MPLAM.

---

## Project structure

```
├── models/
│   └── model.py         MASTNet, MPLAM, MFPE, SPM, TPM, DualPathDecoder, IPR
├── config.py            per-dataset hyperparameter registry
├── datasets.py          data loading, normalisation, TrafficDataset, DataManager
├── train.py             Trainer, Finetuner, make_optimizer, make_scheduler
├── utils.py             metrics, checkpointing, seed, formatting
├── plots.py             all visualisation utilities
└── main.py              CLI entry point
```

---

## Installation

```bash
pip install torch numpy scipy matplotlib
```

Python ≥ 3.9 and PyTorch ≥ 2.0 recommended. No other dependencies required.

---

## Supported datasets

| Dataset   | Nodes | Interval | Description |
|-----------|------:|------:|-------------|
| METR-LA   | 207   | 5 min | LA highway sensors, Mar–Jun 2012 |
| PEMS-BAY  | 325   | 5 min | SF Bay Area detectors, Jan–May 2017 |
| EXPY-TKY  | 1843  | 10 min | Tokyo expressway, Oct–Dec 2021 |
| ATP-CN    | 1748  | 5 min | Chinese ring-road detectors, Sep 2023 |
| ATP-PALL  | 866   | 5 min | ATP-CN directional subset (PALL) |
| ATP-TALL  | 882   | 5 min | ATP-CN directional subset (TALL) |

Each dataset is auto-configured via `config.py`. Any custom dataset in `.npz`, `.npy`, `.h5`, `.pkl`, or `.csv` format is supported without config changes.

---

## Data format

Place files in a single directory. The loader searches for common filenames automatically:

```
data/METR-LA/
├── data.npz          # traffic array  (T, N, F)  or  (N, F, T)
└── adj.npy           # adjacency      (N, N)  optional — identity used if absent
```

A `meta.npy` file of shape `(T, d_m)` can provide external temporal features (weather, events). If absent, temporal features are generated from timestamps.

---

## Usage

### Training

```bash
python main.py \
  --data_dir  ./data/METR-LA \
  --dataset   METR-LA \
  --device    cuda
```

All hyperparameters are loaded from `config.py` when `--dataset` is set. Override any of them on the CLI:

```bash
python main.py \
  --data_dir   ./data/METR-LA \
  --dataset    METR-LA \
  --lr         5e-4 \
  --max_epochs 150 \
  --batch_size 32 \
  --d_h        128 \
  --n_heads    8 \
  --lambda4    0.2
```

### Test-only

```bash
python main.py \
  --data_dir   ./data/METR-LA \
  --dataset    METR-LA \
  --test_only \
  --checkpoint experiments/exp_20240101_120000/checkpoints/best.pth
```

### Fine-tuning

Fine-tune a pre-trained model on a new target domain. Only the listed top-level modules are unfrozen; the backbone is frozen.

```bash
# Pre-train then fine-tune on the same data
python main.py \
  --data_dir  ./data/METR-LA \
  --dataset   METR-LA \
  --finetune

# Load existing checkpoint, fine-tune on a different target
python main.py \
  --data_dir              ./data/METR-LA \
  --dataset               METR-LA \
  --finetune \
  --finetune_checkpoint   experiments/exp_xxx/checkpoints/best.pth \
  --finetune_data_dir     ./data/PEMS-BAY \
  --finetune_dataset      PEMS-BAY \
  --finetune_layers       decoder fusion temporal \
  --finetune_lr           5e-5 \
  --finetune_epochs       30
```

### List datasets

```bash
python main.py --list_datasets
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | path to dataset directory |
| `--dataset` | `None` | dataset name for config lookup |
| `--seq_len` | 12 | input sequence length |
| `--pred_len` | 12 | prediction horizon |
| `--d_m` | 16 | meta-feature dimension |
| `--d_h` | from config | hidden dimension |
| `--n_heads` | 8 | attention heads |
| `--n_spatial` | 4 | spatial GCN layers |
| `--n_temporal` | 4 | transformer layers |
| `--dropout` | 0.1 | dropout rate |
| `--lambda1` | 0.01 | spatial regularisation weight |
| `--lambda2` | 0.10 | Lipschitz temporal weight |
| `--lambda3` | 0.01 | augmentation reg weight |
| `--lambda4` | 0.10 | manifold preservation weight |
| `--optimizer` | `adam` | `adam` / `adamw` / `sgd` |
| `--scheduler` | `cosine` | `cosine` / `step` / `plateau` / `none` |
| `--lr` | from config | learning rate |
| `--weight_decay` | from config | L2 weight decay |
| `--max_epochs` | from config | maximum training epochs |
| `--patience` | 15 | early-stopping patience |
| `--clip_grad` | 1.0 | gradient clip norm |
| `--batch_size` | from config | batch size |
| `--normalize` | `zscore` | `zscore` / `minmax` / `none` |
| `--seed` | 42 | random seed |
| `--device` | auto | `cuda` or `cpu` |
| `--exp_dir` | `experiments` | base directory for outputs |
| `--finetune` | `False` | enable fine-tune stage |
| `--finetune_checkpoint` | `None` | checkpoint to load before fine-tuning |
| `--finetune_data_dir` | `--data_dir` | target data directory for fine-tuning |
| `--finetune_dataset` | `--dataset` | target dataset name for fine-tuning |
| `--finetune_layers` | from config | module names to unfreeze |
| `--finetune_lr` | from config | fine-tune learning rate |
| `--finetune_epochs` | from config | fine-tune epoch budget |
| `--finetune_patience` | 10 | fine-tune early-stopping patience |

---

## Outputs

Each run creates a timestamped directory under `--exp_dir`:

```
experiments/exp_20240101_120000/
├── config.json               all resolved hyperparameters
├── checkpoints/
│   ├── best.pth              best validation checkpoint
│   └── finetune_best.pth     best fine-tune checkpoint (if --finetune)
├── test_results.npz          predictions + ground_truth arrays  (B, N, H, F)
├── finetune_test_results.npz fine-tune test arrays (if --finetune)
├── pretrain_summary.json     overall + per-horizon metrics
└── finetune_summary.json     fine-tune metrics (if --finetune)
```

### Loading results

```python
import numpy as np
from utils import per_horizon_metrics

data  = np.load('experiments/exp_xxx/test_results.npz')
preds = data['predictions']   # (B, N, H, F)
gts   = data['ground_truth']  # (B, N, H, F)

hm = per_horizon_metrics(preds, gts, steps=(3, 6, 12))
for step, m in hm.items():
    print(f'step {step:2d}  MAE={m["MAE"]:.4f}  RMSE={m["RMSE"]:.4f}  MAPE={m["MAPE"]:.2f}%')
```

## Python API

```python
from datasets import prepare_traffic_dataset
from models.model import MASTNet
from train import Trainer, Finetuner, make_optimizer, make_scheduler
from utils import set_seed, create_exp_dir, print_model_summary
from torch.utils.data import DataLoader

set_seed(42)

train_ds, val_ds, test_ds, adj, meta = prepare_traffic_dataset(
    data_dir='./data/METR-LA', seq_len=12, pred_len=12)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

model = MASTNet(
    num_nodes=meta['n_nodes'], input_dim=meta['n_features'],
    d_h=64, horizon=12, seq_len=12, adj=adj).to('cuda')
print_model_summary(model)

optimizer = make_optimizer(model, {'optimizer': 'adam', 'lr': 1e-3})
scheduler = make_scheduler(optimizer, {'scheduler': 'cosine', 'max_epochs': 100})
exp_dir   = create_exp_dir('experiments')

trainer = Trainer(model, train_loader, val_loader, test_loader,
                  optimizer, scheduler, device='cuda', exp_dir=exp_dir)
metrics, history = trainer.train()
print(metrics)   # {'MAE': ..., 'RMSE': ..., 'MAPE': ...}
```

---

## Evaluation

### Datasets

MAST-Net is evaluated on four traffic datasets, including a newly collected large-scale dataset from a Chinese ring road. All datasets use a **6:2:2 train/val/test split** and data is normalised to **[-1, 1]** via min-max scaling (`--normalize minmax`).

### Metrics

Performance is measured by three standard metrics over all L samples:

$$\text{MAE} = \frac{1}{L}\sum_{i=1}^{L}\left\| X^i - \hat{X}^i \right\|$$

$$\text{RMSE} = \sqrt{\frac{1}{L}\sum_{i=1}^{L}\left( X^i - \hat{X}^i \right)^2}$$

$$\text{MAPE} = \frac{1}{L}\sum_{i=1}^{L}\frac{\left\| X^i - \hat{X}^i \right\|}{X^i}$$

where $X^i$ is the ground truth for sample $i$, $\hat{X}^i$ the predicted value, and $L$ the total number of samples.

All three metrics are computed with `null_val=0.0` masking — timesteps where the ground truth is zero are excluded from the average. Evaluation uses feature index 0, matching standard traffic-forecasting conventions.

> **Note:** the 6:2:2 split differs from the METR-LA/PEMS-BAY convention (7:1:2). Pass `--normalize minmax` and ensure `train_ratio=0.6`, `val_ratio=0.2` in `config.py` (already set for ATP-CN, ATP-PALL, ATP-TALL, EXPY-TKY) or via `--dataset` auto-resolution.

---

## Adding a new dataset

1. Place data files in a directory.
2. Optionally add an entry to `DATASET_CONFIGS` in `config.py`.
3. Run with `--data_dir ./data/MyDataset` (no `--dataset` required).

The loader auto-detects `.npz / .npy / .h5 / .pkl / .csv` and infers `(T, N, F)` shape. If an adjacency file is absent, an identity matrix is used.
