import argparse, os, json
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import get_dataset_config, DATASET_CONFIGS
from datasets import prepare_traffic_dataset
from models.model import MASTNet
from train import Trainer, Finetuner, make_optimizer, make_scheduler
from utils import set_seed, create_exp_dir, print_model_summary, per_horizon_metrics


def parse_args():
    p = argparse.ArgumentParser(description='MAST-Net: Manifold-Constrained Adaptive Spatio-Temporal Network')

    p.add_argument('--data_dir',     type=str,   required=True)
    p.add_argument('--dataset',      type=str,   default=None)
    p.add_argument('--seq_len',      type=int,   default=12)
    p.add_argument('--pred_len',     type=int,   default=12)
    p.add_argument('--interval',     type=int,   default=None)
    p.add_argument('--normalize',    type=str,   default='zscore',
                   choices=['zscore', 'minmax', 'none'])
    p.add_argument('--d_m',          type=int,   default=16)

    p.add_argument('--d_h',          type=int,   default=None)
    p.add_argument('--n_heads',      type=int,   default=8)
    p.add_argument('--n_spatial',    type=int,   default=4)
    p.add_argument('--n_temporal',   type=int,   default=4)
    p.add_argument('--dropout',      type=float, default=0.1)

    p.add_argument('--lambda1',      type=float, default=0.01)
    p.add_argument('--lambda2',      type=float, default=0.1)
    p.add_argument('--lambda3',      type=float, default=0.01)
    p.add_argument('--lambda4',      type=float, default=0.1)

    p.add_argument('--optimizer',    type=str,   default='adam',
                   choices=['adam', 'adamw', 'sgd'])
    p.add_argument('--scheduler',    type=str,   default='cosine',
                   choices=['cosine', 'step', 'plateau', 'none'])
    p.add_argument('--lr',           type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--max_epochs',   type=int,   default=None)
    p.add_argument('--patience',     type=int,   default=15)
    p.add_argument('--clip_grad',    type=float, default=1.0)
    p.add_argument('--batch_size',   type=int,   default=None)
    p.add_argument('--num_workers',  type=int,   default=4)

    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--exp_dir',      type=str,   default='experiments')

    p.add_argument('--test_only',    action='store_true')
    p.add_argument('--checkpoint',   type=str,   default=None)
    p.add_argument('--list_datasets',action='store_true')

    # ── fine-tune args ────────────────────────────────────────────────────────
    p.add_argument('--finetune',            action='store_true',
                   help='Run fine-tune stage after (or instead of) pre-training')
    p.add_argument('--finetune_checkpoint', type=str,   default=None,
                   help='Checkpoint to load before fine-tuning (skips pre-training)')
    p.add_argument('--finetune_data_dir',   type=str,   default=None,
                   help='Target data dir for fine-tuning (defaults to --data_dir)')
    p.add_argument('--finetune_dataset',    type=str,   default=None,
                   help='Target dataset name for fine-tuning (defaults to --dataset)')
    p.add_argument('--finetune_layers',     type=str,   nargs='+',
                   default=None,
                   help='Top-level module names to unfreeze, e.g. decoder fusion')
    p.add_argument('--finetune_lr',         type=float, default=None)
    p.add_argument('--finetune_epochs',     type=int,   default=None)
    p.add_argument('--finetune_patience',   type=int,   default=10)

    return p.parse_args()


def resolve_args(args):
    if args.dataset:
        cfg = get_dataset_config(args.dataset)
        if args.d_h          is None: args.d_h          = cfg['d_h']
        if args.batch_size   is None: args.batch_size   = cfg['batch_size']
        if args.lr           is None: args.lr            = cfg['lr']
        if args.weight_decay is None: args.weight_decay  = cfg['weight_decay']
        if args.max_epochs   is None: args.max_epochs    = cfg['max_epochs']
        if args.interval     is None: args.interval      = cfg['interval_minutes']
        args._train_ratio = cfg['train_ratio']
        args._val_ratio   = cfg['val_ratio']

        # Fine-tune defaults from dataset config
        if args.finetune_lr     is None: args.finetune_lr     = cfg.get('finetune_lr', 1e-4)
        if args.finetune_epochs is None: args.finetune_epochs = cfg.get('finetune_epochs', 20)
        if args.finetune_layers is None: args.finetune_layers = cfg.get('finetune_layers',
                                                                         ['decoder', 'fusion'])
    else:
        args._train_ratio = 0.6
        args._val_ratio   = 0.2

    if args.d_h          is None: args.d_h          = 64
    if args.batch_size   is None: args.batch_size   = 64
    if args.lr           is None: args.lr            = 1e-3
    if args.weight_decay is None: args.weight_decay  = 1e-4
    if args.max_epochs   is None: args.max_epochs    = 100
    if args.interval     is None: args.interval      = 5
    if args.finetune_lr     is None: args.finetune_lr     = 1e-4
    if args.finetune_epochs is None: args.finetune_epochs = 20
    if args.finetune_layers is None: args.finetune_layers = ['decoder', 'fusion']
    return args


def build_loaders(args, data_dir=None, dataset=None):
    data_dir = data_dir or args.data_dir
    dataset  = dataset  or args.dataset
    train_ds, val_ds, test_ds, adj, metadata = prepare_traffic_dataset(
        data_dir         = data_dir,
        dataset_name     = dataset,
        seq_len          = args.seq_len,
        pred_len         = args.pred_len,
        train_ratio      = args._train_ratio,
        val_ratio        = args._val_ratio,
        normalize_method = args.normalize,
        interval_minutes = args.interval,
        d_m              = args.d_m,
    )
    pin = (args.device == 'cuda')
    kw  = dict(num_workers=args.num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **kw)
    return train_loader, val_loader, test_loader, adj, metadata


def build_model(args, n_nodes, n_features, adj) -> MASTNet:
    return MASTNet(
        num_nodes           = n_nodes,
        input_dim           = n_features,
        d_h                 = args.d_h,
        d_m                 = args.d_m,
        n_heads             = args.n_heads,
        num_spatial_layers  = args.n_spatial,
        num_temporal_layers = args.n_temporal,
        horizon             = args.pred_len,
        seq_len             = args.seq_len,
        dropout             = args.dropout,
        adj                 = adj,
    ).to(args.device)


def save_config(args, exp_dir: str):
    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def print_header(args, n_nodes, n_features, train_ds, val_ds, test_ds):
    sep = '=' * 64
    print(sep)
    print('  MAST-Net — Manifold-Aware Spatio-Temporal Network')
    print(sep)
    print(f'  Dataset   : {args.dataset or os.path.basename(args.data_dir)}')
    print(f'  Nodes / Features  : {n_nodes} / {n_features}')
    print(f'  Seq → Pred: {args.seq_len} → {args.pred_len}  (d_m={args.d_m})')
    print(f'  Train / Val / Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}')
    print(f'  Device    : {args.device}')
    print(f'  d_h={args.d_h}  heads={args.n_heads}  '
          f'spatial={args.n_spatial}  temporal={args.n_temporal}')
    print(f'  lr={args.lr}  wd={args.weight_decay}  '
          f'bs={args.batch_size}  epochs={args.max_epochs}')
    print(f'  λ1={args.lambda1}  λ2={args.lambda2}  '
          f'λ3={args.lambda3}  λ4={args.lambda4}')
    print(sep)


def print_results(metrics: dict, horizon_metrics: dict = None, stage: str = 'TEST'):
    sep = '=' * 64
    print(sep)
    print(f'  {stage} RESULTS')
    print(sep)
    print(f'  MAE  : {metrics["MAE"]:.4f}')
    print(f'  RMSE : {metrics["RMSE"]:.4f}')
    print(f'  MAPE : {metrics["MAPE"]:.2f}%')
    if horizon_metrics:
        print()
        print('  Per-horizon:')
        for h, m in sorted(horizon_metrics.items()):
            print(f'    step {h:2d} | '
                  f'MAE {m["MAE"]:.4f}  RMSE {m["RMSE"]:.4f}  MAPE {m["MAPE"]:.2f}%')
    print(sep)


def main():
    args = parse_args()

    if args.list_datasets:
        print('\nSupported datasets:')
        for name, cfg in DATASET_CONFIGS.items():
            print(f'  {name:<12}  nodes={cfg["n_nodes"]:<5}  {cfg["description"]}')
        return

    args = resolve_args(args)
    set_seed(args.seed)

    print('\nLoading data ...')
    train_loader, val_loader, test_loader, adj, metadata = build_loaders(args)

    n_nodes    = metadata['n_nodes']
    n_features = metadata['n_features']
    train_ds   = train_loader.dataset
    val_ds     = val_loader.dataset
    test_ds    = test_loader.dataset

    print_header(args, n_nodes, n_features, train_ds, val_ds, test_ds)
    model = build_model(args, n_nodes, n_features, adj)
    print_model_summary(model)

    # ── test-only mode ────────────────────────────────────────────────────────
    if args.test_only:
        assert args.checkpoint, '--checkpoint is required with --test_only'
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded checkpoint: {args.checkpoint}')
        trainer = Trainer(
            model=model, train_loader=train_loader,
            val_loader=val_loader, test_loader=test_loader,
            optimizer=None, scheduler=None,
            device=args.device, exp_dir=args.exp_dir,
        )
        metrics, preds, gts = trainer.test()
        print_results(metrics, per_horizon_metrics(preds, gts))
        return

    exp_dir = create_exp_dir(args.exp_dir)
    save_config(args, exp_dir)
    print(f'Experiment dir: {exp_dir}')

    # ── pre-training (skipped if --finetune_checkpoint supplied) ─────────────
    if args.finetune and args.finetune_checkpoint:
        print(f'\nSkipping pre-training — loading checkpoint: {args.finetune_checkpoint}')
        ckpt = torch.load(args.finetune_checkpoint, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        pretrain_metrics = None
    else:
        optimizer = make_optimizer(model, {
            'optimizer':    args.optimizer,
            'lr':           args.lr,
            'weight_decay': args.weight_decay,
        })
        scheduler = make_scheduler(optimizer, {
            'scheduler':  args.scheduler,
            'max_epochs': args.max_epochs,
            'min_lr':     1e-6,
        })
        trainer = Trainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            optimizer    = optimizer,
            scheduler    = scheduler,
            device       = args.device,
            exp_dir      = exp_dir,
            max_epochs   = args.max_epochs,
            patience     = args.patience,
            clip_grad    = args.clip_grad,
            lambda1      = args.lambda1,
            lambda2      = args.lambda2,
            lambda3      = args.lambda3,
            lambda4      = args.lambda4,
        )
        pretrain_metrics, history = trainer.train()

        npz = np.load(os.path.join(exp_dir, 'test_results.npz'))
        hm  = per_horizon_metrics(npz['predictions'], npz['ground_truth'])
        print_results(pretrain_metrics, hm, stage='PRE-TRAIN TEST')

        with open(os.path.join(exp_dir, 'pretrain_summary.json'), 'w') as f:
            json.dump({
                'dataset':     args.dataset or args.data_dir,
                'metrics':     {k: float(v) for k, v in pretrain_metrics.items()},
                'per_horizon': {str(k): {m: float(v) for m, v in mv.items()}
                                for k, mv in hm.items()},
                'best_val':    float(trainer.best_val),
            }, f, indent=2)

    # ── fine-tune stage ───────────────────────────────────────────────────────
    if args.finetune:
        ft_data_dir = args.finetune_data_dir or args.data_dir
        ft_dataset  = args.finetune_dataset  or args.dataset

        if ft_data_dir != args.data_dir or ft_dataset != args.dataset:
            print(f'\nLoading fine-tune target data from {ft_data_dir} ...')
            ft_train, ft_val, ft_test, _, ft_meta = build_loaders(
                args, data_dir=ft_data_dir, dataset=ft_dataset)
            print(f'  Target nodes={ft_meta["n_nodes"]}  features={ft_meta["n_features"]}')
        else:
            ft_train, ft_val, ft_test = train_loader, val_loader, test_loader

        finetuner = Finetuner(
            model           = model,
            train_loader    = ft_train,
            val_loader      = ft_val,
            test_loader     = ft_test,
            device          = args.device,
            exp_dir         = exp_dir,
            finetune_layers = args.finetune_layers,
            lr              = args.finetune_lr,
            weight_decay    = args.weight_decay,
            max_epochs      = args.finetune_epochs,
            patience        = args.finetune_patience,
            clip_grad       = args.clip_grad,
            lambda1         = args.lambda1,
            lambda2         = args.lambda2,
            lambda3         = args.lambda3,
            lambda4         = args.lambda4,
        )
        ft_metrics, ft_history = finetuner.run()

        ft_npz = np.load(os.path.join(exp_dir, 'finetune_test_results.npz'))
        ft_hm  = per_horizon_metrics(ft_npz['predictions'], ft_npz['ground_truth'])
        print_results(ft_metrics, ft_hm, stage='FINE-TUNE TEST')

        with open(os.path.join(exp_dir, 'finetune_summary.json'), 'w') as f:
            json.dump({
                'dataset':        ft_dataset or ft_data_dir,
                'finetune_layers': args.finetune_layers,
                'metrics':        {k: float(v) for k, v in ft_metrics.items()},
                'per_horizon':    {str(k): {m: float(v) for m, v in mv.items()}
                                   for k, mv in ft_hm.items()},
                'best_val':       float(finetuner.best_val),
            }, f, indent=2)


if __name__ == '__main__':
    main()