import os, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from utils import compute_metrics, save_checkpoint, format_time


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr  = float(cfg.get('lr', 1e-3))
    wd  = float(cfg.get('weight_decay', 1e-4))
    opt = cfg.get('optimizer', 'adam').lower()
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f'Unknown optimizer: {opt}')


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> Optional[object]:
    stype  = cfg.get('scheduler', 'cosine').lower()
    T_max  = int(cfg.get('max_epochs', 100))
    min_lr = float(cfg.get('min_lr', 1e-6))
    if stype == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
    if stype == 'step':
        return StepLR(optimizer, step_size=max(T_max // 3, 1), gamma=0.5)
    if stype == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=min_lr)
    if stype == 'none':
        return None
    raise ValueError(f'Unknown scheduler: {stype}')


class Trainer:
    """
    Single-dataset trainer for MAST-Net.
    train()  → (metrics, history)
    test()   → (metrics, preds, gts)  — arrays: (B, N, H, F) normalised space
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
        exp_dir:    str,
        max_epochs: int   = 100,
        patience:   int   = 15,
        clip_grad:  float = 1.0,
        lambda1:    float = 0.01,
        lambda2:    float = 0.10,
        lambda3:    float = 0.01,
        lambda4:    float = 0.10,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = torch.device(device)
        self.exp_dir      = exp_dir
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.clip_grad    = clip_grad
        self.lambda1      = lambda1
        self.lambda2      = lambda2
        self.lambda3      = lambda3
        self.lambda4      = lambda4

        self.best_val   = float('inf')
        self._no_improv = 0
        self._ckpt      = os.path.join(exp_dir, 'checkpoints', 'best.pth')

    def _move(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        sums, n = defaultdict(float), 0
        for x, y, meta in self.train_loader:
            x, y, meta = self._move(x, y, meta)
            self.optimizer.zero_grad()
            total, pred_l, sp_l, tmp_l, aug_l, d_M, _ = self.model.compute_losses(
                x, y, meta, self.lambda1, self.lambda2, self.lambda3, self.lambda4)
            if torch.isfinite(total):
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
            sums['total']   += total.item()
            sums['pred']    += pred_l.item()
            sums['spatial'] += sp_l.item()
            sums['temp']    += tmp_l.item()
            sums['aug']     += aug_l.item()
            sums['d_M']     += d_M.item()
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def _eval_epoch(self, loader) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_p, all_g = [], []
        for x, y, meta in loader:
            x, y, meta = self._move(x, y, meta)
            yp, _, _   = self.model(x, meta)
            all_p.append(yp.cpu())
            all_g.append(y.cpu())
        preds   = torch.cat(all_p, dim=0)
        gts     = torch.cat(all_g, dim=0)
        metrics = compute_metrics(preds[..., 0], gts[..., 0])
        return metrics, preds, gts

    def train(self) -> Tuple[dict, List[dict]]:
        history = []
        t0      = time.time()

        print(f'\n{"─"*64}')
        print(f'  Training  max_epochs={self.max_epochs}  patience={self.patience}')
        print(f'  λ1={self.lambda1}  λ2={self.lambda2}  λ3={self.lambda3}  λ4={self.lambda4}')
        print(f'{"─"*64}')

        for epoch in range(1, self.max_epochs + 1):
            t_ep        = time.time()
            tr          = self._train_epoch()
            val_m, _, _ = self._eval_epoch(self.val_loader)
            val_loss    = val_m['MAE']

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            record = {
                'epoch':    epoch,
                'val_MAE':  val_loss,
                'val_RMSE': val_m['RMSE'],
                'val_MAPE': val_m['MAPE'],
                **{f'train_{k}': v for k, v in tr.items()},
            }
            history.append(record)

            if val_loss < self.best_val:
                self.best_val   = val_loss
                self._no_improv = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, self._ckpt)
                flag = ' ✓'
            else:
                self._no_improv += 1
                flag = ''

            lr = self.optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:3d}/{self.max_epochs} '
                  f'| pred={tr["pred"]:.4f} d_M={tr["d_M"]:.4f} '
                  f'| val_MAE={val_loss:.4f} RMSE={val_m["RMSE"]:.4f} '
                  f'| lr={lr:.2e} [{time.time()-t_ep:.1f}s]{flag}')

            if self._no_improv >= self.patience:
                print(f'\n  Early stopping at epoch {epoch}')
                break

        print(f'\n  Done in {format_time(time.time()-t0)} | best_val={self.best_val:.4f}')

        if os.path.exists(self._ckpt):
            self.model.load_state_dict(
                torch.load(self._ckpt, map_location=self.device)['model'])

        test_metrics, preds, gts = self.test()
        np.savez(os.path.join(self.exp_dir, 'test_results.npz'),
                 predictions=preds, ground_truth=gts)
        return test_metrics, history

    def test(self) -> Tuple[dict, np.ndarray, np.ndarray]:
        metrics, preds, gts = self._eval_epoch(self.test_loader)
        return metrics, preds.numpy(), gts.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Finetuner — freeze backbone, train only specified top-level submodules
# ─────────────────────────────────────────────────────────────────────────────

class Finetuner:
    """
    Fine-tunes a pre-trained MASTNet on a (small) target dataset.

    Only the submodules listed in `finetune_layers` are unfrozen; all others
    are frozen for the duration of fine-tuning.  A cosine LR schedule is used
    by default.  Best checkpoint is saved as  <exp_dir>/checkpoints/finetune_best.pth.

    Usage:
        finetuner = Finetuner(
            model          = model,                  # already has pretrained weights
            train_loader   = target_train_loader,
            val_loader     = target_val_loader,
            test_loader    = target_test_loader,
            device         = 'cuda',
            exp_dir        = exp_dir,
            finetune_layers= ['decoder', 'fusion'],  # top-level attr names on model
            lr             = 1e-4,
            max_epochs     = 20,
            patience       = 10,
        )
        ft_metrics, ft_history = finetuner.run()
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device:          str,
        exp_dir:         str,
        finetune_layers: List[str] = None,
        lr:              float = 1e-4,
        weight_decay:    float = 1e-4,
        max_epochs:      int   = 20,
        patience:        int   = 10,
        clip_grad:       float = 1.0,
        lambda1:         float = 0.01,
        lambda2:         float = 0.10,
        lambda3:         float = 0.01,
        lambda4:         float = 0.10,
    ):
        self.model           = model
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.test_loader     = test_loader
        self.device          = torch.device(device)
        self.exp_dir         = exp_dir
        self.finetune_layers = finetune_layers or ['decoder', 'fusion']
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.max_epochs      = max_epochs
        self.patience        = patience
        self.clip_grad       = clip_grad
        self.lambda1         = lambda1
        self.lambda2         = lambda2
        self.lambda3         = lambda3
        self.lambda4         = lambda4

        self.best_val = float('inf')
        self._ckpt    = os.path.join(exp_dir, 'checkpoints', 'finetune_best.pth')

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        unfrozen = []
        for layer_name in self.finetune_layers:
            module = getattr(self.model, layer_name, None)
            if module is None:
                print(f'  WARNING: model has no attribute "{layer_name}", skipping')
                continue
            for param in module.parameters():
                param.requires_grad = True
            unfrozen.append(layer_name)
        if not unfrozen:
            raise ValueError(
                f'None of the requested finetune_layers exist on the model: '
                f'{self.finetune_layers}')
        print(f'  Frozen backbone. Trainable layers: {unfrozen}')
        return unfrozen

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _move(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def _train_epoch(self, optimizer) -> Dict[str, float]:
        self.model.train()
        sums, n = defaultdict(float), 0
        for x, y, meta in self.train_loader:
            x, y, meta = self._move(x, y, meta)
            optimizer.zero_grad()
            total, pred_l, sp_l, tmp_l, aug_l, d_M, _ = self.model.compute_losses(
                x, y, meta, self.lambda1, self.lambda2, self.lambda3, self.lambda4)
            if torch.isfinite(total):
                total.backward()
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, self.clip_grad)
                optimizer.step()
            sums['total']   += total.item()
            sums['pred']    += pred_l.item()
            sums['spatial'] += sp_l.item()
            sums['temp']    += tmp_l.item()
            sums['aug']     += aug_l.item()
            sums['d_M']     += d_M.item()
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def _eval_epoch(self, loader) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_p, all_g = [], []
        for x, y, meta in loader:
            x, y, meta = self._move(x, y, meta)
            yp, _, _   = self.model(x, meta)
            all_p.append(yp.cpu())
            all_g.append(y.cpu())
        preds   = torch.cat(all_p, dim=0)
        gts     = torch.cat(all_g, dim=0)
        metrics = compute_metrics(preds[..., 0], gts[..., 0])
        return metrics, preds, gts

    def run(self) -> Tuple[dict, List[dict]]:
        """
        Freeze backbone → fine-tune listed layers → restore best weights →
        evaluate on test set.  Returns (test_metrics, epoch_history).
        """
        self._freeze_backbone()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer  = torch.optim.Adam(trainable_params, lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler  = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        history    = []
        no_improv  = 0
        t0         = time.time()

        print(f'\n{"─"*64}')
        print(f'  Fine-tuning  max_epochs={self.max_epochs}  patience={self.patience}')
        print(f'  lr={self.lr}  layers={self.finetune_layers}')
        print(f'{"─"*64}')

        for epoch in range(1, self.max_epochs + 1):
            t_ep        = time.time()
            tr          = self._train_epoch(optimizer)
            val_m, _, _ = self._eval_epoch(self.val_loader)
            val_loss    = val_m['MAE']
            scheduler.step()

            record = {
                'epoch':    epoch,
                'val_MAE':  val_loss,
                'val_RMSE': val_m['RMSE'],
                'val_MAPE': val_m['MAPE'],
                **{f'train_{k}': v for k, v in tr.items()},
            }
            history.append(record)

            if val_loss < self.best_val:
                self.best_val = val_loss
                no_improv     = 0
                save_checkpoint(self.model, optimizer, epoch, val_loss, self._ckpt)
                flag = ' ✓'
            else:
                no_improv += 1
                flag = ''

            lr_now = optimizer.param_groups[0]['lr']
            print(f'  FT Epoch {epoch:3d}/{self.max_epochs} '
                  f'| pred={tr["pred"]:.4f} '
                  f'| val_MAE={val_loss:.4f} RMSE={val_m["RMSE"]:.4f} '
                  f'| lr={lr_now:.2e} [{time.time()-t_ep:.1f}s]{flag}')

            if no_improv >= self.patience:
                print(f'\n  Early stopping at epoch {epoch}')
                break

        print(f'\n  Fine-tune done in {format_time(time.time()-t0)} '
              f'| best_val={self.best_val:.4f}')

        if os.path.exists(self._ckpt):
            self.model.load_state_dict(
                torch.load(self._ckpt, map_location=self.device)['model'])

        self._unfreeze_all()

        metrics, preds, gts = self._eval_epoch(self.test_loader)
        np.savez(os.path.join(self.exp_dir, 'finetune_test_results.npz'),
                 predictions=preds, ground_truth=gts)
        return metrics, history