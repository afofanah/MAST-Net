"""
config.py — per-dataset hyperparameter registry for MAST-Net
"""

DATASET_CONFIGS = {
    'METR-LA': {
        'n_nodes': 207, 'expected_timesteps': 34272, 'interval_minutes': 5,
        'train_ratio': 0.7, 'val_ratio': 0.1,
        'd_h': 64, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4, 'max_epochs': 100,
        'finetune_lr': 1e-4, 'finetune_epochs': 20, 'finetune_layers': ['decoder', 'fusion'],
        'description': 'Los Angeles highway sensors, 5-min, Mar-Jun 2012',
    },
    'PEMS-BAY': {
        'n_nodes': 325, 'expected_timesteps': 52116, 'interval_minutes': 5,
        'train_ratio': 0.7, 'val_ratio': 0.1,
        'd_h': 64, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4, 'max_epochs': 100,
        'finetune_lr': 1e-4, 'finetune_epochs': 20, 'finetune_layers': ['decoder', 'fusion'],
        'description': 'SF Bay Area detectors, 5-min, Jan-May 2017',
    },
    'EXPY-TKY': {
        'n_nodes': 1843, 'expected_timesteps': 13248, 'interval_minutes': 10,
        'train_ratio': 0.6, 'val_ratio': 0.2,
        'd_h': 128, 'batch_size': 32, 'lr': 5e-4, 'weight_decay': 1e-4, 'max_epochs': 150,
        'finetune_lr': 5e-5, 'finetune_epochs': 30, 'finetune_layers': ['decoder', 'fusion', 'temporal'],
        'description': 'Tokyo expressway detectors, 10-min, Oct-Dec 2021',
    },
    'ATP-CN': {
        'n_nodes': 1748, 'expected_timesteps': 8968, 'interval_minutes': 5,
        'train_ratio': 0.6, 'val_ratio': 0.2,
        'd_h': 128, 'batch_size': 32, 'lr': 5e-4, 'weight_decay': 1e-4, 'max_epochs': 150,
        'finetune_lr': 5e-5, 'finetune_epochs': 30, 'finetune_layers': ['decoder', 'fusion', 'temporal'],
        'description': 'Chinese ring road detectors, 5-min, Sep 2023',
    },
    'ATP-PALL': {
        'n_nodes': 866, 'expected_timesteps': 8968, 'interval_minutes': 5,
        'train_ratio': 0.6, 'val_ratio': 0.2,
        'd_h': 64, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4, 'max_epochs': 100,
        'finetune_lr': 1e-4, 'finetune_epochs': 20, 'finetune_layers': ['decoder', 'fusion'],
        'description': 'ATP-CN directional subset (PALL), 866 detectors',
    },
    'ATP-TALL': {
        'n_nodes': 882, 'expected_timesteps': 8968, 'interval_minutes': 5,
        'train_ratio': 0.6, 'val_ratio': 0.2,
        'd_h': 64, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4, 'max_epochs': 100,
        'finetune_lr': 1e-4, 'finetune_epochs': 20, 'finetune_layers': ['decoder', 'fusion'],
        'description': 'ATP-CN directional subset (TALL), 882 detectors',
    },
}

DEFAULTS = {
    'n_nodes': None, 'expected_timesteps': None, 'interval_minutes': 5,
    'train_ratio': 0.6, 'val_ratio': 0.2,
    'd_h': 64, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4, 'max_epochs': 100,
    'finetune_lr': 1e-4, 'finetune_epochs': 20, 'finetune_layers': ['decoder', 'fusion'],
    'description': 'Unknown dataset',
}


def get_dataset_config(name: str) -> dict:
    key = name.upper().replace('-', '').replace('_', '')
    for k, v in DATASET_CONFIGS.items():
        if k.upper().replace('-', '').replace('_', '') == key:
            return dict(v)
    print(f'WARNING: no config for "{name}", using defaults')
    return dict(DEFAULTS)