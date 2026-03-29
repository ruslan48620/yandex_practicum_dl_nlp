import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import AdamW

import timm

from transformers import AutoModel, AutoTokenizer

from scripts.dataset import create_dataloaders


@dataclass
class TrainingConfig:
    RUN_NAME: str = 'baseline_multimodal_regression'
    
    TEXT_MODEL_NAME: str = 'distilbert-base-uncased'
    IMAGE_MODEL_NAME: str = 'resnet18'

    TEXT_MODEL_UNFREEZE: str = ''
    IMAGE_MODEL_UNFREEZE: str = ''

    IMAGE_COLUMN: str = 'image_path'
    TEXT_COLUMN: str = 'ingredients_text'
    MASS_COLUMN: str = 'total_mass'
    INGREDIENT_COUNT_COLUMN: str = 'ingredient_count'
    TARGET_COLUMN: str = 'total_calories'

    USE_MASS_FEATURE: bool = True
    USE_INGREDIENT_COUNT_FEATURE: bool = True

    DEVICE: str = 'cpu'
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False
    
    SEED: int = 42

    EPOCHS: int = 5
    BATCH_SIZE: int = 16
    MAX_TEXT_LENGTH: int = 128

    HIDDEN_DIM: int = 256
    DROPOUT: float = 0.2
    TEXT_LR: float = 1e-5
    IMAGE_LR: float = 1e-5
    HEAD_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    
    USE_SCHEDULER: bool = True
    SCHEDULER_NAME: str = 'ReduceLROnPlateau'
    SCHEDULER_FACTOR: float = 0.5
    SCHEDULER_PATIENCE: int = 2
    SCHEDULER_MIN_LR: float = 1e-6
    
    LOSS_NAME: str = 'smooth_l1'

    DATA_PATH: str = 'data'
    OUTPUT_PATH: str = 'output'
    MODELS_PATH: str = 'models'
    SAVE_PATH: str = ''
    HISTORY_PATH: str = ''
    CONFIG_SNAPSHOT_PATH: str = ''


def resolve_config(config_source):
    if isinstance(config_source, TrainingConfig):
        config = config_source
    elif isinstance(config_source, (str, Path)):
        config_path = str(config_source)
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config = TrainingConfig(**config_data)
    elif isinstance(config_source, dict):
        config = TrainingConfig(**config_source)
    else:
        raise TypeError('config_source must be TrainingConfig, dict, str or Path')

    return finalize_config_paths(config)


def finalize_config_paths(config):
    run_dir = Path(config.MODELS_PATH) / config.RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    if not config.SAVE_PATH:
        config.SAVE_PATH = str(run_dir / 'best_model.pt')
    if not config.HISTORY_PATH:
        config.HISTORY_PATH = str(run_dir / 'train_history.json')
    if not config.CONFIG_SNAPSHOT_PATH:
        config.CONFIG_SNAPSHOT_PATH = str(run_dir / 'train_config.json')

    return config


def save_config(config_source, config_path=''):
    config = resolve_config(config_source)
    if config_path:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.CONFIG_SNAPSHOT_PATH = str(config_path)

    with open(config.CONFIG_SNAPSHOT_PATH, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    return config_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_requires_grad(module, unfreeze_pattern='', verbose=False):
    if not unfreeze_pattern:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    if unfreeze_pattern == '*':
        for name, param in module.named_parameters():
            param.requires_grad = True
            if verbose:
                print(f'Unfrozen layer: {name}')
        return

    allowed_prefixes = unfreeze_pattern.split('|')
    for name, param in module.named_parameters():
        if any([name.startswith(prefix) for prefix in allowed_prefixes]):
            param.requires_grad = True
            if verbose:
                print(f'Unfrozen layer: {name}')
        else:
            param.requires_grad = False



class MultimodalCaloriesRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.num_features, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
        )

        numeric_features_dim = int(config.USE_MASS_FEATURE) + int(config.USE_INGREDIENT_COUNT_FEATURE)
        self.numeric_proj = None
        projected_numeric_dim = 0
        if numeric_features_dim > 0:
            projected_numeric_dim = config.HIDDEN_DIM // 2
            self.numeric_proj = nn.Sequential(
                nn.Linear(numeric_features_dim, projected_numeric_dim),
                nn.LayerNorm(projected_numeric_dim),
                nn.ReLU(),
            )

        fusion_dim = (config.HIDDEN_DIM * 2) + projected_numeric_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass, ingredient_count):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :]

        image_features = self.image_model(image)

        fused_parts = [
            self.text_proj(text_features),
            self.image_proj(image_features),
        ]

        numeric_parts = []
        if self.config.USE_MASS_FEATURE:
            numeric_parts.append(mass)
        if self.config.USE_INGREDIENT_COUNT_FEATURE:
            numeric_parts.append(ingredient_count)

        if self.numeric_proj is not None and numeric_parts:
            numeric_features = torch.cat(numeric_parts, dim=1)
            fused_parts.append(self.numeric_proj(numeric_features))

        fused_features = torch.cat(fused_parts, dim=1)
        predictions = self.regressor(fused_features).squeeze(1)
        return predictions


def build_optimizer(model, config):
    head_modules = [model.text_proj, model.image_proj, model.regressor]
    if model.numeric_proj is not None:
        head_modules.append(model.numeric_proj)

    optimizer_groups = []

    text_params = [param for param in model.text_model.parameters() if param.requires_grad]
    if text_params:
        optimizer_groups.append({'params': text_params, 'lr': config.TEXT_LR})

    image_params = [param for param in model.image_model.parameters() if param.requires_grad]
    if image_params:
        optimizer_groups.append({'params': image_params, 'lr': config.IMAGE_LR})

    head_params = []
    for module in head_modules:
        head_params.extend(param for param in module.parameters() if param.requires_grad)
    if head_params:
        optimizer_groups.append({'params': head_params, 'lr': config.HEAD_LR})

    return AdamW(optimizer_groups, weight_decay=config.WEIGHT_DECAY)


def build_scheduler(optimizer, config):
    if not config.USE_SCHEDULER:
        return None

    if config.SCHEDULER_NAME == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
        )

    raise ValueError(f'Unsupported scheduler: {config.SCHEDULER_NAME}')


def get_loss_function(config):
    if config.LOSS_NAME == 'l1':
        return nn.L1Loss()
    if config.LOSS_NAME == 'smooth_l1':
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f'Unsupported loss function: {config.LOSS_NAME}')


def move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def compute_regression_metrics(predictions, targets):
    mae = torch.mean(torch.abs(predictions - targets)).item()
    return {'mae': mae}


def run_epoch(model, data_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        targets = batch['target']

        optimizer.zero_grad()

        predictions = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            image=batch['image'],
            mass=batch['mass'],
            ingredient_count=batch['ingredient_count'],
        )
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())

    predictions_tensor = torch.cat(all_predictions)
    targets_tensor = torch.cat(all_targets)
    metrics = compute_regression_metrics(predictions_tensor, targets_tensor)
    metrics['loss'] = total_loss / len(data_loader)
    return metrics


def validate(model, data_loader, criterion, device, return_predictions_df=False):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []
    prediction_rows = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            targets = batch['target']

            predictions = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image=batch['image'],
                mass=batch['mass'],
                ingredient_count=batch['ingredient_count'],
            )
            loss = criterion(predictions, targets)

            detached_predictions = predictions.detach().cpu()
            detached_targets = targets.detach().cpu()

            total_loss += loss.item()
            all_predictions.append(detached_predictions)
            all_targets.append(detached_targets)

            if return_predictions_df:
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            'dish_id': batch['dish_id'],
                            'target': detached_targets.numpy(),
                            'prediction': detached_predictions.numpy(),
                        }
                    )
                )

    predictions_tensor = torch.cat(all_predictions)
    targets_tensor = torch.cat(all_targets)
    metrics = compute_regression_metrics(predictions_tensor, targets_tensor)
    metrics['loss'] = total_loss / len(data_loader)

    if return_predictions_df:
        predictions_df = pd.concat(prediction_rows, ignore_index=True)
        predictions_df['absolute_error'] = (
            predictions_df['target'] - predictions_df['prediction']
        ).abs()
        return metrics, predictions_df
    
    return metrics


def evaluate_best_model(config_source, df, device=None):
    config = resolve_config(config_source)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    _, test_loader = create_dataloaders(config, df, tokenizer)

    checkpoint = torch.load(config.SAVE_PATH, map_location=device)
    model = MultimodalCaloriesRegressor(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = get_loss_function(config)
    test_metrics, test_predictions_df = validate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        return_predictions_df=True,
    )

    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")

    return {
        'test_loss': test_metrics['loss'],
        'test_mae': test_metrics['mae'],
        'predictions_df': test_predictions_df,
    }


def train(config_source, df, device, verbose=True):
    config = resolve_config(config_source)
    save_config(config, config.CONFIG_SNAPSHOT_PATH)
    seed_everything(config.SEED)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    train_loader, validation_loader = create_dataloaders(config, df, tokenizer)

    model = MultimodalCaloriesRegressor(config).to(device)
    set_requires_grad(model.text_model, config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model, config.IMAGE_MODEL_UNFREEZE, verbose=True)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = get_loss_function(config)

    history = []
    best_validation_mae = float('inf')

    print(f'Training started on device: {device}')
    print(f'Config snapshot saved to: {config.CONFIG_SNAPSHOT_PATH}')
    print(f'Best model will be saved to: {config.SAVE_PATH}')

    for epoch in tqdm(range(1, config.EPOCHS + 1), disable=not verbose):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        validation_metrics = validate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(validation_metrics['mae'])
        
        current_lrs = [group['lr'] for group in optimizer.param_groups]

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_mae': train_metrics['mae'],
            'val_loss': validation_metrics['loss'],
            'val_mae': validation_metrics['mae'],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_mae={train_metrics['mae']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_mae={validation_metrics['mae']:.4f} | "
            f"current_lrs={current_lrs}"
        )

        if validation_metrics['mae'] < best_validation_mae:
            best_validation_mae = validation_metrics['mae']
            checkpoint = {
                'epoch': epoch,
                'best_val_mae': best_validation_mae,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
            }
            torch.save(checkpoint, config.SAVE_PATH)
            print(f'Saved new best checkpoint with val_mae={best_validation_mae:.4f}')

    history_path = Path(config.HISTORY_PATH)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open('w', encoding='utf-8') as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    return {
        'config': asdict(config),
        'history': history,
        'best_val_mae': best_validation_mae,
        'best_checkpoint_path': config.SAVE_PATH,
        'history_path': config.HISTORY_PATH,
    }


def plot_training_history(history):
    history_df = pd.DataFrame(history)
    epochs = history_df['epoch']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_df['train_loss'], marker='o', label='train_loss')
    axes[0].plot(epochs, history_df['val_loss'], marker='o', label='val_loss')
    axes[0].set_title('Loss by Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history_df['train_mae'], marker='o', label='train_mae')
    axes[1].plot(epochs, history_df['val_mae'], marker='o', label='val_mae')
    axes[1].set_title('MAE by Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return fig, axes
