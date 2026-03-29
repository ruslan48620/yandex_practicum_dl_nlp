from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DishCaloriesDataset(Dataset):
    def __init__(self, df, config, transforms, return_target=True):
        self.df = df.reset_index(drop=True).copy()
        self.config = config
        self.transforms = transforms
        self.return_target = return_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row[self.config.IMAGE_COLUMN])

        if not image_path.exists():
            raise FileNotFoundError(f'Image file was not found: {image_path}')

        image = np.array(Image.open(image_path).convert("RGB"))
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

        sample = {
            'dish_id': row['dish_id'],
            'image': image,
            'image_path': str(image_path),
            'text': row[self.config.TEXT_COLUMN],
            'mass': row[self.config.MASS_COLUMN],
            'ingredient_count': row[self.config.INGREDIENT_COUNT_COLUMN],
        }

        if self.return_target and self.config.TARGET_COLUMN in row.index:
            sample['target'] = row[self.config.TARGET_COLUMN]

        return sample


def collate_dish_batch(batch, tokenizer, max_length):
    images = torch.stack([item['image'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    texts = [item['text'] for item in batch]
    masses = torch.tensor(
        [item['mass'] for item in batch], dtype=torch.float32
    ).unsqueeze(1)
    ingredient_counts = torch.tensor(
        [item['ingredient_count'] for item in batch], dtype=torch.float32
    ).unsqueeze(1)

    targets = torch.tensor(
        [item['target'] for item in batch], dtype=torch.float32
    )

    tokenized_input = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    collated_batch = {
        'dish_id': [item['dish_id'] for item in batch],
        'image': images,
        'image_path': image_paths,
        'text': texts,
        'mass': masses,
        'ingredient_count': ingredient_counts,
        
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],

        'target': targets
    }

    return collated_batch


def get_image_preprocessing_cfg(config):
    pretrained_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    image_size = pretrained_cfg.input_size[1]
    image_mean = tuple(pretrained_cfg.mean)
    image_std = tuple(pretrained_cfg.std)

    return image_size, image_mean, image_std


def get_transforms(config, ds_type='train'):
    image_size, image_mean, image_std = get_image_preprocessing_cfg(config)
    seed = config.SEED

    if ds_type == 'train':
        augmented_size = int(image_size * 1.15)
        transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=augmented_size),
                A.PadIfNeeded(
                    min_height=augmented_size,
                    min_width=augmented_size,
                    border_mode=0,
                    fill=0,
                ),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.92, 1.08),
                    translate_percent=(-0.06, 0.06),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    border_mode=0,
                    fill=0,
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.20,
                            contrast_limit=0.20,
                            p=1.0,
                        ),
                        A.ColorJitter(
                            brightness=0.20,
                            contrast=0.20,
                            saturation=0.20,
                            hue=0.10,
                            p=1.0,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=15,
                            val_shift_limit=10,
                            p=1.0,
                        ),
                    ],
                    p=0.6,
                ),
                A.Normalize(mean=image_mean, std=image_std),
                ToTensorV2(p=1.0),
            ],
            seed=seed,
        )
    else:
        transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=0,
                    fill=0,
                ),
                A.Normalize(mean=image_mean, std=image_std),
                ToTensorV2(p=1.0),
            ],
            seed=seed,
        )

    return transforms


def create_datasets(df, config):
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    train_dataset = DishCaloriesDataset(
        df=train_df,
        config=config,
        transforms=get_transforms(config=config, ds_type='train'),
        return_target=True,
    )
    test_dataset = DishCaloriesDataset(
        df=test_df,
        config=config,
        transforms=get_transforms(config=config, ds_type='test'),
        return_target=True,
    )
    return train_dataset, test_dataset


def create_dataloaders(config, df, tokenizer):
    train_dataset, test_dataset = create_datasets(df, config)

    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    max_text_length = config.MAX_TEXT_LENGTH
    pin_memory = config.PIN_MEMORY

    collate_fn = partial(
        collate_dish_batch,
        tokenizer=tokenizer,
        max_length=max_text_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader
