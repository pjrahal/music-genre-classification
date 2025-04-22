import os
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from cachedGenreDataset import CachedGenreDataset
from genreCNN import GenreCNN
from settings import logger, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LEARNING_RATE, FEATURES_DIR, DATASET_DIR, METADATA_DIR

@dataclass
class TrainingAssets:
    df_meta: pd.DataFrame
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    device: torch.device
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    best_val_loss: float
    early_stop_counter: int
    num_epochs: int
    patience: int

def download_datasets():
    if not os.path.exists(DATASET_DIR):
        logger.info("Downloading fma_small dataset...")
        subprocess.run("wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip", shell=True, check=True)
        subprocess.run("unzip -o fma_small.zip -d ./", shell=True, check=True)

    if not os.path.exists(METADATA_DIR):
        logger.info("Downloading fma_metadata...")
        subprocess.run("wget -c https://os.unil.cloud.switch.ch/fma/fma_metadata.zip", shell=True, check=True)
        subprocess.run("unzip -o fma_metadata.zip -d ./", shell=True, check=True)

    if not os.path.exists(FEATURES_DIR):
        logger.info("Features not found. Running preprocessing...")
        subprocess.run("python3.11 audioPreprocessing.py", shell=True, check=True)

def prepare_dataloaders(df_meta):
    train_df, test_df = train_test_split(df_meta, test_size=0.2, stratify=df_meta.label, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df.label, random_state=42)

    train_loader = DataLoader(CachedGenreDataset(train_df, FEATURES_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(CachedGenreDataset(val_df, FEATURES_DIR), batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(CachedGenreDataset(test_df, FEATURES_DIR), batch_size=BATCH_SIZE, num_workers=2)

    return train_loader, val_loader, test_loader

def build_training_assets(df_meta, device, train_loader, val_loader, test_loader):
    num_classes = df_meta.label.nunique()
    class_counts = np.bincount(df_meta['label'])
    class_weights = 1. / class_counts
    class_weights /= class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = GenreCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    return TrainingAssets(
        df_meta=df_meta,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val_loss=float('inf'),
        early_stop_counter=0,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )

def getData():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_datasets()
    df_meta = pd.read_csv(os.path.join(FEATURES_DIR, "features_metadata.csv"))
    train_loader, val_loader, test_loader = prepare_dataloaders(df_meta)
    return build_training_assets(df_meta, device, train_loader, val_loader, test_loader)