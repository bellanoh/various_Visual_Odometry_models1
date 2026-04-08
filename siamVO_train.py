# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from dataset import PressSequenceDataset
from pytorch_metric_learning import losses # Triplet Loss 위해 추가 (pip install pytorch-metric-learning)
from utils_siam2 import SeqTransform, save_checkpoint_full, denormalize_labels

class SiameseVODataset(torch.utils.data.Dataset):
    def __init__(self, pairs_file, labels_file, transform=None, label_stds=None, label_means=None):
        self.pairs = np.load(pairs_file)   # shape: [num_pairs, 2, H, W, C]
        self.labels = np.load(labels_file) # shape: [num_pairs, num_labels] ← delta_pose (예: 3 or 4 or 6)
        self.transform = transform
        self.label_stds = label_stds
        self.label_means = label_means

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]          # [2, H, W, C]

        anchor = pair[0]                # [H, W, C]
        positive = pair[1]              # [H, W, C]
        altitude = pair[2]
        yaw = pair[3]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            altitude = self.transform(altitude)
            yaw = self.transform(yaw)

        # delta_pose label (진짜 상대값)
        label = torch.tensor(self.labels[idx]).float()   # (num_labels,)

        # z-score 정규화 (선택사항이지만 추천)
        if self.label_means is not None and self.label_stds is not None:
            label = (label - self.label_means) / self.label_stds

        return anchor, positive, altitude, yaw, label   # ← negative 제거

# Siamese Model
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class siamVONet(nn.Module):
    def __init__(self):
        super(siamVONet, self).__init__()
        self.fe_frame_f = FeatureExtractor(1)
        self.fe_frame_f1 = FeatureExtractor(1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fe_mid = FeatureExtractor(130)
        self.fe_final = FeatureExtractor(64)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(1024, 512)
        self.relu_fc1 = nn.ReLU()
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(0.2)  # Assuming dropout rate of 0.2
        self.fc2 = nn.Linear(512, 512)
        self.relu_fc2 = nn.ReLU()
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, frame_f, frame_f1, altitude, yaw):
        # 입력이 5차원인 경우 4차원으로 reshape (e.g., [B, 1, C, H, W] -> [B, C, H, W])
        if len(frame_f.shape) == 5:
            frame_f = frame_f.view(frame_f.size(0), frame_f.size(2), frame_f.size(3), frame_f.size(4))  # extra dim 제거
        if len(frame_f1.shape) == 5:
            frame_f1 = frame_f1.view(frame_f1.size(0), frame_f1.size(2), frame_f1.size(3), frame_f1.size(4))
        if len(altitude.shape) == 5:
            altitude = altitude.view(altitude.size(0), altitude.size(2), altitude.size(3), altitude.size(4))
        if len(yaw.shape) == 5:
            yaw = yaw.view(yaw.size(0), yaw.size(2), yaw.size(3), yaw.size(4))

        # Process frames
        feat_f = self.fe_frame_f(frame_f)  # (B, 64, 16, 16)
        feat_f1 = self.fe_frame_f1(frame_f1)  # (B, 64, 16, 16)
        concat_frames = torch.cat([feat_f, feat_f1], dim=1)  # (B, 128, 16, 16)

        # Process altitude and yaw
        aux = torch.cat([altitude, yaw], dim=1)  # (B, 2, 32, 32)
        aux_pooled = self.avgpool(aux)  # (B, 2, 16, 16)

        # Concatenate all
        concat_all = torch.cat([concat_frames, aux_pooled], dim=1)  # (B, 130, 16, 16)

        # Further feature extraction
        feat_mid = self.fe_mid(concat_all)  # (B, 64, 8, 8)
        feat_final = self.fe_final(feat_mid)  # (B, 64, 4, 4)

        # Flatten and dense layers
        x = self.flatten(feat_final)  # (B, 1024)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.bn_fc1(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.bn_fc2(x)
        out = self.fc3(x)  # (B, 3)
        return out


# =========================================
# Config
# =========================================
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EMBED_DIM    = 128  # Siamese 임베딩 크기
MARGIN       = 1.0  # Triplet margin
CLIP_NORM    = 1.0
PATIENCE     = 10
IMAGE_SIZE   = 32 #224

# =========================================
# Normalization (from TRAIN labels)
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params.npy"))

# =========================================
# Transforms
# =========================================
train_tf = SeqTransform(image_size=IMAGE_SIZE)
val_tf   = SeqTransform(image_size=IMAGE_SIZE)

# =========================================
# Datasets – .npy 로드
# =========================================
train_dataset = SiameseVODataset(
    os.path.join(DATA_DIR, "train_pairs.npy"),
    os.path.join(DATA_DIR, "train_pair_labels.npy"),
    transform=train_tf,  label_stds=stds, label_means=means # 추가: train transform 전달
)
val_dataset = SiameseVODataset(
    os.path.join(DATA_DIR, "val_pairs.npy"),
    os.path.join(DATA_DIR, "val_pair_labels.npy"),
    transform=val_tf,  label_stds=stds, label_means=means # 추가: val transform 전달
)

# =========================================
# DataLoaders
# =========================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

## =========================================
# Model / Loss / Optimizer / Scheduler
# =========================================
model = siamVONet().to(DEVICE)
criterion = nn.SmoothL1Loss(beta=1.0)        # beta=0.1 → 1.0으로 증가 (더 안정적)

# Optimizer - LR 크게 낮추기
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,                    # 8e-4 → 5e-5 (대폭 낮춤)
    weight_decay=1e-4,          # 5e-4 → 1e-4 (조금 완화)
    betas=(0.9, 0.999)
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=True
)
scaler = GradScaler(enabled=torch.cuda.is_available())
# scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

best_val = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []

print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"  -> GPU Name: {torch.cuda.get_device_name(0)}")
# =========================================
# Training Loop
# =========================================
for epoch in range(1, NUM_EPOCHS + 1):
    # ---- Train ----
    model.train()
    running = 0.0
    for anchor, positive, altitude, yaw, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch:02d}"):
        anchor = anchor.to(DEVICE, non_blocking=True)
        positive = positive.to(DEVICE, non_blocking=True)
        altitude = altitude.to(DEVICE, non_blocking=True)
        yaw = yaw.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            out = model(anchor, positive, altitude, yaw)

            loss = criterion(out, labels[:,0:3])  # 전체 MAE


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * anchor.size(0)
    tr_loss = running / len(train_loader.dataset)


    # ---- Validation ----
    model.eval()
    v_sum = 0.0
    mae_tot_z = torch.zeros(3, dtype=torch.float64)
    mae_tot_dn = torch.zeros(3, dtype=torch.float64)

    with torch.no_grad():
        for anchor, positive, altitude, yaw, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch:02d}"):
            anchor = anchor.to(DEVICE, non_blocking=True)
            positive = positive.to(DEVICE, non_blocking=True)
            altitude = altitude.to(DEVICE, non_blocking=True)
            yaw = yaw.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=False)


            with autocast(enabled=torch.cuda.is_available()):
                out = model(anchor, positive, altitude, yaw)

                batch_loss = criterion(out, labels[:, 0:3])  # 전체 MAE


            v_sum += batch_loss.item() * anchor.size(0)
            # z-score MAE
            mae_tot_z += torch.abs(out - labels[:, 0:3]).sum(dim=0).cpu().double()
            # denormalized MAE (real units)
            outs_dn = denormalize_labels(out, stds, means)
            ys_dn = denormalize_labels(labels[:, 0:3], stds, means)
            mae_tot_dn += torch.from_numpy(np.abs(outs_dn - ys_dn).sum(axis=0)).double()

    va_loss = v_sum / len(val_loader.dataset)
    mae_z = (mae_tot_z / len(val_loader.dataset)).numpy()
    mae_dn = (mae_tot_dn / len(val_loader.dataset)).numpy()

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    scheduler.step(va_loss)

    print(f"[Epoch {epoch:02d}] Train={tr_loss:.6f} | Val={va_loss:.6f}  "
          f"| MAE_z (x,ry,x+dx)=({mae_z[0]:.4f}, {mae_z[1]:.4f}, {mae_z[2]:.4f})  "
          f"| MAE_dn=({mae_dn[0]:.4f}, {mae_dn[1]:.4f}, {mae_dn[2]:.4f})")

    # ---- Checkpoint / EarlyStopping ----
    if va_loss < best_val - 1e-6:
        best_val = va_loss
        epochs_no_improve = 0
        save_checkpoint_full(
            os.path.join(RESULT_DIR, "best.ckpt"),
            epoch, model, optimizer, best_val, scaler=scaler,
            config={
                "embed_dim": EMBED_DIM,
                "margin": MARGIN,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            }
        )
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model.pth"))
        print("  -> Best updated & saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improve {PATIENCE} epochs).")
            break
