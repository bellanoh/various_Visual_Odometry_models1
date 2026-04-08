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
from utils_siam import SeqTransform, save_checkpoint_full, denormalize_labels


# Siamese Dataset (.npy 로드)
# SiameseVODataset 클래스 수정 (transform 적용)
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

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)

        # delta_pose label (진짜 상대값)
        label = torch.tensor(self.labels[idx]).float()   # (num_labels,)

        # z-score 정규화 (선택사항이지만 추천)
        if self.label_means is not None and self.label_stds is not None:
            label = (label - self.label_means) / self.label_stds

        return anchor, positive, label   # ← negative 제거


# Siamese Model (timm EfficientViT 백본)
import timm
class SiameseNet(nn.Module):
    def __init__(self, backbone='resnet50', embed_dim=128, dropout_rate=0.2):  # dropout_rate 파라미터 추가 (기본 0.5)
        super().__init__()

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        # 핵심: 실제 출력 차원을 dummy input으로 확인
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            in_features = feats.shape[1] if feats.ndim == 2 else feats.shape[-1]

        print(f"[Info] Backbone output dim: {in_features}")  # 확인용

        self.fc = nn.Linear(in_features, embed_dim)  # 정확한 차원으로 fc 생성
        #self.dropout = nn.Dropout(dropout_rate)      # Dropout 레이어 추가 (임베딩 후 regularization)
        self.reg_head = nn.Linear(embed_dim, 3)      # 회귀 헤드

    def forward_one(self, x):
        if x.ndim == 5 and x.shape[1] == 1:  # [B, 1, C, H, W] → [B, C, H, W]
            x = x.squeeze(1)

        feats = self.backbone(x)
        emb = self.fc(feats)
        #emb = self.dropout(emb)  # Dropout 적용 (임베딩 후, 과적합 방지)
        return emb

    def forward(self, anchor, positive):
        emb_a = self.forward_one(anchor)
        emb_p = self.forward_one(positive)

        out_p = self.reg_head(emb_p)

        return out_p




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
IMAGE_SIZE   = 224

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
model = SiameseNet(backbone='resnet50', embed_dim=EMBED_DIM, dropout_rate=0.0).to(DEVICE)
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
    for anchor, positive, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch:02d}"):
        anchor = anchor.to(DEVICE, non_blocking=True)
        positive = positive.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            out_p = model(anchor, positive)
            # out_p = nn.Dropout(p=0.5)(out_p)
            loss =  criterion(out_p, labels[:,0:3])


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
        for anchor, positive, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch:02d}"):
            anchor = anchor.to(DEVICE, non_blocking=True)
            positive = positive.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=False)


            with autocast(enabled=torch.cuda.is_available()):
                out_p = model(anchor, positive)

                batch_loss = criterion(out_p, labels[:, 0:3])

            v_sum += batch_loss.item() * anchor.size(0)
            # z-score MAE
            mae_tot_z += torch.abs(out_p - labels[:, 0:3]).sum(dim=0).cpu().double()
            # denormalized MAE (real units)
            outs_dn = denormalize_labels(out_p, stds, means)
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
