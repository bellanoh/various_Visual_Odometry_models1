# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from utils_siam import denormalize_labels, SeqTransform


class SiameseVODataset(torch.utils.data.Dataset):
    def __init__(self, pairs_file, labels_file, transform=None, label_stds=None, label_means=None):
        self.pairs = np.load(pairs_file)   # shape: [num_pairs, 2, H, W, C]
        self.labels = np.load(labels_file) # shape: [num_pairs, num_labels] ← delta_pose (예: 3 or 4 or 6)
        self.transform = transform
        self.label_stds = torch.tensor(label_stds, dtype=torch.float32)
        self.label_means = torch.tensor(label_means, dtype=torch.float32)

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
# Load normalization parameters
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params.npy"))

# =========================================
# Dataset / DataLoader (TEST)
# =========================================
test_tf = SeqTransform(image_size=IMAGE_SIZE)

test_dataset = SiameseVODataset(
    os.path.join(DATA_DIR, "test_pairs.npy"),
    os.path.join(DATA_DIR, "test_pair_labels.npy"),
    transform=test_tf,  label_stds=stds, label_means=means
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

# =========================================
# Load model
# =========================================
model = SiameseNet(backbone='resnet50', embed_dim=EMBED_DIM,  dropout_rate=0.0).to(DEVICE)
state_path = os.path.join(RESULT_DIR, "best_model.pth")
state_dict = torch.load(state_path, map_location=DEVICE)
model.load_state_dict(state_dict)

# =========================================
# Inference
# =========================================

preds_z = []
gts_z = []

with torch.no_grad():
    model.eval()
    for anchor, positive, ys in tqdm(test_loader, desc="[Test]"):
        anchor = anchor.to(DEVICE, non_blocking=True)
        positive = positive.to(DEVICE, non_blocking=True)

        outs = model(anchor, positive)

        # 단일 텐서 확인: tuple이면 첫 번째 요소 사용 (디버깅용, 나중 제거 가능)
        if isinstance(outs, tuple):
            outs = outs[0]  # out_p 추출

        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys.cpu().numpy())

preds_z = np.concatenate(preds_z, axis=0)
gts_z = np.concatenate(gts_z, axis=0)


# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn   = denormalize_labels(gts_z,   stds, means)

