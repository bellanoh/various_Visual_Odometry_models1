# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from dataset import PressSequenceDataset
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
model = siamVONet().to(DEVICE)
state_path = os.path.join(RESULT_DIR, "best_model.pth")
state_dict = torch.load(state_path, map_location=DEVICE)
model.load_state_dict(state_dict)

# =========================================
# Inference
# =========================================
# 테스트 루프 (Siamese 모델에 맞게 수정: test_loader가 anchor, positive, altitude, yaw, ys 형태로 가정)
preds_z = []
gts_z = []

with torch.no_grad():
    model.eval()  # 평가 모드 추가 (BatchNorm 등 안정화)
    for anchor, positive, altitude, yaw, ys in tqdm(test_loader, desc="[Test]"):
        anchor = anchor.to(DEVICE, non_blocking=True)
        positive = positive.to(DEVICE, non_blocking=True)
        altitude = altitude.to(DEVICE, non_blocking=True)
        yaw = yaw.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=False)  # ys (labels)도 to(DEVICE) 추가 (필요 시)

        outs = model(anchor, positive, altitude, yaw)
        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys.cpu().numpy())  # ys가 전체 labels라면 ys[:, :3].numpy()로 3개 컬럼만 사용

preds_z = np.concatenate(preds_z, axis=0)
gts_z = np.concatenate(gts_z, axis=0)

# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn   = denormalize_labels(gts_z,   stds, means)



