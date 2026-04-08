# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader

import time
import gc

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

from utils_siam2 import SeqTransform, save_checkpoint_full, denormalize_labels

# =========================================
# Load normalization parameters
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params.npy"))

# =========================================
# Test Dataset & DataLoader
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
# Load Model
# =========================================
model = siamVONet().to(DEVICE)
state_path = os.path.join(RESULT_DIR, "best_model.pth")
state_dict = torch.load(state_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Model loaded from {state_path}")

# =========================================
# Peak Memory & Time Measurement 준비
# =========================================
torch.cuda.reset_peak_memory_stats(DEVICE)
torch.cuda.empty_cache()
start_time = time.time()

# =========================================
# Inference
# =========================================
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
print(f"Inference completed! Pred shape: {preds_z.shape}, GT shape: {gts_z.shape}")

# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn = denormalize_labels(gts_z, stds, means)

